"""
Eye-NAV Backend – FastAPI REST API (v3)
======================================
Run:  python app.py          (mock/simulator mode)
      python app.py --live   (live AI processor mode)
"""

import os
import json
import time
import threading
import sys
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
# from data_source import DataSource (DELETED)
import numpy as np
import cv2

# Add project root to sys.path to allow importing from src
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent.parent  # backend → UI → src → project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

_processor = None

app = FastAPI(title="Eye-NAV Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.environ.get("PORT", 5050))
_FRONTEND_DIR = _HERE.parent / "frontend"
_SETTINGS_FILE = _HERE / "settings_store.json"

# ── Session state ──────────────────────────────────────────────────────────────
_session = {"running": False, "start_time": None, "destination": ""}
_session_lock = threading.Lock()

# Serialises process_frame calls — YOLO/GPU state is NOT thread-safe.
_frame_lock = threading.Lock()

_DEFAULT_SETTINGS = {
    "companion_mode": False,
    "distance_estimation": True,
    "descriptive_mode": False,
    "developer_mode": False,
}

def _load_settings() -> dict:
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE) as f:
                return {**_DEFAULT_SETTINGS, **json.load(f)}
        except Exception:
            pass
    return dict(_DEFAULT_SETTINGS)

def _save_settings(s: dict):
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/scene")
def api_scene():
    return DataSource.get_scene()

@app.get("/api/depth")
def api_depth():
    return DataSource.get_depth()

@app.get("/api/status")
def api_status():
    data = DataSource.get_status()
    with _session_lock:
        running = _session["running"]
        uptime = (
            round(time.time() - _session["start_time"], 1)
            if running and _session["start_time"]
            else 0
        )
    data.update(
        {
            "session_running": running,
            "session_uptime_seconds": uptime,
            "data_source_mode": DataSource.source_mode(),
        }
    )
    return data

@app.get("/api/settings")
def get_settings():
    return _load_settings()

@app.post("/api/settings")
async def post_settings(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    
    if not isinstance(body, dict):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        
    cur = _load_settings()
    unknown = [k for k in body if k not in cur]
    if unknown:
        return JSONResponse({"error": f"Unknown keys: {unknown}"}, status_code=400)
        
    cur.update(body)
    _save_settings(cur)
    
    # Sync with live processor if it's active
    global _processor
    if _processor:
        try:
            _processor.update_preferences(cur)
        except Exception as e:
            print(f"[Backend] Failed to sync settings to processor: {e}")

    return {"status": "saved", "settings": cur}

@app.post("/api/command")
async def api_command(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    if not body or "action" not in body:
        return JSONResponse({"error": "Missing 'action'"}, status_code=400)
    action = body["action"]

    if action == "start":
        with _session_lock:
            _session["running"] = True
            _session["start_time"] = time.time()
            _session["destination"] = body.get("destination")
        return {"status": "started"}

    elif action == "stop":
        with _session_lock:
            _session["running"] = False
            _session["start_time"] = None
        return {"status": "stopped"}

    elif action == "query":
        response = "System processing..."
        if _processor:
            with _processor._latest_state_lock:
                response = _processor._latest_state.get("llm_response")
        return {"status": "queried", "response": response}

    return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

@app.post("/api/process_frame")
async def process_frame(request: Request):
    """
    Direct frame processing endpoint. 
    Receives binary JPEG data asynchronously, returns visual AI results.
    """
    global _processor
    if _processor is None:
        try:
            from src.main import MainProcessor
            _processor = MainProcessor(show_local=True)
            _processor.update_preferences(_load_settings())
        except Exception as e:
            return JSONResponse({"error": f"Failed to initialize MainProcessor: {e}"}, status_code=500)

    try:
        img_data = await request.body()
    except Exception:
        return JSONResponse({"error": "Error reading request body"}, status_code=400)
        
    if not img_data:
        return JSONResponse({"error": "No image data provided"}, status_code=400)

    try:
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)
        
        with _frame_lock:
            results = _processor.process_frame(frame)
        return results
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {e}"}, status_code=500)

@app.websocket("/api/ws_stream")
async def ws_stream(websocket: WebSocket):
    """
    Persistent WebSocket connection for high-speed, lag-free video streaming.
    Receives JPEG binaries, yields JSON processed results.
    """
    await websocket.accept()
    
    global _processor
    if _processor is None:
        try:
            from src.main import MainProcessor
            _processor = MainProcessor(show_local=True)
            _processor.update_preferences(_load_settings())
        except Exception as e:
            await websocket.close(reason=f"Failed to init processor: {e}")
            return

    try:
        while True:
            # Use general receive to handle both Bytes (images) and Text (logs/commands)
            message = await websocket.receive()
            
            if "bytes" in message:
                img_data = message["bytes"]
                if not img_data: continue

                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with _frame_lock:
                        results = _processor.process_frame(frame)
                    
                    if results and "error" not in results:
                        img_b64 = results.pop("render", None)
                        results["has_render"] = bool(img_b64)
                        await websocket.send_json(results)
                        
                        if img_b64:
                            import base64
                            raw_bytes = base64.b64decode(img_b64)
                            await websocket.send_bytes(raw_bytes)
                else:
                    await websocket.send_json({"error": "Failed to decode frame"})
            
            elif "text" in message:
                import json
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "tts_log":
                        evt = data.get("event", "LOG")
                        txt = data.get("text", "")
                        # Print with green color to distinguish from server logs
                        print(f"\033[92m[DEVICE]\033[0m -> {evt}: \"{txt}\"")
                except:
                    pass

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")

@app.get("/api/health")
def health():
    return {"status": "ok", "port": PORT, "mode": "live" if _processor else "init"}

# ── Static / Frontend ──────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Eye-NAV Backend")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start MainProcessor immediately and show OpenCV windows on server.",
    )
    args = parser.parse_args()

    if args.live:
        print("[Eye-NAV] --live mode: initialising MainProcessor...")
        from src.main import MainProcessor
        _processor = MainProcessor(
            config_path=str(_PROJECT_ROOT / "config.json"),
            show_local=True
        )
        _processor.update_preferences(_load_settings())
        print("[Eye-NAV] MainProcessor ready.")

    print(f"[Eye-NAV] Local access: http://localhost:{PORT}")
    if args.live:
        # Run Uvicorn in a background thread 
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
        uvicorn_server = uvicorn.Server(uvicorn_config)
        uvicorn_thread = threading.Thread(target=uvicorn_server.run, daemon=True)
        uvicorn_thread.start()
        print("[Eye-NAV] FastAPI running in background thread.")
        print("[Eye-NAV] OpenCV windows active. Press 'q' in a window to quit.")

        import cv2 as _cv2
        while _processor.running:
            payload = None
            try:
                payload = _processor.display_queue.get_nowait()
            except Exception:
                pass

            if payload is not None:
                _cv2.imshow("Eye-NAV Server Feed", payload["main"])
                if payload["debug"] is not None:
                    _cv2.imshow("Path Analysis [Debug]", payload["debug"])

            key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                _processor.stop()
                break

        _cv2.destroyAllWindows()
        print("[Eye-NAV] Shutting down.")

    else:
        print("[Eye-NAV] Starting in API-only mode (No local OpenCV windows).")
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
