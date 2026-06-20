# Eye-NAV – UI Module

This folder contains the complete standalone UI for **Eye-NAV** — a phone-first web interface for the AI-powered assistive navigation system.

## Structure

```
ui/
├── backend/                 ← Flask REST API
│   ├── app.py               ← Main Flask server (all endpoints)
│   ├── data_source.py       ← Data abstraction layer (mock ↔ live switch)
│   ├── settings_store.json  ← Auto-created; persisted user settings
│   ├── requirements.txt     ← flask, flask-cors
│   └── mock_data/           ← Fake data files (no main_controller needed)
│       ├── fused_state.json
│       ├── nav_state.json
│       ├── llm_response.json
│       ├── depth_state.json
│       └── system_status.json
├── frontend/                ← Vanilla HTML/CSS/JS mobile app
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
│       └── logo.png
└── run.py                   ← Quick launcher
```

---

## How to Run Locally

### 1. Install dependencies
```powershell
cd ui/backend
pip install -r requirements.txt
```

### 2. Start the server
```powershell
# From project root:
python ui/run.py

# OR from ui/backend directly:
cd ui/backend
python app.py
```

The server starts at **http://localhost:5050**

### 3. Open in browser (phone view)
Open **http://localhost:5050** in Chrome or Edge, then:
- Press **F12** → Toggle device toolbar → Select **Samsung Galaxy S8+** or similar portrait phone size
- The UI will show the splash screen and auto-navigate to Home

---

## API Endpoints

All endpoints return JSON. Base URL: `http://localhost:5050`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | System state: running flag, fps, active modules, uptime |
| `GET` | `/api/scene` | Latest fused AI scene: detected objects with labels, distances, positions, and nav instruction. **This is the main output of `main_controller._save_fused_state()`** |
| `GET` | `/api/depth` | Depth estimation stats: min/max/mean distance, closest object, risk level |
| `GET` | `/api/nav` | GPS navigation state: heading, current & next instruction, ETA, full route |
| `GET` | `/api/llm` | Latest LLM guidance response |
| `GET` | `/api/settings` | Read current settings |
| `POST` | `/api/settings` | Save settings. Body: `{"gps_navigation": false, "developer_mode": true}` |
| `POST` | `/api/command` | Send command. Body: `{"action": "start", "destination": "..."}` \| `{"action": "stop"}` \| `{"action": "query", "text": "..."}` |
| `GET` | `/api/health` | Quick health check |
| `GET` | `/` | Serves the frontend |

### Example requests
```bash
# Check health
curl http://localhost:5050/api/health

# Get current scene
curl http://localhost:5050/api/scene

# Start navigation
curl -X POST http://localhost:5050/api/command \
     -H "Content-Type: application/json" \
     -d '{"action":"start","destination":"Main Library"}'

# Save a setting
curl -X POST http://localhost:5050/api/settings \
     -H "Content-Type: application/json" \
     -d '{"developer_mode":true}'
```

---

## Connecting to `main_controller.py`

### Current setup (Mock mode)
The backend reads from `ui/backend/mock_data/*.json` files. Everything works with zero dependency on the main system.

### Step 1 – File mode (easiest integration)

`main_controller.py` already writes a live singleton file at:
```
data/states/fused_state.json
```
on every frame via `_save_fused_state()`.

**To use it**, just set one environment variable before starting the backend:

```powershell
# PowerShell
$env:DATA_SOURCE = "file"
python ui/run.py
```

```bash
# Linux / macOS
DATA_SOURCE=file python ui/run.py
```

The `data_source.py` will automatically read from:
```
data/states/fused_state.json    ← main_controller output  ✓
data/states/nav_state.json      ← future nav module output
data/states/llm_response.json   ← future LLM module output
data/states/depth_state.json    ← future depth module output
data/states/system_status.json  ← future status output
```

> **No code changes are needed.** Just the ENV variable.

### Step 2 – Running both together

```powershell
# Terminal 1 – main controller (writes fused_state.json)
python main_controller.py

# Terminal 2 – UI backend (reads fused_state.json every 2s via frontend poll)
$env:DATA_SOURCE = "file"
python ui/run.py
```

### Step 3 – Direct integration (future / optional)

When you are ready for tighter coupling, edit the `"start"` handler in `backend/app.py`:

```python
# In api_command(), action == "start":
# Replace the comment with:
from main_controller import MainController
controller = MainController()
threading.Thread(target=controller.start, daemon=True).start()
```

And similarly use a shared `queue.Queue` for the `/api/command` → main_controller bridge.

The `data_source.py` `"live"` mode is reserved for this use case (direct in-process data sharing).

---

## Data Schemas

### `/api/scene` (mirrors `main_controller._save_fused_state`)
```json
{
  "timestamp": 1743320430.5,
  "count": 4,
  "navigation_instruction": "Clear path ahead, continue straight",
  "objects": [
    {
      "label": "person",
      "confidence": 0.92,
      "position": ["middle-center"],
      "distance": "2.3m",
      "color": "blue jacket",
      "timestamp": 1743320430.1
    }
  ]
}
```

### `/api/settings`
```json
{
  "companion_mode": false,
  "gps_navigation": true,
  "distance_estimation": true,
  "descriptive_mode": false,
  "developer_mode": false
}
```

### `POST /api/command`
```json
{ "action": "start", "destination": "Main Library" }
{ "action": "stop" }
{ "action": "query", "text": "What is around me?" }
```

---

## Screens

| Screen | How to reach | Purpose |
|--------|-------------|---------|
| **Splash** | Auto (app load) | Branding, 2 s then → Home |
| **Home** | After splash / after stop | Set destination, tap eye to start |
| **Running** | After tapping start | Audio/haptic toggles, nav instruction, tap to stop |
| **Developer Mode** | Running screen (DEV button, when enabled in Settings) | Live detections, depth bar, LLM output |
| **Settings** | Home → Settings | Toggle features, save/discard |

Developer Mode button only appears on the Running screen when **Developer Mode** is enabled in Settings.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, Flask 2.3+, flask-cors |
| Frontend | Vanilla HTML5 / CSS3 / ES6 JavaScript |
| Fonts | Google Fonts – Inter |
| Data source | JSON files (mock or live from main_controller) |
| Polling | `setInterval` every 2 s (frontend → backend) |

---

## Team

- [Stephin Tomson](https://github.com/stephintomson2152003)
- [Hidoyat Ruzmetov](https://github.com/HidoyatRuzmetov)
- [Chitrangada Juneja](https://github.com/chitrangada-juneja)
