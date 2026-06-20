# Eye-NAV: AI-Powered Navigation for the Visually Impaired 👁️

An AI-powered mobile assistant for the visually impaired, featuring real-time obstacle detection (YOLOv8), metric depth estimation, and an LLM-driven voice interface for enhanced safety and independence.

---

## 🏗️ System Architecture

Our system is built on a **"Decoupled Async Inference"** design. Independent threads for **Object Detection**, **Depth Estimation**, and **Path Analysis** feed real-time data into a non-blocking **LLM Controller** (Gemini 2.5 Flash). 

The mobile interface communicates via a high-speed **Binary WebSocket** pipeline, ensuring sub-500ms latency even over remote cellular connections.

## ✨ Key Features

-   **Binary WebSocket Streaming:** Zero-lag video transmission using raw binary multiplexing (Optimized for 4G/5G).
-   **Dual-Stage Vision:** Real-time **YOLOv8** (Nano to Large) recognition combined with **Depth-Anything-V2**.
-   **Smart Path Analysis:** Dynamic "walkable path" segmentation to keep users safe and centered on walkways.
-   **Low-Chatter TTS:** Smart de-duplication and cooldown to provide clear, professional navigation without audio overload.
-   **Multi-Device Web UI:** A premium Vanilla JS frontend accessible via any smartphone camera.
-   **Lighting Robustness (NEW):** Integrated CLAHE (Contrast-Limited Adaptive Histogram Equalization) and Gamma correction to maintain detection accuracy in harsh sunlight and indoor low-light, scientifically grounded by recent research (Bhandari et al., 2020; Zhao et al., 2024).

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **CUDA-enabled GPU** (Required for real-time performance)
- **Google AI Studio Key** (For the LLM guidance)
- **ngrok Account** (For mobile testing)

### 2. Setup Gemini (LLM)
1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Click on **"Get API key"** and create a new key.
3.  In the project root, create a `.env` file:
    ```env
    GOOGLE_API_KEY=YOUR_ACTUAL_KEY_HERE
    ```

### 3. Setup ngrok (Global Access)
If you haven't set up ngrok before, follow these steps to enable mobile testing:
1.  Sign up at [ngrok.com](https://ngrok.com/).
2.  Download the ngrok agent for Windows.
3.  Copy your **Authtoken** from the ngrok dashboard.
4.  Open a terminal (PowerShell) and run:
    ```powershell
    # Replace with your actual token
    ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
    ```
5.  *That's it!* The Eye-NAV backend will now handle the tunnel automatically.

### 4. Installation
```powershell
# 1. Clone & Enter
git clone https://github.com/stephintomson2152003/Assistive-Navigation-AI.git
cd Assistive-Navigation-AI

# 2. Setup Environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install Requirements
pip install -r requirements.txt
```

---

### 🛠️ Usage & Running

### Option A: Local Desktop Testing (Camera / Video)
Perfect for debugging the AI brain on your laptop using its built-in webcam or a saved video file. This runs **without** the web server or ngrok.

```powershell
# Run using your default laptop camera
python main_controller.py

# OR: Run on a saved video file
python main_controller.py path/to/your_video.mp4
```
*   **Controls:** Press `q` to quit the local windows.

---

### Option B: Full Web System (Mobile + AI + Server)
This is the main way to use Eye-NAV. It starts the AI brain, the web server, and the mobile tunnel simultaneously.

```powershell
cd src/UI/backend
python app.py --live --tunnel
```

#### 📱 How to Connect:
1.  **Local (PC):** Open `http://localhost:5050`.
2.  **Mobile (Phone):** Look for the `GLOBAL ACCESS` link in your terminal. Open this on your phone's browser.
3.  **Start:** Click **Start Navigation** to begin the live feed.

---

## 📂 Repository Structure

-   `src/main.py`: The `MainProcessor`. Orchestrates YOLO, Depth, and LLM threads.
-   `src/UI/backend/app.py`: The FastAPI server handling WebSockets and Tunnels.
-   `src/perception/`: The "Eyes" (Detection, Depth, Path Analysis).
-   `src/nlp/`: The "Voice" (LLM Prompting and Logic).
-   `config.json`: Hardware & AI configuration (Switch models here).


## References

-   Patel, K., & Parmar, B. (2022). Assistive device using computer vision and image processing for visually impaired; review and current status. Disability and Rehabilitation: Assistive Technology, 17(3), 290–297. https://doi.org/10.1080/17483107.2020.1786731
-   Voutsakelis, G., Dimkaros, I., Tzimos, N., Kokkonis, G., & Kontogiannis, S. (2025). Development and Evaluation of a Tool for Blind Users Utilizing AI Object Detection and Haptic Feedback. Machines, 13(5), 398. https://doi.org/10.3390/machines13050398
-   Kadhim, M., & Oleiwi, B. (2022). Blind Assistive System based on Real Time Object Recognition using Machine learning. Engineering and Technology Journal, 40(1), 159-165. https://doi.org/10.30684/etj.v40i1.1933
-   Bhandari, A., Kafle, A., Dhakal, P., Joshi, P. R., & Kshatri, D. B. (2020). Image Enhancement and Object Recognition for Night Vision Surveillance. arXiv:2006.05787 [cs.CV]. https://doi.org/10.48550/arXiv.2006.05787
-   Zhao, D., Shao, F., Zhang, S., Yang, L., Zhang, H., Liu, S., & Liu, Q. (2024). Advanced Object Detection in Low-Light Conditions: Enhancements to YOLOv7 Framework. Remote Sensing, 16(23), 4493. https://doi.org/10.3390/rs16234493
