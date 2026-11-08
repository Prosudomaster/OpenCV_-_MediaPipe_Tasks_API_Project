# 🖥️ Real-Time Face Mesh Studio (OpenCV & MediaPipe Tasks API)

A professional, modular desktop application built with Python that combines **OpenCV** for video stream management, **Google MediaPipe Tasks API** for high-performance facial landmark detection, and **CustomTkinter** for a modern, dark-themed user interface.

---

## 🚀 Key Features

* **Modular Architecture:** Clean separation of concerns with a dedicated backend processor (`face_processor.py`) and a sleek graphical user interface (`app_ui.py`).
* **MediaPipe Tasks API:** Utilizes the latest `FaceLandmarker` model running in video mode for optimal real-time performance.
* **Modern Desktop UI:** Built using `CustomTkinter` featuring a responsive layout, live video feed container, and intuitive Start/Stop camera controls.
* **Tessellation Rendering:** Real-time 3D landmark tracking and facial mesh connection mapping.

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **AI Framework:** Google MediaPipe Tasks API (`mediapipe`)
* **GUI Framework:** CustomTkinter & Pillow (`PIL`)

---

## 📁 Project Structure

```text
OpenCV_-_MediaPipe_Tasks_API_Project/
│
├── models/
│   └── face_landmarker.task     # MediaPipe pre-trained model asset
│
├── face_processor.py            # Backend logic for face mesh detection & drawing
├── app_ui.py                    # CustomTkinter GUI application and event loop
├── sample.ipynb                 # Jupyter notebook for experimentation
└── README.md                    # Project documentation