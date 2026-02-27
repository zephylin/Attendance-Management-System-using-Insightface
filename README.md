# Attendance Management System using InsightFace

![CI Pipeline](https://github.com/zephylin/Attendance-Management-System-using-Insightface/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

A real-time attendance management system powered by **InsightFace** deep learning models for face detection and recognition. Built with **Streamlit** for the web interface, **Redis** for fast data storage, and containerized with **Docker** for easy deployment.

---

## Architecture

```
┌──────────────┐    WebRTC     ┌───────────────────┐    Cosine     ┌─────────────┐
│   Webcam /   │──────────────▸│  InsightFace Model │──Similarity──▸│  ML Search  │
│   Browser    │   Video Feed  │  (buffalo_sc)      │  Matching     │  Algorithm  │
└──────────────┘               └───────────────────┘               └──────┬──────┘
                                                                          │
                               ┌───────────────────┐               ┌──────▼──────┐
                               │   Streamlit UI     │◂──────────── │   Redis DB  │
                               │  (Multi-page App)  │   Read/Write │  (Cloud)    │
                               └───────────────────┘               └─────────────┘
```

**Key Components:**
- **Face Detection & Recognition:** InsightFace `buffalo_sc` model (ONNX) — detects faces and extracts 512-dimensional embeddings
- **Matching Algorithm:** Cosine similarity search against registered face embeddings
- **Data Layer:** Redis (cloud-hosted) for storing face embeddings and attendance logs
- **Frontend:** Streamlit multi-page app with WebRTC for real-time video streaming

---

## Features

- **Real-Time Face Recognition** — Live webcam feed with bounding boxes and identity labels
- **User Registration** — Capture face embeddings via webcam and store in Redis
- **Attendance Reporting** — View today's attendance, full logs, and registered users
- **Secure Configuration** — Credentials managed via environment variables (`.env`)
- **Dockerized** — One-command deployment with Docker Compose
- **CI/CD** — Automated testing and linting on every push via GitHub Actions
- **Tested** — 16 unit tests covering core logic with mocked dependencies

---

## Quick Start

### Prerequisites

- Python 3.11+
- Redis database ([Redis Cloud](https://redis.com/cloud/) free tier works)
- Webcam (for face recognition features)

### 1. Clone & Install

```bash
git clone https://github.com/zephylin/Attendance-Management-System-using-Insightface.git
cd Attendance-Management-System-using-Insightface
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and fill in your Redis credentials:

```bash
cp .env.example .env
# Edit .env with your Redis host, port, and password
```

### 3. Add Model Files

Download the InsightFace `buffalo_sc` model and place the ONNX files in:

```
insightface_model/buffalo_sc/
├── det_500m.onnx
└── w600k_mbf.onnx
```

### 4. Run the App

```bash
streamlit run Home.py
```

### Docker (Alternative)

```bash
# Build and run with Docker Compose
docker compose up --build

# Access at http://localhost:8501
```

---

## Usage

| Page | Description |
|------|-------------|
| **Home** | Landing page — loads models and verifies Redis connection |
| **Real-Time Prediction** | Start webcam, detect faces, and log attendance automatically |
| **Registration Form** | Enter student ID, name, and country, then capture face samples |
| **Report** | View registered users, raw logs, and today's attendance sheet |

---

## Screenshots

### Real-Time Prediction
<img src="https://github.com/user-attachments/assets/aa0131ed-b886-46aa-a721-543d6b846793" alt="Real-time face recognition with bounding boxes" width="600"/>

### User Registration
<img src="https://github.com/user-attachments/assets/677f138a-40ee-4791-910f-785c91ca8027" alt="Registration form with face capture" width="600"/>

### Attendance Report
<img src="https://github.com/user-attachments/assets/ada2566b-6e1e-4ae1-844f-c214412244f7" alt="Attendance reporting dashboard" width="600"/>

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output: 16 passed
```

Tests cover:
- ML search algorithm (exact match, no match, threshold boundary, return types)
- Real-time prediction log management (init, reset, save, unknown filtering)
- Registration form validation (empty/null inputs, missing files, state management)

---

## Project Structure

```
├── .env.example              # Environment variable template
├── .github/workflows/ci.yml  # CI/CD pipeline (lint + test + Docker build)
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # One-command deployment
├── Home.py                   # Streamlit entry point
├── face_rec.py               # Core logic: face detection, matching, registration
├── requirements.txt          # Python dependencies
├── tests/
│   └── test_face_rec.py      # Unit tests (16 tests)
├── pages/
│   ├── 1_Real_Time_Prediction.py
│   ├── 2_Registration_form.py
│   └── 3_Report.py
└── insightface_model/        # ONNX model files (not in repo — see setup)
    └── buffalo_sc/
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Face Recognition | InsightFace (buffalo_sc), ONNX Runtime |
| ML/Math | NumPy, Scikit-learn (cosine similarity) |
| Web Framework | Streamlit, Streamlit-WebRTC |
| Database | Redis (cloud) |
| Computer Vision | OpenCV |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest, unittest.mock |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.
