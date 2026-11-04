# AI Career Preparation Platform

An AI-powered platform to help users practise job interviews and receive useful feedback.

## Vision

The AI mock interview flow: the AI generates a question, the audio module reads it out
(text-to-speech), the user answers through the microphone, the audio module captures and
processes the response, and the CV module independently analyses the live webcam feed.
Answer evaluation, audio analysis and CV observations are then combined into one report.

## Current status

Only the **CV module** is implemented so far. Audio (TTS, microphone, speech analysis) is
owned by another team; question generation, answer evaluation, authentication and the
web app are planned but not started.

## CV module (`cv_module/`)

Python package `interview_cv` that analyses the **live webcam feed** during an interview
and produces a feedback report. There is no video-file or upload processing.

What it does:

- **Face detection and landmarks** – MediaPipe Face Landmarker; face presence, bounding box, head pose.
- **Facial expression** – HSEmotion model running on ONNX Runtime; 8 raw classes grouped into neutral / positive / other.
- **Approximate gaze** – head pose plus iris position to tell camera-facing from looking away.
- **Posture** – MediaPipe Pose Landmarker; shoulder tilt, head position, distance changes versus a session baseline.
- **Session manager** – `CVSession` starts/stops the webcam, records timestamped observations and accepts interview lifecycle events (`interview_started`, `question_started`, `answer_started`, `answer_ended`, `interview_ended`) so observations are tied to questions.
- **Feedback report** – session-wide and per-question metrics, heuristic scores, strengths and cautious suggestions. It never infers confidence, honesty or emotions.

Tests use in-memory synthetic frames, so no webcam is needed to run them. The webcam
path exists but has not been tested on hardware yet.

Quick start:

```powershell
cd cv_module
uv sync --extra dev                              # or: pip install -r requirements.txt && pip install -e .
interview-cv download-models --samples           # fetches ~25 MB of model files (not committed)
interview-cv run --camera 0 --max-seconds 60 --show --report report.json
pytest                                           # 54 tests, no webcam required
```

Details: [cv_module/README.md](../cv_module/README.md) (setup, usage, layout) and
[cv_module/docs/CV_DESIGN.md](../cv_module/docs/CV_DESIGN.md) (architecture, event contract, design decisions).

## Tech stack

Python 3.11, MediaPipe, ONNX Runtime, OpenCV, NumPy, pytest.
