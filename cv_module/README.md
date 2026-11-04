# interview_cv – computer-vision module for the AI mock interview

Analyses the **live webcam feed** during a mock interview and produces timestamped
observations plus a cautious feedback report:

| Feature | How | Output |
|---|---|---|
| Face presence + landmarks | MediaPipe Face Landmarker (478 landmarks, head transform) | `FaceObservation` (present, bbox, head yaw/pitch/roll) |
| Facial expression | HSEmotion EfficientNet-B0 (8 AffectNet classes) via ONNX Runtime | `ExpressionObservation` (label, broad group, probabilities) |
| Approximate gaze | head pose + iris position inside the eye opening | `GazeObservation` (camera / away + direction) |
| Posture | MediaPipe Pose Landmarker (shoulders, nose) | `PostureObservation` (tilt, head offset/height, flags) |
| Session + events | `CVSession` tags observations with question id / phase | per-question summaries in the report |
| Report | aggregation of the observation log | `CVReport` (metrics, heuristic scores, strengths, suggestions) |

Live webcam is the only video input. There is no file/upload processing.
Design notes, the event contract and deviations: [docs/CV_DESIGN.md](docs/CV_DESIGN.md).

## Setup

Python 3.10–3.12 (3.11 is what was tested). From this directory:

```powershell
# Option A: uv (recommended; creates .venv and installs the locked versions)
uv sync --extra dev

# Option B: plain pip
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # pinned runtime + test deps
pip install -e .

# Download the three model files (~25 MB) into cv_module/models and the sample test image
interview-cv download-models --samples
```

Model files are not committed. They go to `cv_module/models/` by default; override
with `--model-dir` or the `INTERVIEW_CV_MODEL_DIR` environment variable.

## Run (manual webcam test)

```powershell
# 60-second webcam session with a preview window (q / Esc or Ctrl+C stops early)
interview-cv run --camera 0 --max-seconds 60 --show --report report.json --log observations.jsonl

# Rebuild a report from a saved observation log
interview-cv report --log observations.jsonl
```

`python -m interview_cv ...` works the same as `interview-cv ...`.
Useful flags: `--sample-fps 5` (analysis rate), `--record debug.mp4` (annotated feed),
`--no-expression`, `--no-posture`, `--no-gaze`, `-v`.

The webcam path only wraps `cv2.VideoCapture`; it has **not** been exercised on real
hardware in this repository yet (the development machine had no working camera).

## Use from the interview application

```python
from interview_cv import CVSession, InterviewEvent, PipelineConfig

session = CVSession(session_id="a3f9...", config=PipelineConfig(sample_fps=5.0), camera=0)
session.start()                                   # opens the webcam, starts the capture thread

# forward lifecycle events as the interview progresses (see docs/CV_DESIGN.md §3 for the contract)
session.handle_event(InterviewEvent("interview_started", "a3f9...", time.time()))
session.handle_event(InterviewEvent("question_started", "a3f9...", time.time(), question_id="q1"))
session.handle_event(InterviewEvent("answer_started",   "a3f9...", time.time(), question_id="q1"))
session.handle_event(InterviewEvent("answer_ended",     "a3f9...", time.time(), question_id="q1"))
session.handle_event(InterviewEvent("interview_ended",  "a3f9...", time.time()))   # also stops the camera

report = session.build_report()                   # CVReport, JSON-serialisable
session.log.save_jsonl("observations.jsonl")
session.close()
```

Events may also be passed as plain dicts with the same keys. If the host owns the
camera, construct with `camera=None` and call `session.process_frame(frame_bgr)` from
its own loop.

## Tests

```powershell
pytest            # 49 unit tests always run; 5 integration tests run when models + sample image exist
pytest -m "not integration"
```

No test needs a webcam or a video file. Integration tests feed in-memory frames
synthesised from the sample photo and are skipped automatically when the model files
or the sample are missing.

## Layout

```
cv_module/
  src/interview_cv/
    config.py          thresholds, model paths, label groups
    schemas.py         observation / event / report dataclasses (JSON round-trip)
    session.py         CVSession: webcam lifecycle, events, log, report
    video/source.py    WebcamVideoSource, FrameSampler, FrameSequenceSource (tests), VideoWriter (debug)
    analyzers/
      face.py          FaceLandmarker wrapper, head pose from transform matrix
      gaze.py          camera-vs-away geometry (pure numpy)
      expression.py    HSEmotion ONNX classifier + face crop
      posture.py       PoseLandmarker wrapper, posture geometry (pure numpy)
    pipeline.py        runs analyzers on one frame -> FrameObservation
    report.py          metrics, scores, suggestions, per-question summaries -> CVReport
    visualize.py       optional overlay drawing
    samples.py         in-memory synthetic frames for tests
    cli.py             command line
  tests/
  models/              downloaded model files (gitignored)
  samples/             downloaded sample image (gitignored)
  docs/CV_DESIGN.md
```

## Licensing

MediaPipe (code + models) and the HSEmotion code are Apache-2.0. The HSEmotion
weights were trained on AffectNet, whose licence is research-only; review this
before any commercial deployment. The sample image is the one Google hosts for
its MediaPipe example notebooks and is only downloaded on demand, never committed.
