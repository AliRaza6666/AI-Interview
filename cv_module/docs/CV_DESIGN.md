# CV module – assessment, recommendations and design decisions

First written 2026-11-04 for the AI Career Preparation Platform team; revised the same
day for the **webcam-only** architecture and the planned audio-team integration.
Sections 1–3 are the assessment and the proposed cross-module contract; section 4
records what was built and where it deviates from the original feature list, so each
deviation can be accepted or reverted.

## 1. Scope and constraints

* The CV module uses the **live webcam only**. All prerecorded-video / file-upload code
  was removed (see 4.1). Reports are produced from live sessions (or from a saved
  observation log of a live session, never from a video file).
* Another team owns audio (TTS, microphone capture, speech-to-text, speech analysis)
  and question generation. This module implements none of that and does not import
  their code; the only coupling is the event contract in section 3.
* Development machine: Windows 11, no working webcam, Python 3.11 managed by `uv`.
  Verification therefore used in-memory synthetic frames; the webcam path is
  implemented but **untested on hardware**.
* Library compatibility on Python 3.11 / Windows verified: MediaPipe 1.2.2 (Tasks API),
  ONNX Runtime 1.31, OpenCV 5.0, NumPy 2.4.

## 2. Architecture

```
host interview app ──events──▶ CVSession ──frames──▶ InterviewCVPipeline ──▶ FrameObservation
        ▲                        │  ▲                    (face, expression,            │
        │                        │  └── WebcamVideoSource     gaze, posture)            ▼
        └────── CVReport ◀────── build_report ◀──────────────────── ObservationLog
```

* **`WebcamVideoSource`** (`video/source.py`): open / read / close around
  `cv2.VideoCapture`; monotonic timestamps since the first frame. `FrameSampler`
  throttles to the analysis rate (default 5 fps). `FrameSequenceSource` replays
  in-memory frames for tests. Capture is fully separate from processing.
* **`InterviewCVPipeline`** (`pipeline.py`): one BGR frame + timestamp in, one
  `FrameObservation` out. Analyzers are independent and individually switchable; a
  failing analyzer leaves its field `None` and records the error for that frame.
* **`CVSession`** (`session.py`): the session manager. Owns the camera, the pipeline,
  the `ObservationLog`, the session id and the current interview context
  (`phase`, `question_id`). `start()` opens the camera and runs a daemon capture
  thread; `handle_event()` updates the context; `stop()` releases the camera and
  finalizes; `build_report()` aggregates. Construct with `camera=None` for a
  host-driven mode where the host calls `process_frame()` itself.
* **`build_report`** (`report.py`): session-wide metrics, heuristic scores,
  strengths, suggestions with evidence, and per-question summaries over answer
  segments.

Where it runs: in-process in the Python host application (or in a worker it spawns).
No web framework is imported, so wrapping it in an HTTP/websocket service later is a
packaging decision, not a code change.

## 3. Proposed cross-module contract (for both teams to review)

### 3.1 Shared identifiers and time base

* `session_id` (string): generated once per interview by the host application and
  passed to every module. The CV module rejects events carrying a different id.
* Time base: **UTC epoch seconds as a float** (`time.time()`) on every event. Each
  module keeps its own clock, so one shared wall-clock base lets audio and CV
  observations be aligned offline without shared process state. The CV log stores
  `started_at_epoch` plus session-relative `timestamp_s` per observation.
* `question_id` (string): assigned by whoever generates questions; opaque to CV.

### 3.2 Lifecycle events

```json
{ "type": "question_started", "session_id": "a3f9…", "timestamp": 1762250000.125,
  "question_id": "q1", "payload": {} }
```

| type | question_id | CV effect |
|---|---|---|
| `interview_started` | – | phase → `idle` |
| `question_started` | required | phase → `question` (question is being presented) |
| `answer_started` | required | phase → `answer` |
| `answer_ended` | required | phase → `idle` |
| `interview_ended` | – | phase → `idle`, camera released, log finalized |

`payload` is optional free-form data (e.g. question text) that CV ignores.
Implemented as `InterviewEvent` in `schemas.py`; the session accepts dataclasses or
plain dicts. If no events are ever sent, the session still works and the report is
session-wide only.

**How events are applied today.** The CV session tags each frame with the phase and
question id that are current *at the moment the frame is captured* (snapshotted before
analysis). The event's own `timestamp` is stored in the log and used for the per-question
answer window, but it is **not** used to re-label frames retroactively.

**Decision needed from the audio team (not implemented yet).** If `answer_ended` (or
`answer_started`) is detected asynchronously, e.g. by voice-activity detection that only
fires after a silence timeout, the event will arrive later than the moment it describes
and the frames in that gap will be tagged with the wrong segment. Two options:

1. The audio team sends events promptly and sets `timestamp` to the *occurrence* time
   (not the send time); CV keeps tagging at arrival. Simplest; good enough if the delay
   is well under one second.
2. CV adds a re-tagging pass at report time that re-labels frames from the event
   timestamps (`started_at_epoch` makes this a pure offline calculation). Robust to any
   delivery latency; requires the occurrence-time semantics above.

Either way, the contract must state that `timestamp` is when the thing happened.

### 3.3 Observation and report schema changes

* `FrameObservation` gained `question_id` (optional) and `phase`
  (`idle` | `question` | `answer`). Old logs without these fields still load.
* `SessionMeta` gained `session_id` and `started_at_epoch`.
* `ObservationLog` gained `events`, so a report can be rebuilt offline from the log.
* `CVReport` gained `questions`: one `QuestionSummary` per question over its answer
  frames (duration, frames, face presence, camera focus, dominant expression group,
  posture flag rates). Session-wide metrics and scores are unchanged.

### 3.4 Optional extensions (not implemented)

* `question_ended` event if "question being read aloud" and "waiting for the user"
  need to be distinguished.
* CV → host status events (e.g. `face_lost`) for live UI prompts.
* An HTTP/websocket wrapper once the host application exists.

## 4. Implementation notes and deviations

### 4.1 Removed for webcam-only operation

`FileVideoSource`, `open_source` (file branch), the `analyze --source <file>` and
`make-sample` CLI commands, file-writing sample generators, `pipeline.run(source)`,
and the file-based tests. Kept: `VideoWriter`, used only for an optional debug
recording of the annotated live feed (`interview-cv run --record`); say the word and
it goes too.

### 4.2 Preserved features and earlier decisions

* **HSEmotion** weights are loaded directly with ONNX Runtime rather than through the
  `hsemotion-onnx` package (unused `onnx` dependency, hard-coded cache dir, no face
  cropping). Preprocessing is identical.
* **Face crop** for expression: square box extended 25 % upward (forehead) with a 30 %
  margin, matching HSEmotion's detector-style training crops. Chosen from two still
  images; validate on real sessions. On the sample photo the tight crop pushed
  "Contempt" to 0.90, which is why only three broad groups (neutral / positive /
  other) feed the suggestions and there is deliberately **no expression score**.
* **Gaze**: head yaw/pitch from MediaPipe's facial transformation matrix plus the iris
  offset inside the eye opening (`effective angle = head + eye`). The report
  optionally re-centres on the session's median effective angles, capped at ±15°, to
  cancel camera-placement bias. Sign conventions verified by flipping/rotating
  images: yaw > 0 = turned toward image right, pitch > 0 = chin down, roll > 0 = top
  of head toward image right. Directions are in **image** coordinates.
* **Posture**: shoulder tilt, head lateral offset and head height relative to
  shoulder width, compared against a baseline from the first 5 s; no spine angle.
  Baseline is finalized at `stop()` for short sessions.
* **Synthetic frames cannot fake head turns** (MediaPipe fits a 3-D face model), so
  the test generator only supports a "face absent" segment; look-away logic is unit
  tested with synthetic landmarks and must be validated on real footage.
* Frames with no face carry `gaze=None` and `expression=None` (not "away"); posture
  is still evaluated and reports `upper_body_visible=False`.
* `posture_stability` score = `1 − max(flag rate)` (flags overlap).

### 4.3 Risks

* Model downloads (~25 MB) are fetched by `interview-cv download-models`; pin or
  vendor them before production.
* HSEmotion weights are trained on AffectNet (research-only licence).
* MediaPipe pins `opencv-contrib-python` and `protobuf`; keep TensorFlow/PyTorch out
  of this environment.
* All thresholds in `config.py` were set from two still images and need tuning on real
  webcam sessions on the test device.
* The report only describes visible behaviour and carries a disclaimer; do not add
  "confidence" / "nervousness" style metrics later.
