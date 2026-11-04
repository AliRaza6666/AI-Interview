"""Command line interface (live webcam only).

    interview-cv run --camera 0 --max-seconds 60 --report report.json --log observations.jsonl
    interview-cv run --camera 0 --show                 # preview window; q / Esc stops the session
    interview-cv download-models [--dir PATH] [--samples]
    interview-cv report --log observations.jsonl [--report report.json]

The ``run`` command is a manual test harness for the session manager. In the real
application the host creates a ``CVSession`` and forwards interview events to it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from .config import MODEL_SOURCES, PipelineConfig, default_model_dir


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # noqa: S310 - fixed https URLs
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def cmd_download_models(args: argparse.Namespace) -> int:
    model_dir = Path(args.dir) if args.dir else default_model_dir()
    for name, url in MODEL_SOURCES.items():
        dest = model_dir / name
        if dest.is_file() and not args.force:
            print(f"exists: {dest}")
            continue
        _download(url, dest)
    if args.samples:
        from .samples import SAMPLE_IMAGE_SOURCES

        sample_dir = Path(args.sample_dir) if args.sample_dir else model_dir.parent / "samples"
        for name, url in SAMPLE_IMAGE_SOURCES.items():
            dest = sample_dir / name
            if dest.is_file() and not args.force:
                print(f"exists: {dest}")
                continue
            _download(url, dest)
    print("done")
    return 0


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.model_dir:
        cfg.models.model_dir = Path(args.model_dir)
    cfg.sample_fps = args.sample_fps
    cfg.max_seconds = args.max_seconds
    cfg.enable_expression = not args.no_expression
    cfg.enable_posture = not args.no_posture
    cfg.enable_gaze = not args.no_gaze
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    import time

    from .report import format_report_text
    from .session import CVSession
    from .video import VideoWriter, open_webcam

    cfg = _build_config(args)
    missing = cfg.models.missing()
    if missing:
        print("Missing model files:\n  " + "\n  ".join(str(m) for m in missing), file=sys.stderr)
        print("Run: interview-cv download-models", file=sys.stderr)
        return 2

    session_id = args.session_id or uuid.uuid4().hex
    show = bool(args.show)
    record = args.record
    writer: Optional[VideoWriter] = None
    stop_requested = False
    if show or record:
        from .visualize import draw_overlay

    def on_frame(frame, analysis) -> None:
        nonlocal writer, stop_requested
        img = draw_overlay(frame.image, analysis, landmarks=not args.no_landmarks)
        if record:
            if writer is None:
                writer = VideoWriter(record, cfg.sample_fps or 5.0, (frame.width, frame.height))
            writer.write(img)
        if show:
            import cv2

            cv2.imshow("interview-cv", img)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                stop_requested = True

    try:
        source = open_webcam(args.camera, max_seconds=args.max_seconds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"session {session_id}: starting camera {args.camera} (Ctrl+C to stop)")
    session = CVSession(session_id, cfg, camera=source, on_frame=on_frame if (show or record) else None)
    try:
        session.start()
        while session.is_running and not stop_requested:
            if session._thread is not None and not session._thread.is_alive():
                break  # camera ended (max_seconds reached or device failure)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        if writer is not None:
            writer.close()
            print(f"annotated recording: {record}")
        if show:
            import cv2

            cv2.destroyAllWindows()

    if session.last_error is not None:
        print(f"capture error: {session.last_error}", file=sys.stderr)
    assert session.log is not None
    if args.log:
        if str(args.log).endswith(".jsonl"):
            session.log.save_jsonl(args.log)
        else:
            session.log.save_json(args.log)
        print(f"observation log: {args.log}")

    report = session.build_report()
    if args.report:
        report.save_json(args.report)
        print(f"report: {args.report}")
    print()
    print(format_report_text(report))
    return 0 if session.last_error is None else 1


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, format_report_text
    from .schemas import ObservationLog

    log_path = Path(args.log)
    observations = ObservationLog.load_jsonl(log_path) if log_path.suffix == ".jsonl" else ObservationLog.load_json(log_path)
    report = build_report(observations)
    if args.report:
        report.save_json(args.report)
        print(f"report: {args.report}")
    print(format_report_text(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="interview-cv", description="Live-webcam computer-vision analysis for mock interviews.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a CV session on the live webcam")
    r.add_argument("--camera", default="0", help="webcam index (default 0)")
    r.add_argument("--session-id", help="shared interview session id (default: random)")
    r.add_argument("--report", help="write the CV report JSON here")
    r.add_argument("--log", help="write the observation log here (.json or .jsonl)")
    r.add_argument("--record", help="debug: write the annotated live feed to this .mp4")
    r.add_argument("--show", action="store_true", help="display annotated frames in a window")
    r.add_argument("--no-landmarks", action="store_true", help="do not draw landmark points on the overlay")
    r.add_argument("--sample-fps", type=float, default=5.0, help="analysis rate (default 5)")
    r.add_argument("--max-seconds", type=float, default=None, help="stop automatically after this many seconds")
    r.add_argument("--model-dir", help="directory with the model files (default: cv_module/models)")
    r.add_argument("--no-expression", action="store_true")
    r.add_argument("--no-posture", action="store_true")
    r.add_argument("--no-gaze", action="store_true")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("download-models", help="download MediaPipe and HSEmotion model files")
    d.add_argument("--dir", help="target directory (default: cv_module/models)")
    d.add_argument("--samples", action="store_true", help="also download the sample test image used by the tests")
    d.add_argument("--sample-dir", help="target directory for samples (default: cv_module/samples)")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download_models)

    rp = sub.add_parser("report", help="rebuild a report from a saved observation log")
    rp.add_argument("--log", required=True)
    rp.add_argument("--report", help="write the report JSON here")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
