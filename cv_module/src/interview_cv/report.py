"""Aggregate an ObservationLog into a CVReport with metrics, heuristic scores and cautious suggestions.

Design rules:
* Only observable behaviour is reported (where the face was, where the head pointed,
  which expression category the classifier picked, how the shoulders sat).
* No inference about confidence, honesty, nervousness or internal emotional state.
* Every suggestion carries the metric it is based on so the user can judge it.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Optional

import numpy as np

from .analyzers.gaze import classify_angles
from .analyzers.posture import ALL_FLAGS
from .config import EXPRESSION_GROUPS, ReportConfig
from .schemas import (
    EVENT_ANSWER_ENDED,
    EVENT_ANSWER_STARTED,
    EVENT_QUESTION_STARTED,
    PHASE_ANSWER,
    CVReport,
    Episode,
    ExpressionMetrics,
    FaceMetrics,
    FrameObservation,
    GazeMetrics,
    ObservationLog,
    PostureMetrics,
    QuestionSummary,
    Suggestion,
)

DISCLAIMER = (
    "These observations are approximate, computed automatically from video, and describe only visible "
    "behaviour (face position, head direction, expression category, shoulder/head geometry). They do not "
    "measure confidence, honesty, nervousness or emotions, and should be read as practice hints, not judgements."
)


# --------------------------------------------------------------------------- helpers


def _rate(count: int, total: int) -> Optional[float]:
    return round(count / total, 4) if total > 0 else None


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def episodes(
    frames: Iterable[FrameObservation],
    state_of: Callable[[FrameObservation], Optional[str]],
    sample_interval_s: float,
) -> list[Episode]:
    """Group consecutive sampled frames with the same (non-None) state into episodes.

    The episode end is extended by one sample interval so a single frame has a
    non-zero duration.
    """
    out: list[Episode] = []
    current: Optional[Episode] = None
    for f in frames:
        s = state_of(f)
        if current is not None and s == current.state:
            current.end_s = f.timestamp_s + sample_interval_s
            continue
        if current is not None:
            out.append(current)
            current = None
        if s is not None:
            current = Episode(state=s, start_s=f.timestamp_s, end_s=f.timestamp_s + sample_interval_s)
    if current is not None:
        out.append(current)
    return out


def _sample_interval(log: ObservationLog) -> float:
    if log.meta.sample_fps and log.meta.sample_fps > 0:
        return 1.0 / log.meta.sample_fps
    ts = [f.timestamp_s for f in log.frames]
    if len(ts) >= 2:
        d = np.diff(ts)
        d = d[d > 0]
        if len(d):
            return float(np.median(d))
    return 0.2


# --------------------------------------------------------------------------- metrics


def face_metrics(log: ObservationLog, cfg: ReportConfig, dt: float) -> FaceMetrics:
    frames = [f for f in log.frames if f.face is not None]
    present = sum(1 for f in frames if f.face and f.face.present)
    eps = [e for e in episodes(frames, lambda f: "absent" if (f.face and not f.face.present) else None, dt)]
    long_eps = [e for e in eps if e.duration_s >= cfg.min_episode_seconds]
    return FaceMetrics(
        frames=len(frames),
        present_frames=present,
        presence_rate=_rate(present, len(frames)),
        longest_absence_s=round(max((e.duration_s for e in eps), default=0.0), 2),
        absence_episodes=len(long_eps),
    )


def recentered_gaze_states(frames: list[FrameObservation], cfg: ReportConfig) -> tuple[list[tuple[str, str]], tuple[float, float]]:
    """Re-classify gaze per frame after subtracting the session's median effective angles.

    Returns ``(states, (yaw_offset, pitch_offset))``. Frames without effective angles keep
    their original state. Offsets are clipped to ``cfg.max_recenter_deg``.
    """
    yaws = [f.gaze.effective_yaw_deg for f in frames if f.gaze and f.gaze.effective_yaw_deg is not None]
    pitches = [f.gaze.effective_pitch_deg for f in frames if f.gaze and f.gaze.effective_pitch_deg is not None]
    if not yaws or not pitches:
        return [(f.gaze.state, f.gaze.direction) for f in frames if f.gaze], (0.0, 0.0)
    cap = cfg.max_recenter_deg
    yaw_off = float(np.clip(np.median(yaws), -cap, cap))
    pitch_off = float(np.clip(np.median(pitches), -cap, cap))
    out: list[tuple[str, str]] = []
    for f in frames:
        g = f.gaze
        assert g is not None
        if g.effective_yaw_deg is None or g.effective_pitch_deg is None:
            out.append((g.state, g.direction))
        else:
            out.append(classify_angles(g.effective_yaw_deg - yaw_off, g.effective_pitch_deg - pitch_off, cfg.gaze))
    return out, (yaw_off, pitch_off)


def gaze_metrics(log: ObservationLog, cfg: ReportConfig, dt: float) -> GazeMetrics:
    frames = [f for f in log.frames if f.gaze is not None]
    n = len(frames)
    if cfg.recenter_gaze:
        per_frame, (yaw_off, pitch_off) = recentered_gaze_states(frames, cfg)
    else:
        per_frame, (yaw_off, pitch_off) = [(f.gaze.state, f.gaze.direction) for f in frames if f.gaze], (0.0, 0.0)
    state_of = {id(f): s for f, s in zip(frames, per_frame)}
    states = Counter(s for s, _ in per_frame)
    directions = Counter(d for s, d in per_frame if s == "away")
    away_eps = [e for e in episodes(frames, lambda f: "away" if state_of[id(f)][0] == "away" else None, dt)]
    yaws = [abs(f.gaze.head_yaw_deg) for f in frames if f.gaze and f.gaze.head_yaw_deg is not None]
    pitches = [abs(f.gaze.head_pitch_deg) for f in frames if f.gaze and f.gaze.head_pitch_deg is not None]
    return GazeMetrics(
        frames_evaluated=n,
        camera_rate=_rate(states.get("camera", 0), n),
        away_rate=_rate(states.get("away", 0), n),
        unknown_rate=_rate(states.get("unknown", 0), n),
        direction_rates={k: _rate(v, n) or 0.0 for k, v in sorted(directions.items())},
        away_episodes=sum(1 for e in away_eps if e.duration_s >= cfg.min_episode_seconds),
        longest_away_s=round(max((e.duration_s for e in away_eps), default=0.0), 2),
        mean_abs_head_yaw_deg=round(float(np.mean(yaws)), 2) if yaws else None,
        mean_abs_head_pitch_deg=round(float(np.mean(pitches)), 2) if pitches else None,
        recenter_yaw_deg=round(yaw_off, 2),
        recenter_pitch_deg=round(pitch_off, 2),
    )


def expression_metrics(log: ObservationLog) -> ExpressionMetrics:
    frames = [f for f in log.frames if f.expression is not None]
    n = len(frames)
    labels = Counter(f.expression.label for f in frames if f.expression)
    groups = Counter(f.expression.group for f in frames if f.expression)
    return ExpressionMetrics(
        frames_evaluated=n,
        label_rates={k: _rate(v, n) or 0.0 for k, v in sorted(labels.items(), key=lambda kv: -kv[1])},
        group_rates={g: _rate(groups.get(g, 0), n) or 0.0 for g in ("neutral", "positive", "other")} if n else {},
        dominant_label=labels.most_common(1)[0][0] if labels else None,
        dominant_group=groups.most_common(1)[0][0] if groups else None,
    )


def posture_metrics(log: ObservationLog, cfg: ReportConfig, dt: float) -> PostureMetrics:
    frames = [f for f in log.frames if f.posture is not None]
    n = len(frames)
    visible = [f for f in frames if f.posture and f.posture.upper_body_visible]
    nv = len(visible)
    flag_counts = Counter(flag for f in visible for flag in (f.posture.flags if f.posture else []))
    flag_eps: dict[str, int] = {}
    for flag in ALL_FLAGS:
        eps = episodes(visible, lambda f, fl=flag: fl if (f.posture and fl in f.posture.flags) else None, dt)
        flag_eps[flag] = sum(1 for e in eps if e.duration_s >= cfg.min_episode_seconds)
    tilts = [abs(f.posture.shoulder_tilt_deg) for f in visible if f.posture and f.posture.shoulder_tilt_deg is not None]
    offsets = [abs(f.posture.head_offset_ratio) for f in visible if f.posture and f.posture.head_offset_ratio is not None]
    return PostureMetrics(
        frames_evaluated=n,
        upper_body_visible_rate=_rate(nv, n),
        flag_rates={flag: (_rate(flag_counts.get(flag, 0), nv) or 0.0) for flag in ALL_FLAGS} if nv else {},
        mean_abs_shoulder_tilt_deg=round(float(np.mean(tilts)), 2) if tilts else None,
        mean_abs_head_offset_ratio=round(float(np.mean(offsets)), 3) if offsets else None,
        flag_episodes=flag_eps if nv else {},
    )


# --------------------------------------------------------------------------- scores & suggestions


def compute_scores(face: FaceMetrics, gaze: GazeMetrics, posture: PostureMetrics) -> dict[str, Optional[float]]:
    """Heuristic 0-100 scores. Expression deliberately has no score (classifier reliability)."""
    scores: dict[str, Optional[float]] = {
        "face_presence": round(face.presence_rate * 100, 1) if face.presence_rate is not None else None,
        "camera_focus": round(gaze.camera_rate * 100, 1) if gaze.camera_rate is not None else None,
        "posture_stability": None,
    }
    if posture.flag_rates:
        # Share of visible frames without any flag is approximated by 1 - max flag rate (flags overlap).
        worst = max(posture.flag_rates.values()) if posture.flag_rates else 0.0
        scores["posture_stability"] = round((1.0 - worst) * 100, 1)
    available = [v for v in scores.values() if v is not None]
    scores["overall"] = round(float(np.mean(available)), 1) if available else None
    return scores


def build_suggestions(
    face: FaceMetrics, gaze: GazeMetrics, expr: ExpressionMetrics, posture: PostureMetrics, cfg: ReportConfig
) -> tuple[list[str], list[Suggestion]]:
    strengths: list[str] = []
    out: list[Suggestion] = []

    # Face presence
    if face.presence_rate is not None:
        if face.presence_rate < cfg.face_presence_warn_below:
            out.append(
                Suggestion(
                    "face",
                    "Stay in frame",
                    "Your face was not detected for part of the session. Check camera placement and lighting so "
                    "your face stays visible throughout.",
                    f"face detected in {_pct(face.presence_rate)} of sampled frames; longest gap {face.longest_absence_s:.1f}s",
                )
            )
        else:
            strengths.append(f"Face stayed visible to the camera ({_pct(face.presence_rate)} of the session).")

    # Gaze
    if gaze.camera_rate is not None:
        if gaze.camera_rate < cfg.camera_focus_warn_below:
            dirs = ", ".join(f"{k} {_pct(v)}" for k, v in gaze.direction_rates.items()) or "n/a"
            out.append(
                Suggestion(
                    "gaze",
                    "Look toward the camera more often",
                    "Your head and eyes were often directed away from the camera. In video interviews, glancing at "
                    "the lens while answering reads as eye contact. Looking away briefly to think is fine.",
                    f"camera-facing in {_pct(gaze.camera_rate)} of frames; away directions: {dirs}",
                )
            )
        elif gaze.camera_rate >= cfg.camera_focus_strength_above:
            strengths.append(f"Mostly camera-facing ({_pct(gaze.camera_rate)} of the time).")
        if gaze.longest_away_s >= cfg.long_look_away_seconds:
            out.append(
                Suggestion(
                    "gaze",
                    "Avoid long stretches of looking away",
                    "There was at least one long stretch of looking away from the camera. If you need notes, keep "
                    "them close to the lens so glances stay short.",
                    f"longest look-away {gaze.longest_away_s:.1f}s; {gaze.away_episodes} episode(s) over {cfg.min_episode_seconds:.0f}s",
                )
            )

    # Expression (observable categories only; broad groups)
    if expr.frames_evaluated:
        pos = expr.group_rates.get("positive", 0.0)
        other = expr.group_rates.get("other", 0.0)
        if pos < cfg.positive_expression_low_below:
            out.append(
                Suggestion(
                    "expression",
                    "Consider a brief smile at natural moments",
                    "The classifier rarely saw a smile-like expression. A short smile when greeting, when a question "
                    "is finished or when closing can make delivery feel more engaged. Keep it natural.",
                    f"'positive' expression category in {_pct(pos)} of frames; dominant category: {expr.dominant_label}",
                )
            )
        elif pos > 0.15:
            strengths.append(f"Smile-like expressions appeared regularly ({_pct(pos)} of frames).")
        if other >= cfg.other_expression_high_above:
            out.append(
                Suggestion(
                    "expression",
                    "Review your resting expression on the recording",
                    "The classifier often picked a category other than neutral or smile. This classifier is "
                    "unreliable on webcam footage, so treat it as a prompt to watch the recording yourself rather "
                    "than as a verdict.",
                    f"'other' categories in {_pct(other)} of frames (breakdown: "
                    + ", ".join(f"{k} {_pct(v)}" for k, v in expr.label_rates.items() if EXPRESSION_GROUPS.get(k) == "other")
                    + ")",
                )
            )

    # Posture
    if posture.upper_body_visible_rate is not None and posture.upper_body_visible_rate < cfg.upper_body_visible_warn_below:
        out.append(
            Suggestion(
                "posture",
                "Frame your head and shoulders",
                "Your shoulders were often out of view, so posture could only be partly assessed. Position the "
                "camera so your head and both shoulders are visible.",
                f"shoulders and face visible together in {_pct(posture.upper_body_visible_rate)} of frames",
            )
        )
    fr = posture.flag_rates
    if fr:
        if fr.get("shoulders_tilted", 0.0) > cfg.posture_flag_warn_above:
            out.append(
                Suggestion(
                    "posture",
                    "Level your shoulders",
                    "One shoulder sat noticeably lower than the other for much of the session. Squaring up to the "
                    "camera usually reads as more settled.",
                    f"shoulder tilt beyond threshold in {_pct(fr['shoulders_tilted'])} of frames; mean |tilt| "
                    f"{posture.mean_abs_shoulder_tilt_deg}°",
                )
            )
        if fr.get("head_off_center", 0.0) > cfg.posture_flag_warn_above:
            out.append(
                Suggestion(
                    "posture",
                    "Keep your head centred over your shoulders",
                    "Your head was often shifted to one side relative to your shoulders (leaning sideways).",
                    f"head off-centre in {_pct(fr['head_off_center'])} of frames",
                )
            )
        if fr.get("head_lowered", 0.0) > cfg.posture_flag_warn_above:
            out.append(
                Suggestion(
                    "posture",
                    "Sit up as the interview goes on",
                    "Compared with the start of the session, your head sat lower relative to your shoulders for a "
                    "large share of the time, which often indicates slumping.",
                    f"head lower than baseline in {_pct(fr['head_lowered'])} of frames",
                )
            )
        if fr.get("moved_closer", 0.0) + fr.get("moved_away", 0.0) > cfg.posture_flag_warn_above:
            out.append(
                Suggestion(
                    "posture",
                    "Keep a steady distance from the camera",
                    "Your distance to the camera changed a lot compared with the start of the session.",
                    f"closer {_pct(fr.get('moved_closer', 0.0))}, further {_pct(fr.get('moved_away', 0.0))} of frames",
                )
            )
        if not any(v > cfg.posture_flag_warn_above for v in fr.values()) and posture.upper_body_visible_rate and posture.upper_body_visible_rate >= cfg.upper_body_visible_warn_below:
            strengths.append("Posture stayed steady: level shoulders and a centred head for most of the session.")

    return strengths, out


# --------------------------------------------------------------------------- per question


def question_summaries(log: ObservationLog, cfg: ReportConfig) -> list[QuestionSummary]:
    """One summary per question over its *answer* frames (phase == 'answer').

    Question order follows the events when present, else first appearance in the frames.
    Questions with events but no answer frames are still listed (frames=0).
    """
    order: list[str] = []
    answer_windows: dict[str, list[Optional[float]]] = {}
    t0 = log.meta.started_at_epoch
    for e in log.events:
        if e.question_id and e.question_id not in order and e.type in (EVENT_QUESTION_STARTED, EVENT_ANSWER_STARTED):
            order.append(e.question_id)
        if e.question_id and t0 is not None:
            win = answer_windows.setdefault(e.question_id, [None, None])
            if e.type == EVENT_ANSWER_STARTED:
                win[0] = round(e.timestamp - t0, 3)
            elif e.type == EVENT_ANSWER_ENDED:
                win[1] = round(e.timestamp - t0, 3)
    answer_frames = [f for f in log.frames if f.phase == PHASE_ANSWER and f.question_id]
    for f in answer_frames:
        if f.question_id not in order:
            order.append(f.question_id)  # type: ignore[arg-type]
    if not order:
        return []

    gaze_frames = [f for f in log.frames if f.gaze is not None]
    if cfg.recenter_gaze:
        per_frame, _ = recentered_gaze_states(gaze_frames, cfg)
    else:
        per_frame = [(f.gaze.state, f.gaze.direction) for f in gaze_frames if f.gaze]
    gaze_state = {id(f): s for f, (s, _) in zip(gaze_frames, per_frame)}

    out: list[QuestionSummary] = []
    for qid in order:
        frames = [f for f in answer_frames if f.question_id == qid]
        n = len(frames)
        with_face = [f for f in frames if f.face is not None]
        present = [f for f in with_face if f.face and f.face.present]
        gz = [gaze_state[id(f)] for f in frames if id(f) in gaze_state]
        groups = Counter(f.expression.group for f in frames if f.expression)
        visible = [f for f in frames if f.posture and f.posture.upper_body_visible]
        flag_counts = Counter(flag for f in visible for flag in (f.posture.flags if f.posture else []))
        start, end = answer_windows.get(qid, [None, None])
        if start is None and frames:
            start = frames[0].timestamp_s
        if end is None and frames:
            end = frames[-1].timestamp_s
        out.append(
            QuestionSummary(
                question_id=qid,
                answer_start_s=start,
                answer_end_s=end,
                answer_duration_s=round(max(0.0, (end or 0.0) - (start or 0.0)), 2) if start is not None and end is not None else 0.0,
                frames=n,
                face_presence_rate=_rate(len(present), len(with_face)),
                camera_focus_rate=_rate(sum(1 for s in gz if s == "camera"), len(gz)),
                dominant_expression_group=groups.most_common(1)[0][0] if groups else None,
                posture_flag_rates={flag: (_rate(flag_counts.get(flag, 0), len(visible)) or 0.0) for flag in ALL_FLAGS} if visible else {},
            )
        )
    return out


# --------------------------------------------------------------------------- entry point


def build_report(log: ObservationLog, config: Optional[ReportConfig] = None) -> CVReport:
    cfg = config or ReportConfig()
    dt = _sample_interval(log)
    face = face_metrics(log, cfg, dt)
    gaze = gaze_metrics(log, cfg, dt)
    expr = expression_metrics(log)
    posture = posture_metrics(log, cfg, dt)
    scores = compute_scores(face, gaze, posture)
    strengths, suggestions = build_suggestions(face, gaze, expr, posture, cfg)
    session = {
        "session_id": log.meta.session_id,
        "source": log.meta.source,
        "duration_s": round(log.meta.duration_s, 2),
        "frames_sampled": log.meta.frames_sampled,
        "frames_total": log.meta.frames_total,
        "sample_fps": log.meta.sample_fps,
        "resolution": f"{log.meta.width}x{log.meta.height}",
        "started_at": log.meta.started_at,
        "started_at_epoch": log.meta.started_at_epoch,
        "events": len(log.events),
        "models": log.meta.models,
    }
    return CVReport(
        session=session,
        face=face,
        gaze=gaze,
        expression=expr,
        posture=posture,
        scores=scores,
        strengths=strengths,
        suggestions=suggestions,
        disclaimer=DISCLAIMER,
        questions=question_summaries(log, cfg),
    )


def format_report_text(report: CVReport) -> str:
    """Human-readable summary for the CLI."""
    lines = []
    s = report.session
    lines.append(f"CV feedback report - {s['source']} ({s['duration_s']}s, {s['frames_sampled']} sampled frames)")
    lines.append("")
    lines.append("Scores (heuristic, 0-100):")
    for k, v in report.scores.items():
        lines.append(f"  {k:<18} {v if v is not None else 'n/a'}")
    lines.append("")
    lines.append(f"Face presence     : {_pct(report.face.presence_rate)} (longest gap {report.face.longest_absence_s}s)")
    lines.append(
        f"Gaze              : camera {_pct(report.gaze.camera_rate)}, away {_pct(report.gaze.away_rate)}, "
        f"longest away {report.gaze.longest_away_s}s, away episodes {report.gaze.away_episodes}"
    )
    if report.expression.frames_evaluated:
        groups = ", ".join(f"{k} {_pct(v)}" for k, v in report.expression.group_rates.items())
        lines.append(f"Expression groups : {groups} (dominant label: {report.expression.dominant_label})")
    if report.posture.frames_evaluated:
        flags = ", ".join(f"{k} {_pct(v)}" for k, v in report.posture.flag_rates.items()) or "n/a"
        lines.append(f"Posture           : upper body visible {_pct(report.posture.upper_body_visible_rate)}; flags: {flags}")
    if report.questions:
        lines.append("")
        lines.append("Per question (answer segments):")
        for q in report.questions:
            flags = ", ".join(f"{k} {_pct(v)}" for k, v in q.posture_flag_rates.items() if v) or "none"
            lines.append(
                f"  {q.question_id}: {q.answer_duration_s}s, {q.frames} frames, face {_pct(q.face_presence_rate)}, "
                f"camera {_pct(q.camera_focus_rate)}, expression {q.dominant_expression_group or 'n/a'}, posture flags: {flags}"
            )
    lines.append("")
    if report.strengths:
        lines.append("Strengths:")
        lines += [f"  + {x}" for x in report.strengths]
    if report.suggestions:
        lines.append("Suggestions:")
        for sg in report.suggestions:
            lines.append(f"  - [{sg.category}] {sg.title}: {sg.message}")
            lines.append(f"      evidence: {sg.evidence}")
    lines.append("")
    lines.append(report.disclaimer)
    return "\n".join(lines)
