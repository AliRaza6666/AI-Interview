"""interview_cv: live-webcam computer-vision analysis for AI mock interviews.

Public entry points:

* :class:`interview_cv.session.CVSession` - owns the webcam, processes frames, accepts
  interview lifecycle events and builds the final report.
* :class:`interview_cv.schemas.InterviewEvent` - the proposed event contract shared with
  the other interview modules.
* :func:`interview_cv.report.build_report` - turns an observation log into a feedback report.
"""

from .config import PipelineConfig
from .schemas import CVReport, FrameObservation, InterviewEvent, ObservationLog
from .session import CVSession

__all__ = ["PipelineConfig", "FrameObservation", "ObservationLog", "CVReport", "InterviewEvent", "CVSession"]
__version__ = "0.2.0"
