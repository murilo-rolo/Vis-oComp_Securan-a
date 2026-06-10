"""
Módulo de inferência em tempo real para detecção de risco.
"""

from .realtime_risk_detector import RealTimeRiskDetector, create_realtime_detector
from .multi_camera_detector import MultiCameraRiskDetector, create_multi_camera_detector

__all__ = [
    'RealTimeRiskDetector',
    'create_realtime_detector',
    'MultiCameraRiskDetector',
    'create_multi_camera_detector'
]

