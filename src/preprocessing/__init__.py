"""
Módulo de pré-processamento para o projeto de detecção de violência em vídeos.
"""

from .organize_videos import organize_rwf2000_dataset
from .extract_frames import preprocess_dataset, extract_frames_from_video

__all__ = [
    "organize_rwf2000_dataset",
    "preprocess_dataset",
    "extract_frames_from_video"
]

