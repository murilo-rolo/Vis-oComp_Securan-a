"""
Módulo de Emotion Recognition para detecção de emoções faciais em vídeos.

Este módulo fornece:
- Extração de emoções usando modelos FER pré-treinados
- Dataset PyTorch para sequências de emoções
- Utilitários para processamento de emotion data
"""

from .extract_emotion import extract_emotions_from_video, process_videos_for_emotion
from .emotion_dataset import EmotionSequenceDataset, get_emotion_dataloaders

__all__ = [
    'extract_emotions_from_video',
    'process_videos_for_emotion',
    'EmotionSequenceDataset',
    'get_emotion_dataloaders'
]

