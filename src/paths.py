import os
import sys
from pathlib import Path

IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE' in os.environ

if IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/Othercomputers/Meu laptop/cv-security-threat-detection-develop")
    DATASET_ROOT = Path("/content/dataset")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset"

DATA_ROOT         = PROJECT_ROOT / "data"
RAW_DATA_ROOT     = DATA_ROOT / "raw"
PROCESSED_ROOT    = DATA_ROOT / "processed"
POSE_ROOT         = DATA_ROOT / "pose"
EMOTION_ROOT      = DATA_ROOT / "emotion"

RESULTS_ROOT      = PROJECT_ROOT / "results"
MODELS_ROOT       = RESULTS_ROOT / "models"
MULTIMODAL_ROOT   = RESULTS_ROOT / "multimodal"
CNN3D_ROOT        = RESULTS_ROOT / "cnn3d"
EMOTION_MODELS_ROOT = RESULTS_ROOT / "emotion"
REPORTS_ROOT      = RESULTS_ROOT / "reports"
EXPERIMENTS_ROOT  = RESULTS_ROOT / "experiments"
COMPARISON_ROOT   = RESULTS_ROOT / "comparison"

RWF2000_ROOT      = DATASET_ROOT / "RWF-2000"
UCF101_ROOT       = DATASET_ROOT / "UCF101"
AFFECTNET_ROOT    = DATASET_ROOT / "AffectNet"
