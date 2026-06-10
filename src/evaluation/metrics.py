"""
Cálculo de métricas de avaliação para modelos de detecção de violência.

Inclui: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC, AUC-PR
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Importar modelos necessários
try:
    from src.models.multimodal_risk import MultimodalRiskDetector
    from src.models.resnet_lstm import create_model as create_video_model
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: List[str] = ["Non-Violent", "Violent"]
) -> Dict:
    """
    Calcula métricas de avaliação completas.
    
    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_pred: Predições (0 ou 1)
        y_proba: Probabilidades da classe positiva (opcional, para AUC)
        class_names: Nomes das classes
    
    Returns:
        Dicionário com todas as métricas
    """
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    
    # Métricas por classe
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Métricas macro (média das classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Métricas weighted
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC e AUC-PR (se probabilidades fornecidas)
    auc_roc = None
    auc_pr = None
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)
        except ValueError:
            pass
    
    # Organizar resultados
    metrics = {
        "accuracy": float(accuracy),
        "precision": {
            class_names[0]: float(precision[0]),
            class_names[1]: float(precision[1]),
            "macro": float(precision_macro),
            "weighted": float(precision_weighted)
        },
        "recall": {
            class_names[0]: float(recall[0]),
            class_names[1]: float(recall[1]),
            "macro": float(recall_macro),
            "weighted": float(recall_weighted)
        },
        "f1_score": {
            class_names[0]: float(f1[0]),
            class_names[1]: float(f1[1]),
            "macro": float(f1_macro),
            "weighted": float(f1_weighted)
        },
        "specificity": float(specificity),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "confusion_matrix_array": cm.tolist()
    }
    
    if auc_roc is not None:
        metrics["auc_roc"] = float(auc_roc)
    if auc_pr is not None:
        metrics["auc_pr"] = float(auc_pr)
    
    return metrics


class MetricsCalculator:
    """
    Classe para calcular e salvar métricas de avaliação.
    """
    
    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: List[str] = ["Non-Violent", "Violent"],
        video_model_path: Optional[str] = None
    ):
        """
        Inicializa o calculador de métricas.
        
        Args:
            model: Modelo PyTorch para avaliação
            dataloader: DataLoader com dados de teste
            device: Device para inferência
            class_names: Nomes das classes
            video_model_path: Caminho para modelo de vídeo (ResNetLSTM) se necessário
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.class_names = class_names
        self.model.eval()
        
        # Verificar se é modelo multimodal e precisa de video_model
        self.video_model = None
        self.is_multimodal = False
        self.use_video_model = False
        
        if HAS_MULTIMODAL and isinstance(model, MultimodalRiskDetector):
            self.is_multimodal = True
            # Verificar se usa fusão "late" (onde video_model é necessário)
            if model.fusion_method == "late":
                self.use_video_model = True
                # Criar/carregar video_model (ResNetLSTM)
                self.video_model = create_video_model(
                    num_frames=16,
                    hidden_size=256,
                    num_layers=2,
                    dropout=0.5,
                    num_classes=2,
                    pretrained=True,
                    device=device
                )
                
                # Carregar pesos se fornecido
                if video_model_path:
                    try:
                        checkpoint = torch.load(video_model_path, map_location=device)
                        if 'model_state_dict' in checkpoint:
                            self.video_model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            self.video_model.load_state_dict(checkpoint)
                        print(f"✓ Video model carregado de: {video_model_path}")
                    except Exception as e:
                        print(f"⚠ Aviso: Não foi possível carregar video_model: {e}")
                        print("  Usando modelo com pesos ImageNet")
                
                self.video_model.eval()
    
    def evaluate(self) -> Dict:
        """
        Avalia o modelo e retorna métricas.
        
        Returns:
            Dicionário com métricas
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Assumir que batch é (inputs, labels) ou similar
                # Adaptar conforme estrutura do dataloader
                if len(batch) == 2:
                    inputs = batch[0]
                    labels = batch[1]
                else:
                    # Tentar inferir estrutura
                    inputs = batch[:-1]
                    labels = batch[-1]
                
                # Mover para device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
                
                # Processar inputs para modelo multimodal com video_model se necessário
                if self.is_multimodal and self.use_video_model and isinstance(inputs, (list, tuple)) and len(inputs) >= 3:
                    # Inputs são (video, pose, emotion, ...)
                    video, pose, emotion = inputs[0], inputs[1], inputs[2]
                    
                    # Extrair features de vídeo se necessário (mesma lógica de train_multimodal.py)
                    if len(video.shape) == 5:  # (batch, T, C, H, W) - frames
                        # Extrair features usando ResNet-LSTM
                        # get_features espera (batch, num_frames, C, H, W)
                        video_features = self.video_model.get_features(video)  # (batch, D_v)
                        # Expandir para ter dimensão temporal
                        video_features = video_features.unsqueeze(1)  # (batch, 1, D_v)
                        # Repetir para T timesteps
                        T = video.shape[1]
                        video_features = video_features.repeat(1, T, 1)  # (batch, T, D_v)
                    else:
                        # Já são features
                        video_features = video
                    
                    # Forward pass com features processadas
                    outputs = self.model(video_features, pose, emotion)
                else:
                    # Forward pass padrão
                    if isinstance(inputs, torch.Tensor):
                        outputs = self.model(inputs)
                    elif isinstance(inputs, (list, tuple)):
                        outputs = self.model(*inputs)
                    else:
                        raise ValueError(f"Formato de input não suportado: {type(inputs)}")
                
                # Obter predições e probabilidades
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())  # Probabilidade classe positiva
                
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels.cpu().numpy())
                    else:
                        all_labels.append(np.array(labels))
        
        # Concatenar resultados
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels) if len(all_labels) > 0 else None
        
        if y_true is None:
            raise ValueError("Labels não fornecidos no dataloader")
        
        # Calcular métricas
        metrics = calculate_metrics(y_true, y_pred, y_proba, self.class_names)
        
        return metrics, y_true, y_pred, y_proba
    
    def save_results(
        self,
        output_dir: str,
        experiment_name: str,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ):
        """
        Salva resultados da avaliação.
        
        Args:
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
            metrics: Dicionário com métricas
            y_true: Labels verdadeiros
            y_pred: Predições
            y_proba: Probabilidades
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar métricas em JSON
        metrics_file = output_path / f"{experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Salvar confusion matrix
        self._plot_confusion_matrix(
            y_true, y_pred, experiment_name, output_path / f"{experiment_name}_confusion_matrix.png"
        )
        
        # Salvar curvas ROC e PR
        if y_proba is not None:
            self._plot_roc_curve(
                y_true, y_proba, experiment_name, output_path / f"{experiment_name}_roc_curve.png"
            )
            self._plot_pr_curve(
                y_true, y_proba, experiment_name, output_path / f"{experiment_name}_pr_curve.png"
            )
    
    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{experiment_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva curva ROC."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{experiment_name} ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    def _plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        experiment_name: str,
        output_path: Path
    ):
        """Plota e salva curva Precision-Recall."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0, 1)
        plt.title(f'{experiment_name} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

