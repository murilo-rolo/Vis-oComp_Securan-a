"""
Script de avaliação do modelo de detecção de violência em vídeos.

Este script:
1. Carrega o modelo treinado
2. Executa inferência no conjunto de teste
3. Calcula métricas: Accuracy, Precision, Recall, F1-score
4. Gera matriz de confusão
5. Salva relatório em results/reports/
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models import create_model
from src.datasets import get_dataloaders
from src import paths as p


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader de teste
        device: Device (cuda/cpu)
    
    Returns:
        Dicionário com métricas calculadas
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print("Executando inferência no conjunto de teste...")
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Avaliando"):
            # Mover para device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            # Coletar predições e labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Converter para arrays numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    class_report = classification_report(
        all_labels,
        all_predictions,
        target_names=['Non-Violent', 'Violent'],
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'num_samples': int(len(all_labels)),
        'num_non_violent': int(np.sum(all_labels == 0)),
        'num_violent': int(np.sum(all_labels == 1))
    }
    
    return metrics


def _format_metrics_report(metrics: dict, title: str = "MÉTRICAS DE AVALIAÇÃO") -> str:
    """Gera relatório formatado de métricas (usado tanto para print quanto para salvar)."""
    cm = np.array(metrics['confusion_matrix'])
    report = metrics['classification_report']
    lines = [
        "=" * 50, title, "=" * 50,
        f"\nNúmero de amostras: {metrics['num_samples']}",
        f"  - Non-Violent: {metrics['num_non_violent']}",
        f"  - Violent: {metrics['num_violent']}",
        "\nMétricas Gerais:",
        f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)",
        f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)",
        f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)",
        "\nMatriz de Confusão:",
        "                Predito",
        "              Non-Violent  Violent",
        f"Real Non-Violent    {cm[0,0]:6d}    {cm[0,1]:6d}",
        f"     Violent        {cm[1,0]:6d}    {cm[1,1]:6d}",
        "\nClassification Report:",
        "  Non-Violent:",
        f"    Precision: {report['0']['precision']:.4f}",
        f"    Recall:    {report['0']['recall']:.4f}",
        f"    F1-Score:  {report['0']['f1-score']:.4f}",
        "  Violent:",
        f"    Precision: {report['1']['precision']:.4f}",
        f"    Recall:    {report['1']['recall']:.4f}",
        f"    F1-Score:  {report['1']['f1-score']:.4f}",
        "=" * 50 + "\n",
    ]
    return "\n".join(lines)


def print_metrics(metrics: dict):
    """Imprime métricas formatadas no console."""
    print(_format_metrics_report(metrics))


def save_report(metrics: dict, output_path: Path, format: str = "both"):
    """
    Salva relatório de métricas em arquivo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format in ("json", "both"):
        json_path = output_path.with_suffix(".json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Relatório JSON salvo em: {json_path}")

    if format in ("txt", "both"):
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, 'w') as f:
            f.write(_format_metrics_report(
                metrics, "RELATÓRIO DE AVALIAÇÃO - DETECÇÃO DE VIOLÊNCIA EM VÍDEOS"
            ))
        print(f"Relatório TXT salvo em: {txt_path}")


def evaluate(
    model_path: str,
    processed_data_root: str = str(p.PROCESSED_ROOT),
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 4,
    device: str = None,
    output_dir: str = str(p.REPORTS_ROOT),
    seed: int = 42
):
    """
    Função principal de avaliação.
    """

    # Configurar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Usando device: {device}")
    
    # Carregar modelo
    print(f"Carregando modelo de: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extrair hiperparâmetros do checkpoint
    num_frames = checkpoint.get('num_frames', num_frames)
    hidden_size = checkpoint.get('hidden_size', 256)
    num_layers = checkpoint.get('num_layers', 2)
    dropout = checkpoint.get('dropout', 0.5)
    
    # Criar modelo
    model = create_model(
        num_frames=num_frames,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
        pretrained=False,  # Não precisa baixar pesos se já carregamos
        device=device
    )
    
    # Carregar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo carregado (época {checkpoint.get('epoch', '?')}, "
          f"val_acc: {checkpoint.get('val_acc', 0):.2f}%)")
    
    # Criar DataLoader de teste
    print("Carregando DataLoader de teste...")
    _, _, test_loader = get_dataloaders(
        processed_data_root=processed_data_root,
        batch_size=batch_size,
        num_frames=num_frames,
        num_workers=num_workers,
        seed=seed
    )
    
    # Avaliar
    metrics = evaluate_model(model, test_loader, device)
    
    # Imprimir métricas
    print_metrics(metrics)
    
    # Salvar relatório
    output_path = Path(output_dir) / "metrics"
    save_report(metrics, output_path, format="both")


def main():
    """Função principal com argparse."""
    parser = argparse.ArgumentParser(
        description="Avaliar modelo de detecção de violência em vídeos"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Caminho para o modelo treinado (.pth)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Número de frames por vídeo (será sobrescrito se estiver no checkpoint)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Se None, detecta automaticamente"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

