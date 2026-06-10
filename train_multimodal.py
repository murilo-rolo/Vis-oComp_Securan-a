"""
Script de treinamento para modelo multimodal de detecção de risco.

Este script treina o MultimodalRiskDetector que combina:
- Video Features (ResNet-LSTM)
- Pose Features (keypoints)
- Emotion Features (vetores de emoção)

Uso:
    python train_multimodal.py --epochs 50 --batch_size 8 --fusion_method late
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.models.multimodal_risk import create_multimodal_model
from src.models.resnet_lstm import ResNetLSTM, create_model as create_video_model
from src.datasets.multimodal_dataset import get_multimodal_dataloaders


def train_epoch(model, video_model, train_loader, criterion, optimizer, device, epoch):
    """Treina por uma época."""
    model.train()
    video_model.eval()  # Video model em modo eval (não treinar)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for video, pose, emotion, labels in pbar:
        video = video.to(device)
        pose = pose.to(device)
        emotion = emotion.to(device)
        labels = labels.to(device)
        
        # Extrair features de vídeo se necessário
        with torch.no_grad():
            if len(video.shape) == 5:  # (batch, T, C, H, W) - frames
                # Extrair features usando ResNet-LSTM
                # get_features espera (batch, num_frames, C, H, W)
                video_features = video_model.get_features(video)  # (batch, D_v)
                # Expandir para ter dimensão temporal
                video_features = video_features.unsqueeze(1)  # (batch, 1, D_v)
                # Repetir para T timesteps
                T = video.shape[1]
                video_features = video_features.repeat(1, T, 1)  # (batch, T, D_v)
            else:
                # Já são features
                video_features = video
        
        # Forward
        optimizer.zero_grad()
        outputs = model(video_features, pose, emotion)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Estatísticas
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Atualizar progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, video_model, val_loader, criterion, device, epoch):
    """Valida o modelo."""
    model.eval()
    video_model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for video, pose, emotion, labels in pbar:
            video = video.to(device)
            pose = pose.to(device)
            emotion = emotion.to(device)
            labels = labels.to(device)
            
            # Extrair features de vídeo se necessário
            if len(video.shape) == 5:  # (batch, T, C, H, W) - frames
                # get_features espera (batch, num_frames, C, H, W)
                video_features = video_model.get_features(video)  # (batch, D_v)
                # Expandir para ter dimensão temporal
                video_features = video_features.unsqueeze(1)  # (batch, 1, D_v)
                # Repetir para T timesteps
                T = video.shape[1]
                video_features = video_features.repeat(1, T, 1)  # (batch, T, D_v)
            else:
                video_features = video
            
            outputs = model(video_features, pose, emotion)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo multimodal de detecção de risco")
    
    # Dados
    parser.add_argument(
        "--video_data_root",
        type=str,
        default="data/processed",
        help="Raiz dos dados de vídeo"
    )
    parser.add_argument(
        "--pose_data_root",
        type=str,
        default="data/pose",
        help="Raiz dos dados de pose"
    )
    parser.add_argument(
        "--emotion_data_root",
        type=str,
        default="data/emotion",
        help="Raiz dos dados de emoção"
    )
    
    # Modelo
    parser.add_argument(
        "--fusion_method",
        type=str,
        choices=["early", "late", "attention"],
        default="late",
        help="Método de fusão multimodal"
    )
    parser.add_argument(
        "--use_temporal_modeling",
        action="store_true",
        help="Usar LSTM para modelagem temporal por modalidade"
    )
    parser.add_argument(
        "--video_model_path",
        type=str,
        default=None,
        help="Caminho para modelo de vídeo pré-treinado (opcional)"
    )
    
    # Treinamento
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Número de frames por vídeo"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Tamanho da janela temporal"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/multimodal",
        help="Diretório para salvar checkpoints"
    )
    
    # Outros
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para treinamento"
    )
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Carregar modelo de vídeo (para extrair features)
    print("Carregando modelo de vídeo...")
    video_model = create_video_model(
        num_frames=args.num_frames,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        num_classes=2,
        pretrained=True,
        device=args.device
    )
    
    if args.video_model_path:
        try:
            checkpoint = torch.load(args.video_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                video_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                video_model.load_state_dict(checkpoint)
            print(f"Modelo de vídeo carregado de: {args.video_model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo de vídeo: {e}")
            print("Usando modelo com pesos ImageNet")
    
    # Criar modelo multimodal
    print("Criando modelo multimodal...")
    multimodal_model = create_multimodal_model(
        video_feature_dim=256,  # Saída do ResNet-LSTM
        pose_feature_dim=99,    # 33 joints * 3 se flatten
        emotion_feature_dim=8,  # 8 classes de emoção
        num_frames=args.window_size,
        fusion_method=args.fusion_method,
        use_temporal_modeling=args.use_temporal_modeling,
        device=args.device
    )
    
    # Criar DataLoaders
    print("Criando DataLoaders...")
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(
        video_data_root=args.video_data_root,
        pose_data_root=args.pose_data_root,
        emotion_data_root=args.emotion_data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        window_size=args.window_size,
        video_mode="frames",
        pose_mode="flatten",
        num_workers=args.num_workers,
        dataset_name="rwf2000"
    )
    
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(multimodal_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Treinamento
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("=" * 60)
    print("Treinamento do Modelo Multimodal")
    print("=" * 60)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Temporal modeling: {args.use_temporal_modeling}")
    print(f"Device: {args.device}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    print()
    
    for epoch in range(1, args.epochs + 1):
        # Treinar
        train_loss, train_acc = train_epoch(
            multimodal_model, video_model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_acc = validate(multimodal_model, video_model, val_loader, criterion, device, epoch)
        
        # Atualizar learning rate
        scheduler.step()
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': multimodal_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'fusion_method': args.fusion_method,
                'use_temporal_modeling': args.use_temporal_modeling
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"\n✓ Melhor modelo salvo! Val Acc: {val_acc:.2f}%")
        
        print()
    
    # Salvar histórico
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

