"""
Script principal para treinamento do modelo de detecção de violência em vídeos.

Este script:
1. Carrega DataLoaders de treino e validação
2. Configura modelo, loss, otimizador
3. Treina o modelo com logs de loss/accuracy
4. Valida a cada época
5. Salva o melhor modelo
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
import sys
from typing import Tuple

# Adicionar src ao path para imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_model
from src.datasets import get_dataloaders


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Treina o modelo por uma época.
    
    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader de treino
        criterion: Função de loss
        optimizer: Otimizador
        device: Device (cuda/cpu)
        epoch: Número da época atual
    
    Returns:
        Tupla (loss_média, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for frames, labels in pbar:
        # Mover para device
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estatísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Atualizar barra de progresso
        current_loss = running_loss / len(pbar)
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Valida o modelo.
    
    Args:
        model: Modelo a ser validado
        val_loader: DataLoader de validação
        criterion: Função de loss
        device: Device (cuda/cpu)
        epoch: Número da época atual
    
    Returns:
        Tupla (loss_média, accuracy)
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for frames, labels in pbar:
            # Mover para device
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Atualizar barra de progresso
            current_loss = running_loss / len(pbar)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train(
    processed_data_root: str = "data/processed",
    batch_size: int = 8,
    num_frames: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    num_workers: int = 4,
    device: str = None,
    save_dir: str = "results/models",
    seed: int = 42,
    early_stopping_patience: int = 10,
    use_scheduler: bool = True
):
    """
    Função principal de treinamento.
    
    Args:
        processed_data_root: Raiz dos dados processados
        batch_size: Tamanho do batch
        num_frames: Número de frames por vídeo
        num_epochs: Número de épocas
        learning_rate: Taxa de aprendizado
        hidden_size: Tamanho do hidden state da LSTM
        num_layers: Número de camadas LSTM
        dropout: Taxa de dropout
        num_workers: Número de workers para DataLoader
        device: Device (None = auto)
        save_dir: Diretório para salvar modelos
        seed: Seed para reprodutibilidade
    """
    # Configurar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Usando device: {device}")
    
    # Configurar seed para reprodutibilidade
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Criar DataLoaders
    print("Carregando DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        processed_data_root=processed_data_root,
        batch_size=batch_size,
        num_frames=num_frames,
        num_workers=num_workers,
        seed=seed
    )
    
    # Criar modelo
    print("Criando modelo...")
    model = create_model(
        num_frames=num_frames,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
        pretrained=True,
        device=device
    )
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        # Verificar se a versão do PyTorch suporta 'verbose'
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            # Versão antiga do PyTorch não tem 'verbose'
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
    
    # Criar diretório para salvar modelos
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Treinamento
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*50)
    print("Iniciando treinamento...")
    print("="*50 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Atualizar learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_acc)
        
        # Log
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0  # Resetar contador de paciência
            
            model_path = save_path / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'num_frames': num_frames,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout
            }, model_path)
            
            print(f"  ✓ Melhor modelo salvo! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\n⚠️  Early stopping ativado após {patience_counter} épocas sem melhoria.")
            print(f"Melhor validação: {best_val_acc:.2f}% na época {best_epoch}")
            break
        
        print("-" * 50)
    
    print("\n" + "="*50)
    print("Treinamento concluído!")
    print(f"Melhor validação: {best_val_acc:.2f}% na época {best_epoch}")
    print(f"Modelo salvo em: {save_path / 'best_model.pth'}")
    print("="*50)


def main():
    """Função principal com argparse."""
    parser = argparse.ArgumentParser(
        description="Treinar modelo de detecção de violência em vídeos"
    )
    
    parser.add_argument(
        "--processed_data_root",
        type=str,
        default="data/processed",
        help="Raiz dos dados processados"
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
        help="Número de frames por vídeo"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Número de épocas"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Tamanho do hidden state da LSTM"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Número de camadas LSTM"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Taxa de dropout"
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
        "--save_dir",
        type=str,
        default="results/models",
        help="Diretório para salvar modelos"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Número de épocas sem melhoria para early stopping (0 = desabilitado)"
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=True,
        help="Usar learning rate scheduler"
    )
    parser.add_argument(
        "--no_scheduler",
        dest="use_scheduler",
        action="store_false",
        help="Desabilitar learning rate scheduler"
    )
    
    args = parser.parse_args()
    
    train(
        processed_data_root=args.processed_data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        use_scheduler=args.use_scheduler
    )


if __name__ == "__main__":
    main()

