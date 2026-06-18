"""
Script de treinamento para modelo de Emotion Recognition no dataset AffectNet.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Optional

from src.models.emotion_cnn import create_emotion_model
from src.training.utils import run_epoch, create_dataloader

# Mapeamento de classes AffectNet
AFFECTNET_CLASSES = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fearful': 4,
    'disgust': 5,
    'surprise': 6,
    'contempt': 7
}


class AffectNetDataset(Dataset):
    """
    Dataset para AffectNet.
    
    Estrutura esperada:
    dataset/AffectNet/
    ├── Train/
    │   ├── neutral/
    │   ├── happy/
    │   └── ...
    └── Test/
        └── ...
    """
    
    def __init__(
        self,
        data_root: Path,
        split: str = "Train",
        transform: Optional[transforms.Compose] = None
    ):
        """
        Inicializa o dataset AffectNet.
        
        Args:
            data_root: Raiz do dataset AffectNet
            split: "Train" ou "Test"
            transform: Transformações a aplicar
        """
        self.data_root = Path(data_root) / split
        self.split = split
        self.transform = transform
        
        # Carregar amostras
        self.samples = []
        self.class_to_idx = AFFECTNET_CLASSES.copy()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Processar cada classe
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_root / class_name
            if not class_dir.exists():
                # Tentar com primeira letra maiúscula
                class_dir = self.data_root / class_name.capitalize()
            
            if class_dir.exists():
                # Listar imagens
                for ext in ['.jpg', '.jpeg', '.png']:
                    for img_path in class_dir.glob(f"*{ext}"):
                        self.samples.append((img_path, class_idx))
        
        print(f"AffectNet {split}: {len(self.samples)} amostras carregadas")
        print(f"  Classes: {len(self.class_to_idx)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Carregar imagem
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Se erro, retornar imagem preta
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Aplicar transformações
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(is_train: bool = True):
    """
    Retorna transformações para treino ou validação.
    
    Args:
        is_train: Se True, retorna augmentations para treino
    
    Returns:
        Compose de transformações
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])





def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de Emotion Recognition no AffectNet")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/AffectNet",
        help="Caminho para o dataset AffectNet"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/emotion",
        help="Diretório para salvar checkpoints"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Número de épocas (padrão: 10, use 50+ para treinamento completo)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamanho do batch"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para treinamento"
    )
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar datasets
    train_dataset = AffectNetDataset(
        data_root=args.dataset_path,
        split="Train",
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = AffectNetDataset(
        data_root=args.dataset_path,
        split="Test",
        transform=get_transforms(is_train=False)
    )
    
    # Criar DataLoaders
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    
    # Criar modelo
    device = torch.device(args.device)
    model = create_emotion_model(
        num_emotions=8,
        pretrained=True,
        dropout=0.5,
        device=args.device
    )
    
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
    print("Treinamento do Modelo de Emotion Recognition")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {args.device}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    print()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
            desc=f"Epoch {epoch} [Train]"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            is_train=False, desc=f"Epoch {epoch} [Val]"
        )
        
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
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

