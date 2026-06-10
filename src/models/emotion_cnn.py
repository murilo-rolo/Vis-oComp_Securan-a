"""
Modelo de CNN para reconhecimento de emoções faciais (FER - Facial Expression Recognition).

Arquitetura baseada em ResNet-18 pré-treinada no ImageNet, adaptada para classificação
de emoções usando o dataset AffectNet.

Classes de emoção (8 classes padrão):
- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Angry
- 4: Fearful
- 5: Disgust
- 6: Surprise
- 7: Contempt
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

# Tentar importar ResNet18_Weights (disponível em torchvision >= 0.13)
try:
    from torchvision.models import ResNet18_Weights
    HAS_WEIGHTS_API = True
except ImportError:
    HAS_WEIGHTS_API = False


class EmotionNet(nn.Module):
    """
    Modelo CNN para classificação de emoções faciais.
    
    Baseado em ResNet-18 pré-treinada no ImageNet, com adaptação para FER.
    
    Arquitetura:
    1. ResNet-18 pré-treinada (sem última camada FC)
    2. Camada FC adaptada para num_emotions classes
    3. Softmax para probabilidades de emoções
    """
    
    # Mapeamento de classes de emoção
    EMOTION_CLASSES = [
        'neutral', 'happy', 'sad', 'angry', 
        'fearful', 'disgust', 'surprise', 'contempt'
    ]
    
    def __init__(
        self,
        num_emotions: int = 8,
        pretrained: bool = True,
        dropout: float = 0.5,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Inicializa o modelo de emoção.
        
        Args:
            num_emotions: Número de classes de emoção (padrão: 8 para AffectNet)
            pretrained: Se True, usa ResNet-18 pré-treinada no ImageNet
            dropout: Taxa de dropout antes da camada final
            input_size: Tamanho de entrada (altura, largura) - padrão: (224, 224)
        """
        super(EmotionNet, self).__init__()
        
        self.num_emotions = num_emotions
        self.input_size = input_size
        
        # Carregar ResNet-18 pré-treinada
        if pretrained:
            if HAS_WEIGHTS_API:
                resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(pretrained=True)
        else:
            if HAS_WEIGHTS_API:
                resnet = models.resnet18(weights=None)
            else:
                resnet = models.resnet18(pretrained=False)
        
        # Remover última camada FC
        # Manter até a camada antes do FC (avgpool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature size do ResNet-18: 512
        self.feature_size = 512
        
        # Camada de dropout
        self.dropout = nn.Dropout(dropout)
        
        # Camada FC final para classificação de emoções
        self.fc = nn.Linear(self.feature_size, num_emotions)
        
        # Inicializar pesos da camada FC
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch_size, 3, H, W) - faces detectadas
        
        Returns:
            Tensor de saída (batch_size, num_emotions) com logits
        """
        # Extrair features com ResNet-18
        # (batch_size, 512, 1, 1) após avgpool
        features = self.backbone(x)
        
        # Remover dimensões espaciais (1, 1)
        # (batch_size, 512)
        features = features.view(features.size(0), self.feature_size)
        
        # Aplicar dropout
        features = self.dropout(features)
        
        # Classificação de emoções
        # (batch_size, num_emotions)
        logits = self.fc(features)
        
        return logits
    
    def predict_emotions(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> torch.Tensor:
        """
        Prediz emoções com probabilidades.
        
        Args:
            x: Tensor de entrada (batch_size, 3, H, W)
            return_probs: Se True, retorna probabilidades (softmax), senão logits
        
        Returns:
            Tensor (batch_size, num_emotions) com probabilidades ou logits
        """
        logits = self.forward(x)
        
        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return probs
        else:
            return logits
    
    def get_emotion_name(self, emotion_idx: int) -> str:
        """
        Retorna o nome da emoção dado o índice.
        
        Args:
            emotion_idx: Índice da emoção (0-7)
        
        Returns:
            Nome da emoção
        """
        if 0 <= emotion_idx < len(self.EMOTION_CLASSES):
            return self.EMOTION_CLASSES[emotion_idx]
        return "unknown"
    
    @classmethod
    def get_emotion_classes(cls) -> list:
        """Retorna lista de classes de emoção."""
        return cls.EMOTION_CLASSES.copy()


def create_emotion_model(
    num_emotions: int = 8,
    pretrained: bool = True,
    dropout: float = 0.5,
    input_size: Tuple[int, int] = (224, 224),
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> EmotionNet:
    """
    Função auxiliar para criar e carregar modelo de emoção.
    
    Args:
        num_emotions: Número de classes de emoção
        pretrained: Se True, usa ResNet-18 pré-treinada
        dropout: Taxa de dropout
        input_size: Tamanho de entrada
        checkpoint_path: Caminho para checkpoint pré-treinado (opcional)
        device: Device para mover o modelo
    
    Returns:
        Modelo EmotionNet no device especificado
    """
    model = EmotionNet(
        num_emotions=num_emotions,
        pretrained=pretrained,
        dropout=dropout,
        input_size=input_size
    )
    
    # Carregar checkpoint se fornecido
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Lidar com diferentes formatos de checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Checkpoint carregado de: {checkpoint_path}")
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            print("Usando modelo com pesos ImageNet pré-treinados")
    
    model = model.to(device)
    model.eval()  # Modo avaliação por padrão
    
    return model

