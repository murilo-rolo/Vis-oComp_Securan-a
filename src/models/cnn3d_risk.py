"""
Modelo CNN 3D para detecção de risco/violência em vídeos.

Este módulo implementa CNN3DRiskDetector baseado em modelos 3D do torchvision
(R3D, R(2+1)D, etc.) para classificação de ações e detecção de violência.

Arquitetura:
1. Backbone 3D pré-treinado (UCF101 ou Kinetics)
2. Adaptação da última camada para classificação binária
3. Suporte a fine-tuning em RWF-2000
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Literal

# Tentar importar modelos 3D do torchvision
try:
    import torchvision.models.video as video_models
    HAS_VIDEO_MODELS = True
except ImportError:
    HAS_VIDEO_MODELS = False
    print("Aviso: torchvision.models.video não disponível. Use torchvision >= 0.13.0")


class CNN3DRiskDetector(nn.Module):
    """
    Modelo CNN 3D para detecção de risco/violência.
    
    Baseado em modelos 3D do torchvision (R3D, R(2+1)D, etc.) pré-treinados
    em datasets de action recognition (UCF101, Kinetics).
    
    Input: Clipe de vídeo (T, C, H, W) onde T é o número de frames
    Output: Logits para classificação binária (violent/non-violent)
    """
    
    def __init__(
        self,
        model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18", "mvit"] = "r2plus1d_18",
        num_classes: int = 2,
        pretrained: bool = True,
        pretrained_dataset: str = "kinetics400",  # "kinetics400" ou "ucf101"
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Inicializa o modelo CNN 3D.
        
        Args:
            model_name: Nome do modelo 3D ("r3d_18", "r2plus1d_18", "mc3_18", "mvit")
            num_classes: Número de classes de saída (2 para binary classification)
            pretrained: Se True, carrega pesos pré-treinados
            pretrained_dataset: Dataset usado para pré-treinamento ("kinetics400" ou "ucf101")
            dropout: Taxa de dropout antes da camada final
            freeze_backbone: Se True, congela pesos do backbone (apenas treina FC)
        """
        super(CNN3DRiskDetector, self).__init__()
        
        if not HAS_VIDEO_MODELS:
            raise ImportError(
                "torchvision.models.video não disponível. "
                "Instale torchvision >= 0.13.0: pip install torchvision>=0.13.0"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.pretrained_dataset = pretrained_dataset
        self.num_frames = 16  # Padrão, pode ser ajustado
        
        # Carregar modelo base
        if model_name == "r3d_18":
            if pretrained:
                try:
                    if pretrained_dataset == "kinetics400":
                        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Kinetics400_Weights.DEFAULT)
                    else:
                        # Para UCF101, precisaríamos de pesos customizados ou usar Kinetics
                        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Kinetics400_Weights.DEFAULT)
                except:
                    self.backbone = video_models.r3d_18(weights=None)
            else:
                self.backbone = video_models.r3d_18(weights=None)
        
        elif model_name == "r2plus1d_18":
            if pretrained:
                try:
                    if pretrained_dataset == "kinetics400":
                        self.backbone = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Kinetics400_Weights.DEFAULT)
                    else:
                        self.backbone = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Kinetics400_Weights.DEFAULT)
                except:
                    self.backbone = video_models.r2plus1d_18(weights=None)
            else:
                self.backbone = video_models.r2plus1d_18(weights=None)
        
        elif model_name == "mc3_18":
            if pretrained:
                try:
                    self.backbone = video_models.mc3_18(weights=video_models.MC3_18_Kinetics400_Weights.DEFAULT)
                except:
                    self.backbone = video_models.mc3_18(weights=None)
            else:
                self.backbone = video_models.mc3_18(weights=None)
        
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")
        
        # Obter dimensão de features do backbone
        # A última camada FC tem input_size = feature_dim
        if hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
            # Remover última camada FC
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif hasattr(self.backbone, 'head'):
            # Para modelos mais novos (MViT, etc.)
            feature_dim = self.backbone.head.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            # Para modelos do torchvision, geralmente a estrutura é:
            # stem -> layer1-4 -> avgpool -> fc
            # Vamos tentar acessar fc diretamente
            try:
                # Tentar acessar como atributo
                if hasattr(self.backbone, 'fc'):
                    feature_dim = self.backbone.fc.in_features
                else:
                    # Última camada deve ser FC
                    last_layer = list(self.backbone.children())[-1]
                    if isinstance(last_layer, nn.Linear):
                        feature_dim = last_layer.in_features
                    else:
                        # Fallback: usar 512 (padrão para ResNet-18)
                        feature_dim = 512
                
                # Remover última camada
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            except:
                # Fallback: assumir feature_dim padrão
                feature_dim = 512
                # Tentar remover último módulo
                modules = list(self.backbone.children())
                if len(modules) > 0:
                    self.backbone = nn.Sequential(*modules[:-1])
        
        self.feature_dim = feature_dim
        
        # Congelar backbone se solicitado
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Camada de dropout
        self.dropout = nn.Dropout(dropout)
        
        # Camada FC final para classificação binária
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Inicializar pesos da camada FC
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch_size, T, C, H, W) ou (batch_size, C, T, H, W)
               Modelos 3D do torchvision esperam (batch, C, T, H, W)
        
        Returns:
            Tensor de saída (batch_size, num_classes) com logits
        """
        # Verificar e ajustar formato se necessário
        # Modelos 3D do torchvision esperam (batch, C, T, H, W)
        if len(x.shape) == 5:
            batch_size, dim1, dim2, H, W = x.shape
            
            # Se primeiro dim (T) é menor que segundo (C=3), é (batch, T, C, H, W)
            # Ou se dim1 parece ser número de frames (16, 32, etc.)
            # Converter para (batch, C, T, H, W)
            if dim1 < dim2 or (dim1 in [8, 16, 32, 64] and dim2 == 3):
                # Assumir que é (batch, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)  # (batch, T, C, H, W) -> (batch, C, T, H, W)
        
        # Extrair features com backbone
        # Backbone retorna (batch, feature_dim, 1, 1, 1) após pooling
        features = self.backbone(x)
        
        # Remover dimensões espaciais e temporais
        # features shape: (batch, feature_dim, 1, 1, 1) ou (batch, feature_dim)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Aplicar dropout
        features = self.dropout(features)
        
        # Classificação final
        logits = self.classifier(features)
        
        return logits
    
    def load_pretrained_ucf101(
        self,
        checkpoint_path: str,
        strict: bool = True
    ):
        """
        Carrega pesos pré-treinados em UCF101.
        
        Args:
            checkpoint_path: Caminho para checkpoint pré-treinado
            strict: Se True, requer que todas as chaves correspondam
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Lidar com diferentes formatos de checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Carregar pesos do backbone (ignorar classifier se existir)
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('classifier'):
                # Remover prefixo 'backbone.' se existir
                new_key = key.replace('backbone.', '')
                backbone_state_dict[new_key] = value
        
        try:
            self.backbone.load_state_dict(backbone_state_dict, strict=strict)
            print(f"✓ Pesos do backbone carregados de: {checkpoint_path}")
        except Exception as e:
            print(f"⚠ Aviso ao carregar pesos: {e}")
            print("  Tentando carregar sem strict matching...")
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
    
    def freeze_backbone_layers(self, num_layers: int = -1):
        """
        Congela as primeiras N camadas do backbone.
        
        Args:
            num_layers: Número de camadas para congelar (-1 = todas)
        """
        if num_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Congelar primeiras N camadas
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers[:num_layers]):
                for param in layer.parameters():
                    param.requires_grad = False


def create_cnn3d_model(
    model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18"] = "r2plus1d_18",
    num_classes: int = 2,
    pretrained: bool = True,
    pretrained_dataset: str = "kinetics400",
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> CNN3DRiskDetector:
    """
    Função auxiliar para criar modelo CNN 3D.
    
    Args:
        model_name: Nome do modelo 3D
        num_classes: Número de classes
        pretrained: Se True, usa pesos pré-treinados
        pretrained_dataset: Dataset de pré-treinamento
        dropout: Taxa de dropout
        freeze_backbone: Se True, congela backbone
        checkpoint_path: Caminho para checkpoint customizado (opcional)
        device: Device para mover o modelo
    
    Returns:
        Modelo CNN3DRiskDetector no device especificado
    """
    model = CNN3DRiskDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained and checkpoint_path is None,  # Não usar pretrained se checkpoint fornecido
        pretrained_dataset=pretrained_dataset,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    # Definir num_frames
    model.num_frames = num_frames
    
    # Carregar checkpoint customizado se fornecido
    if checkpoint_path is not None:
        if "ucf101" in checkpoint_path.lower() or "ucf" in checkpoint_path.lower():
            model.load_pretrained_ucf101(checkpoint_path)
        else:
            # Carregar checkpoint genérico
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Checkpoint carregado de: {checkpoint_path}")
    
    model = model.to(device)
    return model

