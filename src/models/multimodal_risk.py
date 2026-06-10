"""
Modelo Multimodal para Detecção de Risco/Violência.

Este módulo implementa fusão de múltiplas modalidades:
- Video Features (ResNet-LSTM)
- Pose Features (keypoints de pose)
- Emotion Features (vetores de emoção facial)

Estratégias de fusão:
1. Early Fusion: Concatena features brutas antes do processamento
2. Late Fusion: Processa cada modalidade separadamente e funde no final
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


class MultimodalRiskDetector(nn.Module):
    """
    Modelo multimodal para detecção de risco/violência.
    
    Combina três modalidades:
    - Video: Features de ResNet-LSTM (T x D_v)
    - Pose: Keypoints de pose (T x D_p)
    - Emotion: Vetores de emoção (T x D_e)
    
    Arquitetura:
    1. Processamento por modalidade (MLPs ou LSTMs)
    2. Fusão em espaço comum
    3. Classificação binária (violent/non-violent)
    """
    
    def __init__(
        self,
        # Dimensões de entrada
        video_feature_dim: int = 256,      # D_v: saída do ResNet-LSTM
        pose_feature_dim: int = 99,        # D_p: 33 joints * 3 (x, y, visibility) se flatten
        emotion_feature_dim: int = 8,      # D_e: 8 classes de emoção
        num_frames: int = 16,              # T: tamanho da janela temporal
        
        # Dimensões de processamento
        video_hidden_dim: int = 128,       # Dimensão após processamento de vídeo
        pose_hidden_dim: int = 64,         # Dimensão após processamento de pose
        emotion_hidden_dim: int = 32,      # Dimensão após processamento de emoção
        
        # Fusão
        fusion_dim: int = 256,             # Dimensão do espaço de fusão
        fusion_method: Literal["early", "late", "attention"] = "late",
        
        # Classificação
        num_classes: int = 2,              # 2: violent/non-violent
        dropout: float = 0.5,
        
        # Processamento temporal
        use_temporal_modeling: bool = True,  # Se True, usa LSTM por modalidade
        temporal_hidden_size: int = 64
    ):
        """
        Inicializa o modelo multimodal.
        
        Args:
            video_feature_dim: Dimensão das features de vídeo (D_v)
            pose_feature_dim: Dimensão das features de pose (D_p)
            emotion_feature_dim: Dimensão das features de emoção (D_e)
            num_frames: Número de frames na sequência (T)
            video_hidden_dim: Dimensão oculta para processamento de vídeo
            pose_hidden_dim: Dimensão oculta para processamento de pose
            emotion_hidden_dim: Dimensão oculta para processamento de emoção
            fusion_dim: Dimensão do espaço de fusão
            fusion_method: Método de fusão ("early", "late", "attention")
            num_classes: Número de classes de saída
            dropout: Taxa de dropout
            use_temporal_modeling: Se True, usa LSTM para modelagem temporal
            temporal_hidden_size: Tamanho do hidden state para LSTM temporal
        """
        super(MultimodalRiskDetector, self).__init__()
        
        self.video_feature_dim = video_feature_dim
        self.pose_feature_dim = pose_feature_dim
        self.emotion_feature_dim = emotion_feature_dim
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        self.use_temporal_modeling = use_temporal_modeling
        
        # Processamento por modalidade
        if fusion_method == "early":
            # Early Fusion: Concatena features brutas
            total_input_dim = video_feature_dim + pose_feature_dim + emotion_feature_dim
            
            if use_temporal_modeling:
                # LSTM para processar sequência concatenada
                self.video_processor = nn.LSTM(
                    total_input_dim,
                    temporal_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0
                )
                video_output_dim = temporal_hidden_size
            else:
                # MLP simples
                self.video_processor = nn.Sequential(
                    nn.Linear(total_input_dim, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                video_output_dim = fusion_dim
            
            # Não precisa processar modalidades separadamente
            self.pose_processor = None
            self.emotion_processor = None
            
        else:
            # Late Fusion ou Attention: Processa cada modalidade separadamente
            
            # Processador de vídeo
            if use_temporal_modeling:
                self.video_processor = nn.Sequential(
                    nn.LSTM(
                        video_feature_dim,
                        temporal_hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=0
                    ),
                    nn.Dropout(dropout)
                )
                video_lstm = self.video_processor[0]
                video_output_dim = temporal_hidden_size
            else:
                self.video_processor = nn.Sequential(
                    nn.Linear(video_feature_dim, video_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                video_output_dim = video_hidden_dim
            
            # Processador de pose
            if use_temporal_modeling:
                self.pose_processor = nn.Sequential(
                    nn.LSTM(
                        pose_feature_dim,
                        temporal_hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=0
                    ),
                    nn.Dropout(dropout)
                )
                pose_lstm = self.pose_processor[0]
                pose_output_dim = temporal_hidden_size
            else:
                self.pose_processor = nn.Sequential(
                    nn.Linear(pose_feature_dim, pose_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                pose_output_dim = pose_hidden_dim
            
            # Processador de emoção
            if use_temporal_modeling:
                self.emotion_processor = nn.Sequential(
                    nn.LSTM(
                        emotion_feature_dim,
                        temporal_hidden_size,
                        num_layers=1,
                        batch_first=True,
                        dropout=0
                    ),
                    nn.Dropout(dropout)
                )
                emotion_lstm = self.emotion_processor[0]
                emotion_output_dim = temporal_hidden_size
            else:
                self.emotion_processor = nn.Sequential(
                    nn.Linear(emotion_feature_dim, emotion_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                emotion_output_dim = emotion_hidden_dim
        
        # Módulo de fusão
        if fusion_method == "early":
            # Early fusion já foi feito, só precisa classificar
            fusion_input_dim = video_output_dim
            
        elif fusion_method == "late":
            # Late Fusion: Concatena features processadas
            fusion_input_dim = video_output_dim + pose_output_dim + emotion_output_dim
            
        elif fusion_method == "attention":
            # Attention Fusion: Usa attention para combinar modalidades
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            # Projeção para espaço comum
            self.video_proj = nn.Linear(video_output_dim, fusion_dim)
            self.pose_proj = nn.Linear(pose_output_dim, fusion_dim)
            self.emotion_proj = nn.Linear(emotion_output_dim, fusion_dim)
            fusion_input_dim = fusion_dim
        else:
            raise ValueError(f"Método de fusão não suportado: {fusion_method}")
        
        # Camadas de fusão e classificação
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classificador final
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos das camadas."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _process_temporal(
        self,
        processor: nn.Module,
        features: torch.Tensor,
        is_lstm: bool = False
    ) -> torch.Tensor:
        """
        Processa features temporais.
        
        Args:
            processor: Processador (LSTM ou MLP)
            features: Features de entrada (batch, T, D)
            is_lstm: Se True, processor é LSTM
        
        Returns:
            Features processadas (batch, D_out) ou (batch, T, D_out)
        """
        # Verificar e ajustar shape se necessário
        if len(features.shape) == 2:
            # (batch, D) -> (batch, 1, D)
            features = features.unsqueeze(1)
        elif len(features.shape) != 3:
            raise ValueError(
                f"Features devem ter shape (batch, T, D) ou (batch, D), "
                f"mas recebeu shape {features.shape}"
            )
        
        if is_lstm:
            # LSTM retorna (batch, T, hidden_size)
            lstm_out, _ = processor(features)
            # Usar último timestep
            return lstm_out[:, -1, :]
        else:
            # MLP processa cada timestep
            # Se features são (batch, T, D), aplicar MLP em cada timestep
            batch_size, T, D = features.shape
            features_flat = features.view(batch_size * T, D)
            processed = processor(features_flat)
            processed = processed.view(batch_size, T, -1)
            # Média temporal
            return processed.mean(dim=1)
    
    def forward(
        self,
        video_features: torch.Tensor,
        pose_features: torch.Tensor,
        emotion_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass do modelo multimodal.
        
        Args:
            video_features: Features de vídeo (batch, T, D_v) ou (batch, D_v)
            pose_features: Features de pose (batch, T, D_p) ou (batch, T, num_joints, 3)
            emotion_features: Features de emoção (batch, T, D_e)
        
        Returns:
            Logits de classificação (batch, num_classes)
        """
        batch_size = video_features.shape[0]
        
        # Garantir que todas as features têm dimensão temporal
        # Se video_features é (batch, D_v), expandir para (batch, 1, D_v)
        if len(video_features.shape) == 2:
            video_features = video_features.unsqueeze(1)  # (batch, 1, D_v)
        elif len(video_features.shape) == 5:
            # Video é (batch, T, C, H, W) - precisa fazer flatten para (batch, T, C*H*W)
            batch_size, T, C, H, W = video_features.shape
            video_features = video_features.view(batch_size, T, C * H * W)
        
        if len(pose_features.shape) == 4 and pose_features.shape[-1] == 3:
            # Pose é (batch, T, num_joints, 3), flatten para (batch, T, num_joints*3)
            pose_features = pose_features.view(batch_size, pose_features.shape[1], -1)
        elif len(pose_features.shape) == 2:
            pose_features = pose_features.unsqueeze(1)  # (batch, 1, D_p)
        
        if len(emotion_features.shape) == 2:
            emotion_features = emotion_features.unsqueeze(1)  # (batch, 1, D_e)
        
        # Garantir que todas têm o mesmo número de timesteps
        T = max(video_features.shape[1], pose_features.shape[1], emotion_features.shape[1])
        
        if video_features.shape[1] < T:
            # Repetir último frame
            last_frame = video_features[:, -1:, :]
            padding = last_frame.repeat(1, T - video_features.shape[1], 1)
            video_features = torch.cat([video_features, padding], dim=1)
        
        if pose_features.shape[1] < T:
            last_frame = pose_features[:, -1:, :]
            padding = last_frame.repeat(1, T - pose_features.shape[1], 1)
            pose_features = torch.cat([pose_features, padding], dim=1)
        
        if emotion_features.shape[1] < T:
            last_frame = emotion_features[:, -1:, :]
            padding = last_frame.repeat(1, T - emotion_features.shape[1], 1)
            emotion_features = torch.cat([emotion_features, padding], dim=1)
        
        # Processar modalidades
        if self.fusion_method == "early":
            # Concatena features brutas
            combined = torch.cat([video_features, pose_features, emotion_features], dim=2)
            # Processa sequência concatenada
            if self.use_temporal_modeling:
                processed, _ = self.video_processor(combined)
                processed = processed[:, -1, :]  # Último timestep
            else:
                processed = self._process_temporal(self.video_processor, combined, is_lstm=False)
            fusion_input = processed
        
        elif self.fusion_method == "late":
            # Processa cada modalidade separadamente
            if self.use_temporal_modeling:
                video_out, _ = self.video_processor[0](video_features)
                video_out = video_out[:, -1, :]
                if len(self.video_processor) > 1:
                    video_out = self.video_processor[1](video_out)
                
                pose_out, _ = self.pose_processor[0](pose_features)
                pose_out = pose_out[:, -1, :]
                if len(self.pose_processor) > 1:
                    pose_out = self.pose_processor[1](pose_out)
                
                emotion_out, _ = self.emotion_processor[0](emotion_features)
                emotion_out = emotion_out[:, -1, :]
                if len(self.emotion_processor) > 1:
                    emotion_out = self.emotion_processor[1](emotion_out)
            else:
                video_out = self._process_temporal(self.video_processor, video_features, is_lstm=False)
                pose_out = self._process_temporal(self.pose_processor, pose_features, is_lstm=False)
                emotion_out = self._process_temporal(self.emotion_processor, emotion_features, is_lstm=False)
            
            # Concatena features processadas
            fusion_input = torch.cat([video_out, pose_out, emotion_out], dim=1)
        
        elif self.fusion_method == "attention":
            # Processa cada modalidade
            if self.use_temporal_modeling:
                video_out, _ = self.video_processor[0](video_features)
                pose_out, _ = self.pose_processor[0](pose_features)
                emotion_out, _ = self.emotion_processor[0](emotion_features)
            else:
                video_out = self._process_temporal(self.video_processor, video_features, is_lstm=False)
                pose_out = self._process_temporal(self.pose_processor, pose_features, is_lstm=False)
                emotion_out = self._process_temporal(self.emotion_processor, emotion_features, is_lstm=False)
            
            # Projeta para espaço comum
            video_proj = self.video_proj(video_out.unsqueeze(1))  # (batch, 1, fusion_dim)
            pose_proj = self.pose_proj(pose_out.unsqueeze(1))
            emotion_proj = self.emotion_proj(emotion_out.unsqueeze(1))
            
            # Concatena para attention
            modalities = torch.cat([video_proj, pose_proj, emotion_proj], dim=1)  # (batch, 3, fusion_dim)
            
            # Self-attention
            attended, _ = self.attention(modalities, modalities, modalities)
            
            # Pooling (média)
            fusion_input = attended.mean(dim=1)  # (batch, fusion_dim)
        
        # Fusão e classificação
        fused = self.fusion_layers(fusion_input)
        logits = self.classifier(fused)
        
        return logits


def create_multimodal_model(
    video_feature_dim: int = 256,
    pose_feature_dim: int = 99,
    emotion_feature_dim: int = 8,
    num_frames: int = 16,
    fusion_method: Literal["early", "late", "attention"] = "late",
    use_temporal_modeling: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> MultimodalRiskDetector:
    """
    Função auxiliar para criar modelo multimodal.
    
    Args:
        video_feature_dim: Dimensão das features de vídeo
        pose_feature_dim: Dimensão das features de pose
        emotion_feature_dim: Dimensão das features de emoção
        num_frames: Número de frames
        fusion_method: Método de fusão
        use_temporal_modeling: Se True, usa LSTM por modalidade
        device: Device para mover o modelo
        **kwargs: Argumentos adicionais para MultimodalRiskDetector
    
    Returns:
        Modelo MultimodalRiskDetector no device especificado
    """
    model = MultimodalRiskDetector(
        video_feature_dim=video_feature_dim,
        pose_feature_dim=pose_feature_dim,
        emotion_feature_dim=emotion_feature_dim,
        num_frames=num_frames,
        fusion_method=fusion_method,
        use_temporal_modeling=use_temporal_modeling,
        **kwargs
    )
    
    model = model.to(device)
    return model

