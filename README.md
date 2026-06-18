# Detecção de Situações de Risco em Vídeos de Sistemas de Vigilância

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](LICENSE)

Sistema avançado de detecção de violência em vídeos de segurança (CCTV) utilizando Deep Learning e Visão Computacional. Este projeto implementa uma **arquitetura multimodal** que combina múltiplas fontes de informação (vídeo, pose corporal e emoção facial) para detecção precisa de situações de risco e violência.

## Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Datasets](#datasets)
- [Arquitetura dos Modelos](#arquitetura-dos-modelos)
- [Instalação](#instalação)
- [Uso](#uso)
  - [Pré-processamento](#1-pré-processamento)
  - [Treinamento](#2-treinamento)
  - [Avaliação](#3-avaliação)
  - [Inferência em Tempo Real](#5-inferência-em-tempo-real)
- [Pipeline Completo de Execução](#pipeline-completo-de-execução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos de Sistema](#requisitos-de-sistema)
- [Troubleshooting](#troubleshooting)
- [Referências](#referências)

## Visão Geral

Este projeto foi desenvolvido como parte de um trabalho de mestrado em Visão Computacional. O sistema é capaz de analisar vídeos de câmeras de segurança e identificar automaticamente situações de violência, auxiliando em sistemas de monitoramento automatizado.

### Aplicações

- Monitoramento automatizado de segurança
- Alertas em tempo real para situações de risco
- Análise de grandes volumes de vídeo
- Redução de falsos positivos em sistemas de segurança

## Características

- **Arquitetura Multimodal**: Combina features de vídeo, pose corporal e emoção facial para detecção robusta
- **Múltiplos Modelos**: 
  - ResNet-LSTM para análise temporal de vídeo
  - CNN 3D (R3D, R(2+1)D, MC3) para action recognition
  - EmotionNet para classificação de emoções faciais
  - MultimodalRiskDetector para fusão de múltiplas modalidades
- **Estratégias de Fusão**: Early Fusion, Late Fusion e Attention-based Fusion
- **Transfer Learning**: Utiliza pesos pré-treinados (ImageNet, Kinetics400, UCF101)
- **Pipeline Completo**: Pré-processamento, extração de features, treinamento, avaliação e inferência em tempo real
- **Detecção em Tempo Real**: Suporte para webcam e streams RTSP
- **Sistema de Alertas**: Threshold configurável e detecção de janelas consecutivas
- **Altamente Configurável**: Parâmetros ajustáveis para diferentes cenários
- **Otimizado para Recursos Limitados**: Suporta treinamento em CPUs e GPUs
- **Métricas Detalhadas**: Gera relatórios completos de avaliação
- **Código modular**: Funções de treino/validação centralizadas em `src/training/utils.py`
- **Gerenciamento Centralizado de Caminhos**: `src/paths.py` detecta automaticamente o ambiente (local ou Google Colab) e configura todos os diretórios do projeto

## Datasets

O projeto utiliza múltiplos datasets para diferentes propósitos:

### Dataset Principal: RWF-2000

**RWF-2000** (Real-World Fighting Dataset) - Dataset principal para detecção de violência:

- **2.000 vídeos** rotulados como `violent` (Fight) e `non_violent` (NonFight)
- Vídeos gravados por câmeras de segurança (CCTV) em cenários reais
- Divisão: conjunto de treino e validação
- Disponível em repositórios públicos (ex.: [Kaggle](https://www.kaggle.com/datasets/vulamnguyen/rwf2000/data))

**Estrutura esperada:**
```
dataset/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

### Dataset de Pré-treinamento: UCF101

**UCF101** - Usado para pré-treinamento de modelos CNN 3D:

- **9 classes relevantes** selecionadas para detecção de violência:
  - BoxingPunchingBag, BoxingSpeedBag, Fencing, Nunchucks, Punch, SumoWrestling
  - Archery, CliffDiving, MilitaryParade (opcionais)
- Filtrado de 101 para 9 classes para otimizar treinamento
- Disponível em: [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

**Estrutura esperada:**
```
dataset/UCF101/
├── train/
│   ├── BoxingPunchingBag/
│   ├── BoxingSpeedBag/
│   ├── Fencing/
│   └── ... (9 classes)
├── test/
└── val/
```

### Dataset de Emoção: AffectNet

**AffectNet** - Usado para treinar o modelo de reconhecimento de emoções faciais:

- **8 classes de emoção**: Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprise, Contempt
- ~1.000.000 imagens de faces com anotações de emoção
- Disponível em: [AffectNet](http://mohammadmahoor.com/affectnet/)

**Estrutura esperada:**
```
dataset/AffectNet/
├── Train/
│   ├── neutral/
│   ├── happy/
│   ├── sad/
│   └── ... (8 classes)
└── Test/
```

> **Nota**: Os diretórios `dataset/` não são versionados no Git devido ao tamanho dos arquivos. Você precisará baixar os datasets separadamente.

## Arquitetura dos Modelos

O projeto implementa múltiplas arquiteturas para diferentes aspectos da detecção de violência:

### 1. ResNet-LSTM (Modelo Base de Vídeo)

Arquitetura híbrida para análise temporal de vídeo:

```
┌─────────────┐
│   Frames    │ (16 frames por vídeo)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   ResNet-18     │ (pré-treinada no ImageNet)
│  Feature Ext.   │ → 512 dimensões por frame
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│      LSTM       │ (2 camadas, hidden_size=256)
│  Temporal Model │ → Modela dependências temporais
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   FC Layer      │ (classificação binária)
│   Output: 2     │ → [non-violent, violent]
└─────────────────┘
```

**Componentes:**
- ResNet-18 pré-treinada no ImageNet (extrai 512 dims por frame)
- LSTM com 2 camadas (hidden_size=256) para modelagem temporal
- Camada FC final para classificação binária

### 2. CNN 3D (Action Recognition)

Modelos 3D para reconhecimento de ações em vídeo:

- **R3D-18**: ResNet 3D com convoluções 3D puras
- **R(2+1)D-18**: ResNet com convoluções factorizadas (2D+1D)
- **MC3-18**: Mixed Convolution 3D

**Pipeline:**
1. Pré-treinamento em UCF101 (9 classes relevantes)
2. Fine-tuning em RWF-2000 (2 classes: violent/non-violent)

### 3. EmotionNet (Reconhecimento de Emoções)

Modelo CNN para classificação de emoções faciais:

- Base: ResNet-18 adaptado
- Input: Face extraída e redimensionada (224×224)
- Output: 8 classes de emoção (probabilidades)
- Treinado em: AffectNet

**Classes de Emoção:**
- Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprise, Contempt

### 4. MultimodalRiskDetector (Modelo Principal)

Arquitetura multimodal que combina todas as modalidades:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Video Features │  │  Pose Features  │  │ Emotion Features│
│  (ResNet-LSTM)  │  │  (MediaPipe)    │  │  (EmotionNet)   │
│  T × 256        │  │  T × 99         │  │  T × 8          │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Video Processor │  │ Pose Processor  │  │Emotion Processor│
│ (LSTM/MLP)      │  │ (LSTM/MLP)      │  │ (LSTM/MLP)      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Fusion Module  │
                    │  (Early/Late/   │
                    │   Attention)    │
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Classifier     │
                    │  (Binary)       │
                    └─────────────────┘
```

**Estratégias de Fusão:**
- **Early Fusion**: Concatena features brutas antes do processamento
- **Late Fusion**: Processa cada modalidade separadamente e funde no final (recomendado)
- **Attention-based Fusion**: Usa Multi-Head Attention para aprender pesos adaptativos

**Modalidades:**
- **Vídeo**: Features extraídas do ResNet-LSTM ou CNN 3D (256 dims)
- **Pose**: 33 keypoints do MediaPipe (99 dims: x, y, visibility)
- **Emoção**: 8 classes de emoção do EmotionNet (8 dims)

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (para clonar o repositório)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/adrianalima99/cv-security-threat-detection.git
   cd cv-security-threat-detection
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt --prefer-binary
   ```
   > **Nota para Python 3.14:** Use `--prefer-binary` para garantir instalação de wheels pré-compilados e evitar erros de compilação.

4. **Baixe os datasets necessários:**
   
   **RWF-2000 (Obrigatório - Dataset Principal):**
   - Baixe de uma fonte confiável (Kaggle, GitHub, etc.)
   - Extraia na pasta `dataset/RWF-2000/` seguindo a estrutura mencionada acima
   
   **UCF101 (Opcional - Para pré-treinamento CNN 3D):**
   - Baixe o dataset UCF101 completo
   - O projeto já está configurado para usar apenas 9 classes relevantes
   - Extraia na pasta `dataset/UCF101/`
   
   **AffectNet (Opcional - Para treinar EmotionNet):**
   - Baixe o dataset AffectNet
   - Extraia na pasta `dataset/AffectNet/`
   - Necessário apenas se quiser treinar o modelo de emoção do zero

## Uso

### 1. Pré-processamento

O pré-processamento é dividido em múltiplas etapas para extrair diferentes modalidades:

#### Passo 1: Organizar vídeos

Organiza os vídeos do dataset RWF-2000:

```bash
python run_preprocessing.py
```

Ou manualmente:
```bash
python -m src.preprocessing.organize_videos
```

#### Passo 2: Extrair frames de vídeo

Extrai N frames por vídeo, redimensiona e normaliza:

```bash
python run_preprocessing.py
# ou
python -m src.preprocessing.extract_frames
```

#### Passo 3: Extrair keypoints de pose

Extrai keypoints de pose usando MediaPipe:

```bash
python run_pose_preprocessing.py --dataset rwf2000 --num_frames 16
```

**Opções:**
- `--dataset`: `rwf2000`, `ucf101` ou `both`
- `--num_frames`: Número de frames a processar por vídeo

#### Passo 4: Extrair emoções faciais

Extrai vetores de emoção usando EmotionNet:

```bash
# Primeiro, treine o modelo de emoção (se ainda não tiver)
python train_emotion_model.py --dataset_path dataset/AffectNet --epochs 50

# Depois, extraia emoções do RWF-2000
python run_emotion_preprocessing.py --dataset rwf2000 --model_path models/emotion/best_model.pth
```

**Configuração customizada:**

```python
from src.preprocessing import preprocess_dataset

preprocess_dataset(
    raw_data_root="data/raw",
    processed_data_root="data/processed",
    num_frames=16,          # Número de frames por vídeo
    target_size=(112, 112), # Tamanho dos frames
    normalize=True          # Normalização dos pixels
)
```

### 2. Treinamento

O projeto oferece múltiplos scripts de treinamento para diferentes modelos:

#### 2.1. Treinamento ResNet-LSTM (Modelo Base)

Treinamento do modelo ResNet-LSTM para análise de vídeo:

```bash
python -m src.training.train \
    --batch_size 8 \
    --num_frames 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --hidden_size 256 \
    --num_layers 2 \
    --dropout 0.5
```

**Treinamento rápido para teste:**
```bash
python tools/train_quick_test.py
```

#### 2.2. Treinamento EmotionNet

Treina o modelo de reconhecimento de emoções no AffectNet:

```bash
python train_emotion_model.py \
    --dataset_path dataset/AffectNet \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

#### 2.3. Treinamento CNN 3D

Pipeline de duas etapas: pré-treinamento + fine-tuning:

**Etapa 1: Pré-treinamento em UCF101 (9 classes)**
```bash
python train_cnn3d.py \
    --stage pretrain \
    --dataset ucf101 \
    --epochs 50 \
    --batch_size 8 \
    --model_name r2plus1d_18
```

**Etapa 2: Fine-tuning em RWF-2000 (2 classes)**
```bash
python train_cnn3d.py \
    --stage finetune \
    --dataset rwf2000 \
    --pretrained_path results/cnn3d/ucf101/best_model.pth \
    --epochs 30 \
    --batch_size 8
```

#### 2.4. Treinamento Multimodal (Modelo Principal)

Treina o modelo multimodal completo combinando vídeo, pose e emoção:

```bash
python train_multimodal.py \
    --epochs 50 \
    --batch_size 8 \
    --fusion_method late \
    --use_temporal_modeling \
    --video_model_path results/models/best_model.pth \
    --emotion_model_path models/emotion/best_model.pth
```

**Parâmetros principais:**
- `--fusion_method`: `early`, `late` (recomendado) ou `attention`
- `--use_temporal_modeling`: Usa LSTM para modelagem temporal por modalidade
- `--video_model_path`: Caminho para modelo ResNet-LSTM pré-treinado
- `--emotion_model_path`: Caminho para modelo EmotionNet pré-treinado (não usado atualmente, o modelo multimodal não carrega EmotionNet separadamente)

#### Parâmetros Principais

| Parâmetro | Descrição | Padrão | Recomendação |
|-----------|-----------|--------|--------------|
| `--batch_size` | Tamanho do batch | 8 | 4-8 para GPU, 2-4 para CPU |
| `--num_frames` | Frames por vídeo | 16 | 8-32 dependendo do vídeo |
| `--num_epochs` | Número de épocas | 50 | 10 para teste, 50+ para produção |
| `--learning_rate` | Taxa de aprendizado | 1e-4 | 1e-4 a 1e-3 |
| `--hidden_size` | Tamanho do hidden state LSTM | 256 | 128-512 |
| `--num_layers` | Camadas LSTM | 2 | 1-3 |
| `--dropout` | Taxa de dropout | 0.5 | 0.3-0.7 |
| `--num_workers` | Workers do DataLoader | 4 | 0-4 dependendo da CPU |
| `--early_stopping_patience` | Paciência do early stopping | 10 | 5-15 |

**Dicas para computadores com poucos recursos:**
- Use `--batch_size 4` ou `--batch_size 2` para reduzir uso de memória
- Use `--num_workers 0` ou `--num_workers 2` para reduzir uso de CPU
- Use `--num_epochs 10` para testes rápidos
- Use `--device cpu` se tiver problemas com GPU

O melhor modelo será salvo automaticamente em `results/models/best_model.pth`.

### 3. Avaliação

Avalie os modelos treinados no conjunto de teste:

#### 3.1. Avaliação de Modelo Unimodal

```bash
python run_evaluation.py \
    --model_path results/models/best_model.pth \
    --batch_size 8
```

#### 3.2. Avaliação de Modelo Multimodal

```bash
python run_evaluation.py \
    --model_path results/multimodal/best_model.pth \
    --model_type multimodal \
    --batch_size 8
```

Os relatórios serão salvos em:
- `results/reports/metrics.txt` (formato texto)
- `results/reports/metrics.json` (formato JSON)

### 4. Métricas de Avaliação

O script de avaliação calcula as seguintes métricas:

- **Accuracy**: Acurácia geral do modelo
- **Precision**: Precisão por classe (violent/non-violent)
- **Recall**: Recall por classe (sensibilidade)
- **F1-Score**: F1-score por classe (média harmônica)
- **Matriz de Confusão**: Visualização de erros de classificação
- **ROC-AUC**: Área sob a curva ROC
- **PR-AUC**: Área sob a curva Precision-Recall

### 5. Inferência em Tempo Real

Execute detecção de violência em tempo real usando webcam ou stream RTSP:

```bash
python run_realtime_risk_detection.py \
    --multimodal_model results/multimodal/best_model.pth \
    --video_model results/models/best_model.pth \
    --emotion_model models/emotion/best_model.pth \
    --source 0 \
    --risk_threshold 0.8 \
    --consecutive_windows 3
```

**Parâmetros:**
- `--source`: `0` para webcam ou URL RTSP (ex: `rtsp://...`)
- `--risk_threshold`: Threshold de probabilidade para alerta (0.0-1.0)
- `--consecutive_windows`: Número de janelas consecutivas acima do threshold para alerta
- `--use_cnn3d`: Usar CNN 3D ao invés de ResNet-LSTM para vídeo
- `--cnn3d_model`: Caminho para modelo CNN 3D (se usar `--use_cnn3d`)

## Estrutura de Dados

O projeto utiliza uma estrutura padronizada para organizar dados brutos e processados. É importante seguir esta estrutura para garantir que todos os scripts funcionem corretamente.

### Estrutura Completa de Diretórios

```
cv-security-threat-detection/
├── dataset/                           # Datasets originais (não versionados)
│   ├── RWF-2000/                      # Dataset principal (violência CCTV)
│   │   ├── train/
│   │   │   ├── Fight/                 # Vídeos violentos (treino)
│   │   │   │   └── *.avi
│   │   │   └── NonFight/              # Vídeos não violentos (treino)
│   │   │       └── *.avi
│   │   └── val/
│   │       ├── Fight/                  # Vídeos violentos (validação)
│   │       └── NonFight/               # Vídeos não violentos (validação)
│   ├── UCF101/                        # Dataset de pré-treinamento (9 classes)
│   │   ├── train/
│   │   │   ├── BoxingPunchingBag/
│   │   │   ├── BoxingSpeedBag/
│   │   │   └── ... (9 classes)
│   │   ├── test/
│   │   └── val/
│   └── AffectNet/                     # Dataset de emoções (8 classes)
│       ├── Train/
│       │   ├── neutral/
│       │   ├── happy/
│       │   └── ... (8 classes)
│       └── Test/
│
├── data/                              # Dados processados
│   ├── raw/                           # Vídeos organizados (após organize_videos.py)
│   │   ├── violent/                   # Todos os vídeos violentos (Fight)
│   │   │   └── *.avi
│   │   └── non_violent/               # Todos os vídeos não violentos (NonFight)
│   │       └── *.avi
│   │
│   ├── processed/                     # Frames extraídos (após extract_frames.py)
│   │   ├── violent/                   # Frames de vídeos violentos
│   │   │   └── <video_id>/            # Pasta por vídeo
│   │   │       ├── frame_0000.jpg
│   │   │       ├── frame_0001.jpg
│   │   │       └── ... (16 frames)
│   │   └── non_violent/               # Frames de vídeos não violentos
│   │       └── <video_id>/
│   │           └── ...
│   │
│   ├── pose/                          # Keypoints de pose (após extract_pose.py)
│   │   ├── rwf2000/                   # Pose do RWF-2000
│   │   │   ├── train/
│   │   │   │   ├── violent/           # Pose de vídeos violentos (treino)
│   │   │   │   │   └── <video_id>.npy # Shape: (num_frames, 33, 3)
│   │   │   │   └── non_violent/       # Pose de vídeos não violentos (treino)
│   │   │   │       └── <video_id>.npy
│   │   │   └── val/                   # Pose de vídeos (validação)
│   │   │       ├── violent/
│   │   │       └── non_violent/
│   │   └── ucf101/                    # Pose do UCF101 (opcional)
│   │       ├── train/
│   │       │   └── <class_name>/
│   │       └── test/
│   │
│   └── emotion/                       # Vetores de emoção (após extract_emotion.py)
│       └── rwf2000/                   # Emoções do RWF-2000
│           ├── train/
│           │   ├── violent/           # Emoções de vídeos violentos (treino)
│           │   │   └── <video_id>.npy # Shape: (num_frames, 8)
│           │   └── non_violent/       # Emoções de vídeos não violentos (treino)
│           │       └── <video_id>.npy
│           └── val/                   # Emoções de vídeos (validação)
│               ├── violent/
│               └── non_violent/
```

### Convenções de Nomenclatura

**Vídeos:**
- **Dataset original**: `dataset/RWF-2000/{split}/{Fight|NonFight}/<video_name>.avi`
- **Após organização**: `data/raw/{violent|non_violent}/<video_name>.avi`
- **ID do vídeo**: Nome do arquivo sem extensão (ex: `video_0001`)

**Frames:**
- **Estrutura**: `data/processed/{violent|non_violent}/<video_id>/frame_XXXX.jpg`
- **Formato**: `frame_0000.jpg`, `frame_0001.jpg`, ..., `frame_0015.jpg` (16 frames)

**Pose (Keypoints):**
- **Estrutura**: `data/pose/rwf2000/{split}/{violent|non_violent}/<video_id>.npy`
- **Formato**: Array NumPy com shape `(num_frames, 33, 3)`
  - 33 keypoints do MediaPipe
  - 3 valores: (x, y, visibility)
- **Exemplo**: `data/pose/rwf2000/train/violent/video_0001.npy`

**Emoção:**
- **Estrutura**: `data/emotion/rwf2000/{split}/{violent|non_violent}/<video_id>.npy`
- **Formato**: Array NumPy com shape `(num_frames, 8)`
  - 8 classes: [neutral, happy, sad, angry, fearful, disgust, surprise, contempt]
  - Valores: Probabilidades normalizadas (soma = 1.0)
- **Exemplo**: `data/emotion/rwf2000/train/violent/video_0001.npy`

### Validação da Estrutura

Para validar se sua estrutura de dados está correta, use o script de validação:

```bash
python tools/validate_data_structure.py
```

Este script verifica:
- ✅ Existência de diretórios obrigatórios
- ✅ Estrutura de pastas correta
- ✅ Correspondência entre modalidades (vídeo, pose, emoção)
- ✅ Formato e shape dos arquivos .npy
- ✅ Consistência de IDs de vídeo entre modalidades

## Estrutura do Projeto

```
cv-security-threat-detection/
├── dataset/                    # Datasets (não versionados)
│   ├── RWF-2000/              # Dataset principal (violência CCTV)
│   ├── UCF101/                # Dataset de pré-treinamento (9 classes)
│   └── AffectNet/             # Dataset de emoções (8 classes)
├── data/                      # Dados processados
│   ├── raw/                   # Vídeos organizados
│   ├── processed/             # Frames extraídos
│   ├── pose/                  # Keypoints de pose
│   └── emotion/               # Vetores de emoção
├── src/
│   ├── paths.py                # Gerenciamento centralizado de caminhos
│   ├── preprocessing/          # Pré-processamento
│   │   ├── organize_videos.py
│   │   └── extract_frames.py
│   ├── pose/                  # Extração de pose
│   │   ├── extract_pose.py
│   │   └── pose_dataset.py
│   ├── emotion/               # Extração de emoção
│   │   ├── extract_emotion.py
│   │   └── emotion_dataset.py
│   ├── datasets/              # Datasets e DataLoaders
│   │   ├── surveillance_dataset.py
│   │   ├── multimodal_dataset.py
│   │   └── video3d_dataset.py
│   ├── models/                # Modelos de Deep Learning
│   │   ├── resnet_lstm.py     # ResNet-LSTM
│   │   ├── cnn3d_risk.py      # CNN 3D
│   │   ├── emotion_cnn.py     # EmotionNet
│   │   └── multimodal_risk.py # MultimodalRiskDetector
│   ├── training/              # Scripts de treinamento
│   │   ├── train.py           # Treinamento ResNet-LSTM
│   │   └── utils.py           # Funções compartilhadas (run_epoch, dataloader, etc.)
│   ├── evaluation/             # Avaliação
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── ablation_study.py
│   │   └── robustness_eval.py
│   └── inference/              # Inferência
│       ├── realtime_risk_detector.py
│       └── multi_camera_detector.py
├── examples/                  # Scripts de exemplo
│   ├── example_usage.py
│   ├── example_cnn3d_usage.py
│   └── ... (outros exemplos)
├── tools/                     # Scripts utilitários
│   ├── filter_ucf101_classes.py
│   ├── validate_data_structure.py
│   └── ... (outros utilitários)
├── docs/                      # Documentação adicional
│   └── ANALISE_COMPLETA_PROJETO.md
├── results/                   # Resultados
│   ├── models/                # Modelos treinados
│   ├── multimodal/            # Modelos multimodais
│   ├── cnn3d/                 # Modelos CNN 3D
│   ├── emotion/               # Modelos de emoção
│   └── reports/               # Relatórios
├── local_docs/                # Documentação técnica detalhada
├── train_*.py                 # Scripts de treinamento (raiz)
├── run_*.py                   # Scripts de execução (raiz)
├── requirements.txt           # Dependências
└── README.md                  # Este arquivo
```

## Requisitos de Sistema

### Mínimos

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **RAM**: 8GB
- **Espaço em disco**: ~10GB (dataset + dados processados)
- **CPU**: Qualquer processador moderno

### Recomendados

- **GPU**: NVIDIA com CUDA compatível (para treinamento mais rápido)
- **RAM**: 16GB+
- **Espaço em disco**: 20GB+ (para múltiplos experimentos)
- **CPU**: Multi-core para pré-processamento paralelo

### Dependências Principais

- `torch>=2.0.0` - Framework de Deep Learning
- `torchvision>=0.15.0` - Modelos pré-treinados e transformações
- `opencv-python>=4.8.0` - Processamento de vídeo
- `numpy>=2.4.0` - Operações numéricas (Python 3.14 requer >=2.4.0)
- `scikit-learn>=1.3.0` - Métricas de avaliação
- `tqdm>=4.65.0` - Barras de progresso
- `Pillow>=10.0.0` - Processamento de imagens
- `mediapipe>=0.10.0` - Detecção de pose
- `facenet-pytorch>=2.5.0` - Detecção de faces
- `matplotlib>=3.7.0` - Plotagem de gráficos (curvas ROC, PR)
- `seaborn>=0.12.0` - Visualização de matriz de confusão

## Troubleshooting

### Problemas Comuns

**1. Erro de memória durante o treinamento**
- Solução: Reduza o `batch_size` para 2 ou 4
- Solução: Reduza o `num_frames` para 8

**2. Erro ao instalar numpy no Python 3.14 (compilação do código-fonte)**
- Problema: NumPy 2.2.x e anteriores tentam compilar do código-fonte no Python 3.14
- Solução: Use `pip install -r requirements.txt --prefer-binary` para forçar wheels pré-compilados
- Solução alternativa: O `requirements.txt` já especifica `numpy>=2.4.0`, que tem wheels para Python 3.14

**3. Dataset não encontrado**
- Verifique se o dataset está em `dataset/RWF-2000/`
- Verifique a estrutura de pastas (train/Fight, train/NonFight, etc.)

**4. Erro ao baixar pesos do ImageNet**
- Verifique sua conexão com a internet
- O PyTorch baixará automaticamente na primeira execução

**5. Treinamento muito lento**
- Use GPU se disponível: `--device cuda`
- Reduza `num_workers` se estiver usando CPU
- Considere reduzir `num_frames` ou `batch_size`

**6. Erro de importação de módulos**
- Certifique-se de que o ambiente virtual está ativado
- Reinstale as dependências: `pip install -r requirements.txt`


## Pipeline Completo de Execução

### Opção 1: Pipeline Automatizado (Recomendado)

Use o script master `train_pipeline.py` para automatizar todo o processo:

```bash
# Treinar tudo do zero (pipeline completo)
python train_pipeline.py --all

# Treinar apenas modelos base
python train_pipeline.py --base_models

# Treinar apenas multimodal (assumindo modelos base já existem)
python train_pipeline.py --multimodal

# Treinar com opções customizadas
python train_pipeline.py --all --skip_emotion --skip_cnn3d --epochs 30 --batch_size 4

# Forçar retreinamento mesmo se modelos já existirem
python train_pipeline.py --all --force_retrain
```

**Vantagens do script master:**
- ✅ Valida pré-requisitos automaticamente
- ✅ Detecta modelos já treinados e pergunta se deseja retreinar
- ✅ Orquestra toda a sequência de treinamento
- ✅ Fornece relatório detalhado ao final
- ✅ Trata erros e permite continuar com etapas opcionais

### Opção 2: Pipeline Manual (Passo a Passo)

Se preferir executar manualmente:

1. **Pré-processamento de Dados**
   ```bash
   # 1. Organizar vídeos
   python run_preprocessing.py
   
   # 2. Extrair frames
   # (já incluído no passo 1)
   
   # 3. Extrair pose
   python run_pose_preprocessing.py --dataset rwf2000
   
   # 4. Treinar EmotionNet e extrair emoções
   python train_emotion_model.py --dataset_path dataset/AffectNet
   python run_emotion_preprocessing.py --dataset rwf2000 --model_path models/emotion/best_model.pth
   ```

2. **Treinamento de Modelos Base**
   ```bash
   # 1. Treinar ResNet-LSTM
   python -m src.training.train --epochs 50
   
   # 2. Treinar EmotionNet (se ainda não tiver)
   python train_emotion_model.py --dataset_path dataset/AffectNet
   
   # 3. (Opcional) Pré-treinar CNN 3D em UCF101
   python train_cnn3d.py --stage pretrain --dataset ucf101
   python train_cnn3d.py --stage finetune --dataset rwf2000
   ```

3. **Treinamento Multimodal**
   ```bash
   python train_multimodal.py \
       --epochs 50 \
       --fusion_method late \
       --video_model_path results/models/best_model.pth \
       --emotion_model_path models/emotion/best_model.pth
   ```

4. **Avaliação**
   ```bash
   python run_evaluation.py \
       --model_path results/multimodal/best_model.pth
   ```

5. **Inferência em Tempo Real**
   ```bash
   python run_realtime_risk_detection.py \
       --multimodal_model results/multimodal/best_model.pth \
       --video_model results/models/best_model.pth \
       --emotion_model models/emotion/best_model.pth
   ```

### Ordem de Execução Recomendada

**Sequência completa:**
1. Pré-processamento → 2. Modelos Base → 3. Multimodal → 4. Avaliação → 5. Inferência

**Dependências:**
- Multimodal requer: ResNet-LSTM (obrigatório), EmotionNet (opcional), Pose e Emotion extraídos
- CNN 3D requer: UCF101 para pré-treinamento (opcional)
- EmotionNet requer: AffectNet para treinamento (opcional)

## Referências

### Datasets
- **RWF-2000**: Real-World Fighting Dataset para detecção de violência
- **UCF101**: Action Recognition Dataset (9 classes relevantes selecionadas)
- **AffectNet**: Facial Expression Dataset para reconhecimento de emoções

### Modelos e Arquiteturas
- **ResNet**: Deep Residual Learning for Image Recognition (He et al., 2015)
- **LSTM**: Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)
- **R(2+1)D**: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)
- **MediaPipe**: Framework de ML para detecção de pose

### Ferramentas
- **PyTorch**: Framework de Deep Learning
- **OpenCV**: Biblioteca de Visão Computacional
- **MediaPipe**: Detecção de pose e landmarks
