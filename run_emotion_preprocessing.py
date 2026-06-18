"""
Script principal para pré-processamento de Emotion Recognition.

Este script extrai vetores de emoção de vídeos do dataset RWF-2000
e salva os resultados em arquivos .npy organizados para uso posterior.

Uso:
    # Processar RWF-2000 com modelo pré-treinado
    python run_emotion_preprocessing.py --model_path models/emotion_model.pth --num_frames 16
    
    # Processar sem modelo pré-treinado (usa ImageNet weights)
    python run_emotion_preprocessing.py --num_frames 16
"""

import argparse
import torch
from src.models.emotion_cnn import create_emotion_model
from src.emotion.extract_emotion import process_dataset_for_emotion
from src import paths as p


def main():
    parser = argparse.ArgumentParser(
        description="Extrai vetores de emoção de vídeos do dataset RWF-2000"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["rwf2000"],
        default="rwf2000",
        help="Dataset a processar (padrão: 'rwf2000')"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Caminho para modelo EmotionNet pré-treinado (opcional, usa ImageNet weights se não fornecido)"
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    
    parser.add_argument(
        "--face_detector",
        type=str,
        choices=["mtcnn", "retinaface", "haar"],
        default="mtcnn",
        help="Método de detecção de faces (padrão: 'mtcnn')"
    )
    
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Método de agregação temporal (padrão: 'mean')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para processamento (padrão: 'cuda' se disponível, senão 'cpu')"
    )
    
    args = parser.parse_args()
    
    # Validar diretórios
    if not p.DATASET_ROOT.exists():
        print(f"Erro: Diretório de datasets não encontrado: {p.DATASET_ROOT}")
        print("  Certifique-se de que o dataset está em 'dataset/RWF-2000'")
        return
    
    p.EMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Pré-processamento de Emotion Recognition")
    print("=" * 60)
    print(f"Dataset raiz: {p.DATASET_ROOT}")
    print(f"Saída raiz: {p.EMOTION_ROOT}")
    print(f"Número de frames: {args.num_frames if args.num_frames else 'Todos'}")
    print(f"Detector de faces: {args.face_detector}")
    print(f"Agregação: {args.aggregation}")
    print(f"Device: {args.device}")
    print(f"Modelo: {args.model_path if args.model_path else 'ImageNet weights (não treinado)'}")
    print("=" * 60)
    print()
    
    # Carregar modelo
    print("Carregando modelo de emoção...")
    try:
        model = create_emotion_model(
            num_emotions=8,
            pretrained=True,
            checkpoint_path=args.model_path,
            device=args.device
        )
        print(f"Modelo carregado com sucesso!")
        if args.model_path:
            print(f"  Checkpoint: {args.model_path}")
        else:
            print(f"  Usando pesos ImageNet (modelo não treinado em emoções)")
            print(f"  ⚠️  Para melhor performance, treine o modelo no AffectNet primeiro!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    # Processar dataset
    if args.dataset == "rwf2000":
        rwf2000_path = p.DATASET_ROOT / "RWF-2000"
        if rwf2000_path.exists():
            print("\n" + "=" * 60)
            print("Processando RWF-2000...")
            print("=" * 60)
            process_dataset_for_emotion(
                dataset_root=str(rwf2000_path),
                output_root=str(p.EMOTION_ROOT),
                model=model,
                dataset_name="rwf2000",
                num_frames=args.num_frames,
                face_detector_method=args.face_detector,
                aggregation=args.aggregation
            )
        else:
            print(f"\nAviso: Dataset RWF-2000 não encontrado em {rwf2000_path}")
            print("  Pulando processamento...")
    
    print("\n" + "=" * 60)
    print("Pré-processamento concluído!")
    print("=" * 60)
    print(f"\nVetores de emoção salvos em: {p.EMOTION_ROOT}")
    print("\nEstrutura criada:")
    print("  data/emotion/")
    print("    rwf2000/")
    print("      train/violent/<video>.npy")
    print("      train/non_violent/<video>.npy")
    print("      val/violent/<video>.npy")
    print("      val/non_violent/<video>.npy")
    print("\nPróximos passos:")
    print("  1. Use EmotionSequenceDataset para carregar os dados em PyTorch")
    print("  2. Treine um modelo temporal (LSTM, Transformer) com os vetores de emoção")
    print("  3. Combine com features de ResNet e Pose para sistema multimodal")


if __name__ == "__main__":
    main()

