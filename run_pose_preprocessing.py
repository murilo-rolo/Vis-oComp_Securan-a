"""
Script principal para pré-processamento de pose estimation.

Este script extrai keypoints de pose de vídeos dos datasets UCF101 e RWF-2000
e salva os resultados em arquivos .npy organizados para uso posterior.

Uso:
    # Processar RWF-2000
    python run_pose_preprocessing.py --dataset rwf2000 --num_frames 16
    
    # Processar UCF101
    python run_pose_preprocessing.py --dataset ucf101 --num_frames 16
    
    # Processar ambos
    python run_pose_preprocessing.py --dataset both --num_frames 16
"""

import argparse
from pathlib import Path
from src.pose.extract_pose import process_dataset_for_pose


def main():
    parser = argparse.ArgumentParser(
        description="Extrai keypoints de pose de vídeos dos datasets UCF101 e RWF-2000"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ucf101", "rwf2000", "both"],
        default="both",
        help="Dataset a processar: 'ucf101', 'rwf2000' ou 'both'"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset",
        help="Diretório raiz dos datasets (padrão: 'dataset')"
    )
    
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/pose",
        help="Diretório raiz de saída para keypoints (padrão: 'data/pose')"
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Confiança mínima para detecção inicial (padrão: 0.5, range: 0.0-1.0). "
             "Valores mais altos (0.7-0.9) reduzem falsos positivos mas podem perder detecções válidas. "
             "Recomendado: 0.5-0.7 para melhor balanceamento."
    )
    
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Confiança mínima para rastreamento (padrão: 0.5, range: 0.0-1.0). "
             "Valores mais altos (0.7-0.9) melhoram estabilidade mas podem perder rastreamento em movimentos rápidos. "
             "Recomendado: 0.5-0.7 para melhor balanceamento."
    )
    
    parser.add_argument(
        "--model_complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Complexidade do modelo MediaPipe (padrão: 1). "
             "0=Lite (mais rápido, menos preciso), "
             "1=Full (balanceado), "
             "2=Heavy (mais lento, mais preciso). "
             "Recomendado: 2 para máxima precisão em detecção de ameaças."
    )
    
    args = parser.parse_args()
    
    # Validar limites dos parâmetros de confiança
    if not (0.0 <= args.min_detection_confidence <= 1.0):
        parser.error("--min_detection_confidence deve estar entre 0.0 e 1.0")
    
    if not (0.0 <= args.min_tracking_confidence <= 1.0):
        parser.error("--min_tracking_confidence deve estar entre 0.0 e 1.0")
    
    # Validar num_frames se fornecido
    if args.num_frames is not None and args.num_frames <= 0:
        parser.error("--num_frames deve ser um número positivo ou None para processar todos os frames")
    
    # Validar diretórios
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Erro: Diretório de datasets não encontrado: {dataset_root}")
        print("  Certifique-se de que os datasets estão em 'dataset/UCF101' e 'dataset/RWF-2000'")
        return
    
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Pré-processamento de Pose Estimation")
    print("=" * 60)
    print(f"Dataset raiz: {dataset_root}")
    print(f"Saída raiz: {output_root}")
    print(f"Número de frames: {args.num_frames if args.num_frames else 'Todos'}")
    print(f"Confiança detecção: {args.min_detection_confidence} (range: 0.0-1.0)")
    print(f"Confiança rastreamento: {args.min_tracking_confidence} (range: 0.0-1.0)")
    print(f"Complexidade modelo: {args.model_complexity} (0=Lite, 1=Full, 2=Heavy)")
    
    # Avisos sobre valores aumentados
    if args.min_detection_confidence > 0.7:
        print(f"⚠️  AVISO: Confiança de detecção alta ({args.min_detection_confidence}) pode reduzir detecções válidas")
    if args.min_tracking_confidence > 0.7:
        print(f"⚠️  AVISO: Confiança de rastreamento alta ({args.min_tracking_confidence}) pode perder rastreamento em movimentos rápidos")
    if args.model_complexity == 2:
        print(f"ℹ️  INFO: Modelo Heavy (complexidade 2) será mais lento mas mais preciso")
    if args.num_frames and args.num_frames > 32:
        print(f"ℹ️  INFO: Processando {args.num_frames} frames por vídeo (pode aumentar tempo de processamento)")
    
    print("=" * 60)
    print()
    
    # Processar datasets
    if args.dataset in ["ucf101", "both"]:
        ucf101_path = dataset_root / "UCF101"
        if ucf101_path.exists():
            print("\n" + "=" * 60)
            print("Processando UCF101...")
            print("=" * 60)
            process_dataset_for_pose(
                dataset_root=str(ucf101_path),
                output_root=str(output_root),
                dataset_name="ucf101",
                num_frames=args.num_frames,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                model_complexity=args.model_complexity
            )
        else:
            print(f"\nAviso: Dataset UCF101 não encontrado em {ucf101_path}")
            print("  Pulando processamento de UCF101...")
    
    if args.dataset in ["rwf2000", "both"]:
        rwf2000_path = dataset_root / "RWF-2000"
        if rwf2000_path.exists():
            print("\n" + "=" * 60)
            print("Processando RWF-2000...")
            print("=" * 60)
            process_dataset_for_pose(
                dataset_root=str(rwf2000_path),
                output_root=str(output_root),
                dataset_name="rwf2000",
                num_frames=args.num_frames,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                model_complexity=args.model_complexity
            )
        else:
            print(f"\nAviso: Dataset RWF-2000 não encontrado em {rwf2000_path}")
            print("  Pulando processamento de RWF-2000...")
    
    print("\n" + "=" * 60)
    print("Pré-processamento concluído!")
    print("=" * 60)
    print(f"\nKeypoints salvos em: {output_root}")
    print("\nEstrutura criada:")
    print("  data/pose/")
    if args.dataset in ["ucf101", "both"]:
        print("    ucf101/")
        print("      train/<classe>/<video>.npy")
        print("      test/<classe>/<video>.npy")
    if args.dataset in ["rwf2000", "both"]:
        print("    rwf2000/")
        print("      train/violent/<video>.npy")
        print("      train/non_violent/<video>.npy")
        print("      val/violent/<video>.npy")
        print("      val/non_violent/<video>.npy")
    print("\nPróximos passos:")
    print("  1. Use PoseSequenceDataset para carregar os dados em PyTorch")
    print("  2. Treine um modelo temporal (LSTM, Transformer) com os keypoints")
    print("  3. Combine com features de ResNet para sistema multimodal")


if __name__ == "__main__":
    main()

