"""
Script principal para detecção de risco em tempo real.

Este script executa o pipeline completo de inferência em tempo real:
1. Captura vídeo de webcam ou RTSP
2. Processa janelas temporais
3. Extrai features multimodais
4. Gera alertas
5. Exibe resultado com overlay

Uso:
    # Webcam
    python run_realtime_risk_detection.py --multimodal_model results/multimodal/best_model.pth
    
    # RTSP
    python run_realtime_risk_detection.py --multimodal_model results/multimodal/best_model.pth --source rtsp://...
    
    # Com CNN 3D
    python run_realtime_risk_detection.py --multimodal_model results/multimodal/best_model.pth --use_cnn3d --cnn3d_model results/cnn3d/rwf2000/best_model.pth
"""

import argparse
import torch
from src.inference.realtime_risk_detector import create_realtime_detector


def main():
    parser = argparse.ArgumentParser(
        description="Detecção de risco em tempo real usando modelo multimodal"
    )
    
    # Modelos
    parser.add_argument(
        "--multimodal_model",
        type=str,
        required=True,
        help="Caminho para modelo multimodal treinado"
    )
    parser.add_argument(
        "--video_model",
        type=str,
        default=None,
        help="Caminho para modelo de vídeo (ResNet-LSTM) - opcional se usar CNN 3D"
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        default=None,
        help="Caminho para modelo de emoção - opcional"
    )
    parser.add_argument(
        "--use_cnn3d",
        action="store_true",
        help="Usar CNN 3D ao invés de ResNet-LSTM para vídeo"
    )
    parser.add_argument(
        "--cnn3d_model",
        type=str,
        default=None,
        help="Caminho para modelo CNN 3D (requerido se --use_cnn3d)"
    )
    
    # Fonte de vídeo
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fonte de vídeo: '0' para webcam ou URL RTSP (padrão: '0')"
    )
    
    # Configuração de processamento
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Tamanho da janela temporal (padrão: 16)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=8,
        help="Sobreposição entre janelas (padrão: 8)"
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Tamanho dos frames para processamento (H W) - padrão: 224 224"
    )
    
    # Alertas
    parser.add_argument(
        "--risk_threshold",
        type=float,
        default=0.8,
        help="Threshold de probabilidade para alerta (padrão: 0.8)"
    )
    parser.add_argument(
        "--consecutive_windows",
        type=int,
        default=3,
        help="Número de janelas consecutivas acima do threshold para alerta (padrão: 3)"
    )
    
    # Display
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Não exibir vídeo (apenas processar)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para inferência (padrão: 'cuda' se disponível)"
    )
    
    args = parser.parse_args()
    
    # Validações
    if args.use_cnn3d and not args.cnn3d_model:
        parser.error("--cnn3d_model é obrigatório quando --use_cnn3d é usado")
    
    if not args.use_cnn3d and not args.video_model:
        print("⚠ Aviso: --video_model não fornecido. Usando modelo ResNet-LSTM sem checkpoint.")
    
    print("=" * 60)
    print("Pipeline de Detecção de Risco em Tempo Real")
    print("=" * 60)
    print(f"Fonte de vídeo: {args.source}")
    print(f"Modelo multimodal: {args.multimodal_model}")
    print(f"Backbone de vídeo: {'CNN 3D' if args.use_cnn3d else 'ResNet-LSTM'}")
    print(f"Janela temporal: {args.window_size} frames")
    print(f"Sobreposição: {args.overlap} frames")
    print(f"Threshold de risco: {args.risk_threshold}")
    print(f"Janelas consecutivas: {args.consecutive_windows}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print()
    
    # Criar detector
    try:
        detector = create_realtime_detector(
            multimodal_model_path=args.multimodal_model,
            video_model_path=args.video_model,
            emotion_model_path=args.emotion_model,
            video_source=args.source,
            window_size=args.window_size,
            risk_threshold=args.risk_threshold,
            consecutive_windows=args.consecutive_windows,
            use_cnn3d=args.use_cnn3d,
            cnn3d_model_path=args.cnn3d_model,
            device=args.device
        )
    except Exception as e:
        print(f"Erro ao criar detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Executar pipeline
    try:
        detector.run(display=not args.no_display)
    except KeyboardInterrupt:
        print("\nPipeline interrompido pelo usuário")
    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

