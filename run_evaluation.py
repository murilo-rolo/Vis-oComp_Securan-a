"""
Script principal para executar todos os experimentos de avaliação.

Uso:
    # Avaliar baseline
    python run_evaluation.py --model baseline --model_path results/models/best_model.pth
    
    # Avaliar multimodal
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth
    
    # Executar todos os experimentos
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth --all
"""

import argparse
import torch
from pathlib import Path
import json

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.robustness_eval import RobustnessEvaluator, DEFAULT_DISTORTION_CONFIGS
from src.evaluation.performance_eval import PerformanceEvaluator
from src.evaluation.limitations_analysis import LimitationsAnalyzer
from src.evaluation.ablation_study import AblationStudy
from src.models.resnet_lstm import create_model as create_video_model
from src.models.multimodal_risk import create_multimodal_model
from src.datasets.surveillance_dataset import get_dataloaders
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src import paths as p


def load_model(model_type: str, model_path: str, device: str):
    """Carrega modelo do checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_type == "baseline":
        model = create_video_model(
            num_frames=16,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
            num_classes=2,
            pretrained=True,
            device=device
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    elif model_type == "multimodal":
        fusion_method = checkpoint.get('fusion_method', 'late')
        use_temporal = checkpoint.get('use_temporal_modeling', True)
        
        model = create_multimodal_model(
            video_feature_dim=256,
            pose_feature_dim=99,
            emotion_feature_dim=8,
            num_frames=16,
            fusion_method=fusion_method,
            use_temporal_modeling=use_temporal,
            device=device
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Executar experimentos de avaliação")
    
    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "multimodal"],
        required=True,
        help="Tipo de modelo"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Caminho para checkpoint do modelo"
    )
    parser.add_argument(
        "--video_model_path",
        type=str,
        default=None,
        help="Caminho para modelo de vídeo (ResNetLSTM) - necessário para multimodal com fusão 'late'"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    
    # Experimentos
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executar todos os experimentos"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calcular métricas básicas"
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Testar robustez"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Avaliar performance"
    )
    parser.add_argument(
        "--limitations",
        action="store_true",
        help="Analisar limitações"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Nome do experimento (padrão: tipo do modelo)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para inferência"
    )
    
    args = parser.parse_args()
    
    # Definir experimentos a executar
    if args.all:
        run_metrics = True
        run_robustness = True
        run_performance = True
        run_limitations = True
    else:
        run_metrics = args.metrics
        run_robustness = args.robustness
        run_performance = args.performance
        run_limitations = args.limitations
        
        # Se nenhum especificado, executar métricas básicas
        if not any([run_metrics, run_robustness, run_performance, run_limitations]):
            run_metrics = True
    
    # Nome do experimento
    if args.experiment_name is None:
        args.experiment_name = args.model
    
    print("=" * 60)
    print("Advanced Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Dir: {p.EXPERIMENTS_ROOT}")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Device: {args.device}")
    print()
    print("Experiments to run:")
    print(f"  - Metrics: {run_metrics}")
    print(f"  - Robustness: {run_robustness}")
    print(f"  - Performance: {run_performance}")
    print(f"  - Limitations: {run_limitations}")
    print("=" * 60)
    print()
    
    # Carregar modelo
    print("Loading model...")
    model = load_model(args.model, args.model_path, args.device)
    print("✓ Model loaded")
    
    # Carregar dataset
    print("Loading dataset...")
    if args.model == "baseline":
        _, _, test_loader = get_dataloaders(
            processed_data_root=str(p.PROCESSED_ROOT),
            batch_size=args.batch_size,
            num_frames=16
        )
    elif args.model == "multimodal":
        _, _, test_loader = get_multimodal_dataloaders(
            video_data_root=str(p.PROCESSED_ROOT),
            pose_data_root=str(p.POSE_ROOT),
            emotion_data_root=str(p.EMOTION_ROOT),
            batch_size=args.batch_size,
            num_frames=16,
            window_size=16,
            video_mode="frames",
            pose_mode="keypoints"
        )
    print(f"✓ Dataset loaded ({len(test_loader)} batches)")
    
    # Executar experimentos
    results = {}
    
    # 1. Métricas básicas
    if run_metrics:
        print("\n" + "=" * 60)
        print("1. Calculating Basic Metrics")
        print("=" * 60)
        calculator = MetricsCalculator(
            model, 
            test_loader, 
            device=args.device,
            video_model_path=args.video_model_path
        )
        metrics, y_true, y_pred, y_proba = calculator.evaluate()
        results["metrics"] = metrics
        
        # Salvar resultados
        output_path = p.EXPERIMENTS_ROOT / args.experiment_name / "metrics"
        output_path.mkdir(parents=True, exist_ok=True)
        calculator.save_results(
            str(output_path),
            "baseline" if args.model == "baseline" else "multimodal",
            metrics,
            y_true,
            y_pred,
            y_proba
        )
        print("✓ Metrics saved")
    
    # 2. Robustez
    if run_robustness:
        print("\n" + "=" * 60)
        print("2. Testing Robustness")
        print("=" * 60)
        robustness_eval = RobustnessEvaluator(model, test_loader, device=args.device)
        robustness_results = robustness_eval.evaluate_all_distortions(
            DEFAULT_DISTORTION_CONFIGS,
            str(p.EXPERIMENTS_ROOT),
            f"{args.experiment_name}/robustness"
        )
        results["robustness"] = robustness_results
        print("✓ Robustness tests completed")
    
    # 3. Performance
    if run_performance:
        print("\n" + "=" * 60)
        print("3. Evaluating Performance")
        print("=" * 60)
        perf_eval = PerformanceEvaluator(model, test_loader, device=args.device)
        perf_results = perf_eval.evaluate_all(
            str(p.EXPERIMENTS_ROOT),
            f"{args.experiment_name}/performance"
        )
        results["performance"] = perf_results
        print("✓ Performance evaluation completed")
    
    # 4. Limitações
    if run_limitations:
        print("\n" + "=" * 60)
        print("4. Analyzing Limitations")
        print("=" * 60)
        limitations_analyzer = LimitationsAnalyzer(model, test_loader, device=args.device)
        limitations_results = limitations_analyzer.generate_error_report(
            str(p.EXPERIMENTS_ROOT),
            f"{args.experiment_name}/limitations"
        )
        results["limitations"] = limitations_results
        print("✓ Limitations analysis completed")
    
    # Salvar resumo geral
    summary_path = p.EXPERIMENTS_ROOT / args.experiment_name / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to: {p.EXPERIMENTS_ROOT / args.experiment_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

