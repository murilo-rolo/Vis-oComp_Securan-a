"""
Estudo de ablação para entender contribuição de cada modalidade.
"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
import json

from .metrics import MetricsCalculator
from .utils import save_results, create_experiment_dir


class AblationStudy:
    """
    Estudo de ablação para modelos multimodais.
    """
    
    def __init__(
        self,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa estudo de ablação.
        
        Args:
            dataloader: DataLoader com dados de teste
            device: Device para inferência
        """
        self.dataloader = dataloader
        self.device = torch.device(device)
    
    def evaluate_configuration(
        self,
        model,
        config_name: str
    ) -> Dict:
        """
        Avalia uma configuração específica.
        
        Args:
            model: Modelo para avaliar
            config_name: Nome da configuração
        
        Returns:
            Dicionário com métricas
        """
        calculator = MetricsCalculator(model, self.dataloader, device=self.device)
        metrics, y_true, y_pred, y_proba = calculator.evaluate()
        
        return {
            "config": config_name,
            "metrics": metrics
        }
    
    def run_ablation(
        self,
        models: Dict[str, torch.nn.Module],
        output_dir: str,
        experiment_name: str = "ablation"
    ) -> Dict:
        """
        Executa estudo de ablação completo.
        
        Args:
            models: Dict {config_name: model}
            output_dir: Diretório de saída
            experiment_name: Nome do experimento
        
        Returns:
            Dicionário com resultados de todas as configurações
        """
        output_path = create_experiment_dir(output_dir, experiment_name)
        results = {}
        
        print("=" * 60)
        print("Running Ablation Study")
        print("=" * 60)
        
        for config_name, model in models.items():
            print(f"\nEvaluating: {config_name}")
            result = self.evaluate_configuration(model, config_name)
            results[config_name] = result
            
            # Salvar resultado individual
            save_results(result, output_path / f"{config_name}_metrics.json")
        
        # Comparação
        comparison = self._compare_configurations(results)
        results["comparison"] = comparison
        
        # Salvar comparação
        save_results(comparison, output_path / "ablation_comparison.json")
        
        # Salvar resumo completo
        save_results(results, output_path / "ablation_summary.json")
        
        return results
    
    def _compare_configurations(self, results: Dict) -> Dict:
        """Compara diferentes configurações."""
        comparison = {
            "configurations": list(results.keys()),
            "metrics_comparison": {}
        }
        
        # Extrair métricas de cada configuração
        metric_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr"]
        
        for metric_name in metric_names:
            comparison["metrics_comparison"][metric_name] = {}
            
            for config_name, result in results.items():
                metrics = result["metrics"]
                
                if metric_name in metrics:
                    if isinstance(metrics[metric_name], dict):
                        # Métricas por classe
                        comparison["metrics_comparison"][metric_name][config_name] = {
                            "macro": metrics[metric_name].get("macro"),
                            "weighted": metrics[metric_name].get("weighted")
                        }
                    else:
                        comparison["metrics_comparison"][metric_name][config_name] = metrics[metric_name]
        
        # Identificar melhor configuração por métrica
        best_configs = {}
        for metric_name, config_values in comparison["metrics_comparison"].items():
            if config_values:
                # Encontrar melhor valor
                best_config = max(config_values.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else x[1].get("macro", 0))
                best_configs[metric_name] = {
                    "best_config": best_config[0],
                    "value": best_config[1]
                }
        
        comparison["best_configurations"] = best_configs
        
        return comparison

