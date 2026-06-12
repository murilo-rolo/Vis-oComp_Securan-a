"""
Script auxiliar para executar todo o pipeline de pré-processamento.

Este script:
1. Organiza os vídeos do RWF-2000
2. Extrai e processa os frames
"""

from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Organiza o diretório de arquivos do dataset"
    )

    parser.add_argument(
        "--dataset_root",
        type = str,
        default="dataset/RWF-2000",
        help="Determina o diretório no qual está os datasets"
    )

    parser.add_argument(
        "--dataset_output",
        type=str,
        default="data/raw",
        help="Determina o diretório para o qual estarão os datasets sem serem processados"
    )

    parser.add_argument(
        "--dataset_processed",
        type=str,
        default="data/processed",
        help="Determina para qual diretório irá os dados que já tiveram seus frames extraídos"
    )

    args = parser.parse_args()

    print("="*50)
    print("PRÉ-PROCESSAMENTO DO DATASET RWF-2000")
    print("="*50)
    
    # Passo 1: Organizar vídeos
    print("\n[1/2] Organizando vídeos...")
    num_violent, num_non_violent = organize_rwf2000_dataset(
        dataset_root=args.dataset_root,
        output_root=args.dataset_output
    )
    
    # Passo 2: Extrair frames
    print("\n[2/2] Extraindo e processando frames...")
    preprocess_dataset(
        raw_data_root=args.dataset_output,
        processed_data_root=args.dataset_processed,
        num_frames=16,
        target_size=(112, 112),
        normalize=True
    )
    
    print("\n" + "="*50)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("="*50)

