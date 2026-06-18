"""
Script auxiliar para executar todo o pipeline de pré-processamento.

Este script:
1. Organiza os vídeos do RWF-2000
2. Extrai e processa os frames
"""

from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset
from src import paths as p
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Organiza o diretório de arquivos do dataset"
    )

    parser.add_argument(
        "--dataset_root",
        type = str,
        default=None,
        help="Determina o diretório no qual está os datasets"
    )

    parser.add_argument(
        "--dataset_output",
        type=str,
        default=None,
        help="Determina o diretório para o qual estarão os datasets sem serem processados"
    )

    parser.add_argument(
        "--dataset_processed",
        type=str,
        default=None,
        help="Determina para qual diretório irá os dados que já tiveram seus frames extraídos"
    )

    args = parser.parse_args()
    if args.dataset_root is None:
        args.dataset_root = str(p.RWF2000_ROOT)
    if args.dataset_output is None:
        args.dataset_output = str(p.RAW_DATA_ROOT)
    if args.dataset_processed is None:
        args.dataset_processed = str(p.PROCESSED_ROOT)

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

