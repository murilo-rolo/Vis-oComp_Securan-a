"""
Script auxiliar para executar todo o pipeline de pré-processamento.

Este script:
1. Organiza os vídeos do RWF-2000
2. Extrai e processa os frames
"""

from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset

if __name__ == "__main__":
    print("="*50)
    print("PRÉ-PROCESSAMENTO DO DATASET RWF-2000")
    print("="*50)
    
    # Passo 1: Organizar vídeos
    print("\n[1/2] Organizando vídeos...")
    num_violent, num_non_violent = organize_rwf2000_dataset(
        dataset_root="dataset/RWF-2000",
        output_root="data/raw"
    )
    
    # Passo 2: Extrair frames
    print("\n[2/2] Extraindo e processando frames...")
    preprocess_dataset(
        raw_data_root="data/raw",
        processed_data_root="data/processed",
        num_frames=16,
        target_size=(112, 112),
        normalize=True
    )
    
    print("\n" + "="*50)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("="*50)

