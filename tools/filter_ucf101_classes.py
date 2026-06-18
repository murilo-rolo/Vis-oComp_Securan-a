"""
Script para filtrar o dataset UCF101 mantendo apenas as classes relevantes.

Classes a manter:
- RELEVANTES (6): BoxingPunchingBag, BoxingSpeedBag, Fencing, Nunchucks, Punch, SumoWrestling
- OPCIONAIS SELECIONADAS (3): Archery, CliffDiving, MilitaryParade

Total: 9 classes (de 101)
"""

import shutil

from src import paths as p

# Classes a manter
CLASSES_TO_KEEP = [
    # Classes RELEVANTES
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "Fencing",
    "Nunchucks",
    "Punch",
    "SumoWrestling",
    # Classes OPCIONAIS selecionadas
    "Archery",
    "CliffDiving",
    "MilitaryParade"
]

# Caminho do dataset
DATASET_ROOT = p.UCF101_ROOT
SPLITS = ["train", "test", "val"]


def get_all_classes_in_dataset():
    """Retorna todas as classes presentes no dataset."""
    all_classes = set()
    
    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    all_classes.add(class_dir.name)
    
    return sorted(all_classes)


def filter_csv_file(csv_path: Path, classes_to_keep: list):
    """Filtra arquivo CSV mantendo apenas as classes desejadas."""
    if not csv_path.exists():
        print(f"[AVISO] Arquivo nao encontrado: {csv_path}")
        return
    
    # Ler CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"[AVISO] Arquivo vazio: {csv_path}")
        return
    
    # Header
    header = lines[0]
    
    # Filtrar linhas
    initial_count = len(lines) - 1  # Excluir header
    filtered_lines = [header]
    
    for line in lines[1:]:
        # Extrair label (última coluna)
        parts = line.strip().split(',')
        if len(parts) >= 3:
            label = parts[-1].strip()
            if label in classes_to_keep:
                filtered_lines.append(line)
    
    final_count = len(filtered_lines) - 1  # Excluir header
    
    # Salvar CSV filtrado
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
    
    # Contar classes únicas
    unique_classes = set()
    for line in filtered_lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            unique_classes.add(parts[-1].strip())
    
    print(f"[OK] {csv_path.name}: {initial_count} -> {final_count} linhas ({len(unique_classes)} classes)")


def remove_class_directories(classes_to_remove: list):
    """Remove diretórios das classes não desejadas."""
    removed_count = 0
    
    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in classes_to_remove:
                try:
                    shutil.rmtree(class_dir)
                    removed_count += 1
                    print(f"[OK] Removido: {split}/{class_dir.name}")
                except Exception as e:
                    print(f"[ERRO] Erro ao remover {class_dir}: {e}")
    
    return removed_count


def validate_filtering(classes_to_keep: list):
    """Valida que apenas as classes desejadas permanecem."""
    print("\n" + "="*60)
    print("VALIDAÇÃO DO FILTRO")
    print("="*60)
    
    remaining_classes = set()
    
    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            continue
        
        split_classes = set()
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                split_classes.add(class_dir.name)
                remaining_classes.add(class_dir.name)
        
        print(f"\n{split.upper()}:")
        print(f"  Classes encontradas: {len(split_classes)}")
        print(f"  Classes: {sorted(split_classes)}")
    
    # Verificar se há classes extras
    extra_classes = remaining_classes - set(classes_to_keep)
    missing_classes = set(classes_to_keep) - remaining_classes
    
    print(f"\n{'='*60}")
    print(f"Total de classes restantes: {len(remaining_classes)}")
    
    if extra_classes:
        print(f"[AVISO] Classes extras encontradas: {sorted(extra_classes)}")
    else:
        print("[OK] Nenhuma classe extra encontrada")
    
    if missing_classes:
        print(f"[AVISO] Classes esperadas nao encontradas: {sorted(missing_classes)}")
    else:
        print("[OK] Todas as classes esperadas estao presentes")
    
    print(f"\nClasses mantidas ({len(remaining_classes)}):")
    for cls in sorted(remaining_classes):
        print(f"  - {cls}")


def main():
    """Função principal."""
    global DATASET_ROOT
    DATASET_ROOT = p.UCF101_ROOT

    print("="*60)
    print("FILTRO DE CLASSES UCF101")
    print("="*60)
    print(f"\nClasses a manter ({len(CLASSES_TO_KEEP)}):")
    for cls in CLASSES_TO_KEEP:
        print(f"  - {cls}")
    
    # Obter todas as classes do dataset
    all_classes = get_all_classes_in_dataset()
    classes_to_remove = [cls for cls in all_classes if cls not in CLASSES_TO_KEEP]
    
    print(f"\nTotal de classes no dataset: {len(all_classes)}")
    print(f"Classes a remover: {len(classes_to_remove)}")
    
    print("\n" + "="*60)
    print("INICIANDO FILTRO...")
    print("="*60)
    
    # 1. Filtrar arquivos CSV
    print("\n[1/3] Filtrando arquivos CSV...")
    for split in SPLITS:
        csv_path = DATASET_ROOT / f"{split}.csv"
        filter_csv_file(csv_path, CLASSES_TO_KEEP)
    
    # 2. Remover diretórios das classes não desejadas
    print("\n[2/3] Removendo diretórios das classes não desejadas...")
    removed_count = remove_class_directories(classes_to_remove)
    print(f"[OK] Total de diretorios removidos: {removed_count}")
    
    # 3. Validar resultado
    print("\n[3/3] Validando resultado...")
    validate_filtering(CLASSES_TO_KEEP)
    
    print("\n" + "="*60)
    print("FILTRO CONCLUÍDO!")
    print("="*60)
    print(f"\nResumo:")
    print(f"  - Classes mantidas: {len(CLASSES_TO_KEEP)}")
    print(f"  - Classes removidas: {len(classes_to_remove)}")
    print(f"  - Redução: {len(classes_to_remove)/len(all_classes)*100:.1f}%")


if __name__ == "__main__":
    main()

