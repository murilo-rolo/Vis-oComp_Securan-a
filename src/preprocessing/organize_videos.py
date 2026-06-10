"""
Módulo para organizar vídeos do dataset RWF-2000 nas pastas corretas.

Este script lê os vídeos da pasta dataset/RWF-2000 e os organiza em:
- data/raw/violent/ (vídeos da classe Fight)
- data/raw/non_violent/ (vídeos da classe NonFight)
"""

import os
import shutil
from pathlib import Path
from typing import Tuple
import sys


def organize_rwf2000_dataset(
    dataset_root: str = "dataset/RWF-2000",
    output_root: str = "data/raw"
) -> Tuple[int, int]:
    """
    Organiza os vídeos do RWF-2000 nas pastas violent e non_violent.
    
    Args:
        dataset_root: Caminho raiz do dataset RWF-2000
        output_root: Caminho raiz de saída (data/raw)
    
    Returns:
        Tupla (num_violent, num_non_violent) com contagem de vídeos organizados
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    
    # Criar pastas de destino se não existirem
    violent_dir = output_path / "violent"
    non_violent_dir = output_path / "non_violent"
    violent_dir.mkdir(parents=True, exist_ok=True)
    non_violent_dir.mkdir(parents=True, exist_ok=True)
    
    # Contadores
    num_violent = 0
    num_non_violent = 0
    num_errors = 0
    
    def safe_copy_file(src_file: Path, dest_file: Path, counter: int) -> bool:
        """
        Copia arquivo de forma segura, lidando com problemas de encoding.
        
        Args:
            src_file: Arquivo de origem
            dest_file: Arquivo de destino
            counter: Contador para gerar nome único se necessário
        
        Returns:
            True se copiou com sucesso, False caso contrário
        """
        try:
            # Converter para string para evitar problemas com Path e encoding
            try:
                src_str = str(src_file.resolve())
            except Exception:
                # Se resolve() falhar, tentar direto
                src_str = str(src_file)
            
            # Verificar se arquivo de origem existe
            if not os.path.exists(src_str):
                return False
            
            # Tentar copiar com nome original
            if not dest_file.exists():
                try:
                    dest_str = str(dest_file.resolve())
                except Exception:
                    dest_str = str(dest_file)
                
                # Criar diretório de destino se não existir
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copiar arquivo
                shutil.copy2(src_str, dest_str)
                return True
            return True  # Já existe
        except (OSError, UnicodeEncodeError, UnicodeDecodeError, PermissionError) as e:
            # Se falhar, tentar com nome seguro
            try:
                # Gerar nome seguro baseado no contador
                safe_name = f"video_{counter:06d}.avi"
                dest_safe = dest_file.parent / safe_name
                
                if not dest_safe.exists():
                    try:
                        src_str = str(src_file.resolve())
                    except Exception:
                        src_str = str(src_file)
                    
                    if os.path.exists(src_str):
                        dest_safe_str = str(dest_safe.resolve())
                        shutil.copy2(src_str, dest_safe_str)
                        print(f"  Arquivo com encoding problemático renomeado: {safe_name}")
                        return True
                return True
            except Exception as e2:
                # Último recurso: tentar com caminho absoluto direto
                try:
                    src_abs = os.path.abspath(str(src_file))
                    if os.path.exists(src_abs):
                        safe_name = f"video_{counter:06d}.avi"
                        dest_abs = os.path.join(str(dest_file.parent.resolve()), safe_name)
                        if not os.path.exists(dest_abs):
                            shutil.copy2(src_abs, dest_abs)
                            return True
                except:
                    pass
                return False
    
    # Processar pastas train e val
    for split in ["train", "val"]:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"Aviso: Pasta {split_path} não encontrada. Pulando...")
            continue
        
        # Processar vídeos Fight (violent)
        fight_dir = split_path / "Fight"
        if fight_dir.exists():
            print(f"Processando vídeos Fight de {split}...")
            
            # Tentar usar glob primeiro, se falhar usar os.listdir
            try:
                video_files = list(fight_dir.glob("*.avi"))
            except Exception:
                # Fallback: usar os.listdir que lida melhor com encoding no Windows
                try:
                    fight_dir_str = str(fight_dir.resolve())
                    video_files = [
                        fight_dir / fname 
                        for fname in os.listdir(fight_dir_str) 
                        if fname.lower().endswith('.avi')
                    ]
                except Exception as e:
                    print(f"  Erro ao listar arquivos: {str(e)[:100]}")
                    video_files = []
            
            print(f"  Encontrados {len(video_files)} vídeos")
            
            for idx, video_file in enumerate(video_files, 1):
                try:
                    # Tentar obter nome do arquivo de forma segura
                    try:
                        video_name = video_file.name
                    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
                        # Se falhar, usar nome baseado no índice
                        video_name = f"video_{num_violent + idx:06d}.avi"
                    
                    # Garantir que temos um Path válido
                    if isinstance(video_file, str):
                        video_file = Path(video_file)
                    
                    dest_path = violent_dir / video_name
                    
                    if safe_copy_file(video_file, dest_path, num_violent + idx):
                        num_violent += 1
                    else:
                        num_errors += 1
                        
                    if idx % 100 == 0:
                        print(f"  Processados {idx}/{len(video_files)} vídeos...")
                except Exception as e:
                    print(f"  Erro ao processar arquivo (índice {idx}): {str(e)[:100]}")
                    num_errors += 1
        
        # Processar vídeos NonFight (non_violent)
        nonfight_dir = split_path / "NonFight"
        if nonfight_dir.exists():
            print(f"Processando vídeos NonFight de {split}...")
            
            # Tentar usar glob primeiro, se falhar usar os.listdir
            try:
                video_files = list(nonfight_dir.glob("*.avi"))
            except Exception:
                # Fallback: usar os.listdir que lida melhor com encoding no Windows
                try:
                    nonfight_dir_str = str(nonfight_dir.resolve())
                    video_files = [
                        nonfight_dir / fname 
                        for fname in os.listdir(nonfight_dir_str) 
                        if fname.lower().endswith('.avi')
                    ]
                except Exception as e:
                    print(f"  Erro ao listar arquivos: {str(e)[:100]}")
                    video_files = []
            
            print(f"  Encontrados {len(video_files)} vídeos")
            
            for idx, video_file in enumerate(video_files, 1):
                try:
                    # Tentar obter nome do arquivo de forma segura
                    try:
                        video_name = video_file.name
                    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
                        # Se falhar, usar nome baseado no índice
                        video_name = f"video_{num_non_violent + idx:06d}.avi"
                    
                    # Garantir que temos um Path válido
                    if isinstance(video_file, str):
                        video_file = Path(video_file)
                    
                    dest_path = non_violent_dir / video_name
                    
                    if safe_copy_file(video_file, dest_path, num_non_violent + idx):
                        num_non_violent += 1
                    else:
                        num_errors += 1
                        
                    if idx % 100 == 0:
                        print(f"  Processados {idx}/{len(video_files)} vídeos...")
                except Exception as e:
                    print(f"  Erro ao processar arquivo (índice {idx}): {str(e)[:100]}")
                    num_errors += 1
    
    print(f"\nOrganização concluída:")
    print(f"  - Vídeos violentos: {num_violent}")
    print(f"  - Vídeos não violentos: {num_non_violent}")
    print(f"  - Total: {num_violent + num_non_violent}")
    if num_errors > 0:
        print(f"  - Erros encontrados: {num_errors}")
    
    return num_violent, num_non_violent


if __name__ == "__main__":
    # Executar organização
    organize_rwf2000_dataset()

