"""
Script para validar a estrutura de dados do projeto.

Este script verifica:
1. Existência de diretórios obrigatórios
2. Estrutura de pastas correta
3. Correspondência entre modalidades (vídeo, pose, emoção)
4. Formato e shape dos arquivos .npy
5. Consistência de IDs de vídeo entre modalidades
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

from src import paths as p


class DataStructureValidator:
    """Validador de estrutura de dados."""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = str(p.PROJECT_ROOT)
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []
        self.info = []
    
    def validate(self) -> Tuple[bool, List[str], List[str], List[str]]:
        """
        Executa todas as validações.
        
        Returns:
            (success, errors, warnings, info)
        """
        print("=" * 70)
        print("VALIDAÇÃO DA ESTRUTURA DE DADOS")
        print("=" * 70)
        
        # Validar estrutura de datasets
        self._validate_datasets()
        
        # Validar estrutura de dados processados
        self._validate_processed_data()
        
        # Validar correspondência entre modalidades
        self._validate_modality_consistency()
        
        # Validar formato de arquivos
        self._validate_file_formats()
        
        # Resumo
        print("\n" + "=" * 70)
        print("RESUMO DA VALIDAÇÃO")
        print("=" * 70)
        
        if self.errors:
            print(f"\n❌ ERROS ENCONTRADOS: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ Nenhum erro encontrado!")
        
        if self.warnings:
            print(f"\n⚠️  AVISOS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.info:
            print(f"\nℹ️  INFORMAÇÕES: {len(self.info)}")
            for info in self.info:
                print(f"  - {info}")
        
        success = len(self.errors) == 0
        return success, self.errors, self.warnings, self.info
    
    def _validate_datasets(self):
        """Valida estrutura dos datasets originais."""
        print("\n[1/4] Validando datasets originais...")
        
        # RWF-2000
        rwf_path = p.RWF2000_ROOT
        if rwf_path.exists():
            for split in ["train", "val"]:
                split_path = rwf_path / split
                if not split_path.exists():
                    self.warnings.append(f"RWF-2000/{split} não encontrado (opcional)")
                    continue
                
                for class_name in ["Fight", "NonFight"]:
                    class_path = split_path / class_name
                    if not class_path.exists():
                        self.errors.append(f"RWF-2000/{split}/{class_name} não encontrado")
                    else:
                        video_count = len(list(class_path.glob("*.avi")))
                        if video_count == 0:
                            self.warnings.append(f"RWF-2000/{split}/{class_name} está vazio")
                        else:
                            self.info.append(f"RWF-2000/{split}/{class_name}: {video_count} vídeos")
        else:
            self.warnings.append("Dataset RWF-2000 não encontrado (obrigatório para treinamento)")
        
        # UCF101 (opcional)
        ucf_path = p.UCF101_ROOT
        if ucf_path.exists():
            self.info.append("Dataset UCF101 encontrado (opcional para pré-treinamento)")
        else:
            self.info.append("Dataset UCF101 não encontrado (opcional)")
        
        # AffectNet (opcional)
        affect_path = p.AFFECTNET_ROOT
        if affect_path.exists():
            self.info.append("Dataset AffectNet encontrado (opcional para treinar EmotionNet)")
        else:
            self.info.append("Dataset AffectNet não encontrado (opcional)")
    
    def _validate_processed_data(self):
        """Valida estrutura de dados processados."""
        print("\n[2/4] Validando dados processados...")
        
        data_root = p.DATA_ROOT
        
        # Raw videos
        raw_path = data_root / "raw"
        if raw_path.exists():
            for class_name in ["violent", "non_violent"]:
                class_path = raw_path / class_name
                if class_path.exists():
                    video_count = len(list(class_path.glob("*.avi")))
                    if video_count > 0:
                        self.info.append(f"data/raw/{class_name}: {video_count} vídeos")
                else:
                    self.warnings.append(f"data/raw/{class_name} não encontrado")
        else:
            self.warnings.append("data/raw não encontrado (execute organize_videos.py primeiro)")
        
        # Processed frames
        processed_path = data_root / "processed"
        if processed_path.exists():
            for class_name in ["violent", "non_violent"]:
                class_path = processed_path / class_name
                if class_path.exists():
                    video_dirs = [d for d in class_path.iterdir() if d.is_dir()]
                    if video_dirs:
                        self.info.append(f"data/processed/{class_name}: {len(video_dirs)} vídeos processados")
                else:
                    self.warnings.append(f"data/processed/{class_name} não encontrado")
        else:
            self.warnings.append("data/processed não encontrado (execute extract_frames.py primeiro)")
        
        # Pose data
        pose_path = data_root / "pose" / "rwf2000"
        if pose_path.exists():
            for split in ["train", "val"]:
                split_path = pose_path / split
                if split_path.exists():
                    for class_name in ["violent", "non_violent"]:
                        class_path = split_path / class_name
                        if class_path.exists():
                            npy_files = list(class_path.glob("*.npy"))
                            if npy_files:
                                self.info.append(f"data/pose/rwf2000/{split}/{class_name}: {len(npy_files)} arquivos")
                else:
                    self.warnings.append(f"data/pose/rwf2000/{split} não encontrado")
        else:
            self.warnings.append("data/pose/rwf2000 não encontrado (execute extract_pose.py primeiro)")
        
        # Emotion data
        emotion_path = data_root / "emotion" / "rwf2000"
        if emotion_path.exists():
            for split in ["train", "val"]:
                split_path = emotion_path / split
                if split_path.exists():
                    for class_name in ["violent", "non_violent"]:
                        class_path = split_path / class_name
                        if class_path.exists():
                            npy_files = list(class_path.glob("*.npy"))
                            if npy_files:
                                self.info.append(f"data/emotion/rwf2000/{split}/{class_name}: {len(npy_files)} arquivos")
                else:
                    self.warnings.append(f"data/emotion/rwf2000/{split} não encontrado")
        else:
            self.warnings.append("data/emotion/rwf2000 não encontrado (execute extract_emotion.py primeiro)")
    
    def _validate_modality_consistency(self):
        """Valida correspondência entre modalidades."""
        print("\n[3/4] Validando correspondência entre modalidades...")
        
        pose_path = p.POSE_ROOT / "rwf2000"
        emotion_path = p.EMOTION_ROOT / "rwf2000"
        processed_path = p.PROCESSED_ROOT
        
        if not pose_path.exists():
            self.warnings.append("Não é possível validar correspondência: pose não encontrado")
            return
        
        # Coletar IDs de vídeo de cada modalidade
        for split in ["train", "val"]:
            split_pose = pose_path / split
            split_emotion = emotion_path / split if emotion_path.exists() else None
            split_processed = processed_path if processed_path.exists() else None
            
            if not split_pose.exists():
                continue
            
            for class_name in ["violent", "non_violent"]:
                pose_class = split_pose / class_name
                emotion_class = split_emotion / class_name if split_emotion else None
                
                if not pose_class.exists():
                    continue
                
                # IDs de pose
                pose_ids = {f.stem for f in pose_class.glob("*.npy")}
                
                # IDs de emoção
                if emotion_class and emotion_class.exists():
                    emotion_ids = {f.stem for f in emotion_class.glob("*.npy")}
                    
                    # Verificar correspondência
                    only_pose = pose_ids - emotion_ids
                    only_emotion = emotion_ids - pose_ids
                    
                    if only_pose:
                        self.warnings.append(
                            f"Pose sem emoção correspondente ({split}/{class_name}): {len(only_pose)} vídeos"
                        )
                    if only_emotion:
                        self.warnings.append(
                            f"Emoção sem pose correspondente ({split}/{class_name}): {len(only_emotion)} vídeos"
                    )
                    
                    common = pose_ids & emotion_ids
                    if common:
                        self.info.append(
                            f"Modalidades alinhadas ({split}/{class_name}): {len(common)} vídeos"
                        )
                
                # IDs de frames processados
                if split_processed:
                    processed_class = split_processed / class_name
                    if processed_class.exists():
                        processed_ids = {d.name for d in processed_class.iterdir() if d.is_dir()}
                        
                        only_pose_proc = pose_ids - processed_ids
                        if only_pose_proc:
                            self.warnings.append(
                                f"Pose sem frames correspondentes ({split}/{class_name}): {len(only_pose_proc)} vídeos"
                            )
    
    def _validate_file_formats(self):
        """Valida formato e shape dos arquivos."""
        print("\n[4/4] Validando formato de arquivos...")
        
        pose_path = p.POSE_ROOT / "rwf2000"
        emotion_path = p.EMOTION_ROOT / "rwf2000"
        
        # Validar arquivos de pose
        if pose_path.exists():
            sample_count = 0
            for split in ["train", "val"]:
                split_path = pose_path / split
                if not split_path.exists():
                    continue
                
                for class_name in ["violent", "non_violent"]:
                    class_path = split_path / class_name
                    if not class_path.exists():
                        continue
                    
                    npy_files = list(class_path.glob("*.npy"))[:5]  # Amostra de 5 arquivos
                    for npy_file in npy_files:
                        try:
                            data = np.load(npy_file)
                            if len(data.shape) != 3 or data.shape[1] != 33 or data.shape[2] != 3:
                                self.errors.append(
                                    f"Formato inválido de pose: {npy_file} "
                                    f"(esperado: (num_frames, 33, 3), encontrado: {data.shape})"
                                )
                            sample_count += 1
                        except Exception as e:
                            self.errors.append(f"Erro ao carregar {npy_file}: {str(e)}")
            
            if sample_count > 0:
                self.info.append(f"Formato de pose validado em {sample_count} arquivos de amostra")
        
        # Validar arquivos de emoção
        if emotion_path.exists():
            sample_count = 0
            for split in ["train", "val"]:
                split_path = emotion_path / split
                if not split_path.exists():
                    continue
                
                for class_name in ["violent", "non_violent"]:
                    class_path = split_path / class_name
                    if not class_path.exists():
                        continue
                    
                    npy_files = list(class_path.glob("*.npy"))[:5]  # Amostra de 5 arquivos
                    for npy_file in npy_files:
                        try:
                            data = np.load(npy_file)
                            if len(data.shape) != 2 or data.shape[1] != 8:
                                self.errors.append(
                                    f"Formato inválido de emoção: {npy_file} "
                                    f"(esperado: (num_frames, 8), encontrado: {data.shape})"
                                )
                            sample_count += 1
                        except Exception as e:
                            self.errors.append(f"Erro ao carregar {npy_file}: {str(e)}")
            
            if sample_count > 0:
                self.info.append(f"Formato de emoção validado em {sample_count} arquivos de amostra")


def main():
    """Função principal."""
    validator = DataStructureValidator()
    success, errors, warnings, info = validator.validate()
    
    if success:
        print("\n✅ Estrutura de dados válida!")
        return 0
    else:
        print("\n❌ Estrutura de dados possui erros. Corrija antes de continuar.")
        return 1


if __name__ == "__main__":
    exit(main())

