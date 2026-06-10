# Tools

Esta pasta contém scripts utilitários e ferramentas auxiliares do projeto.

## Scripts Disponíveis

- **filter_ucf101_classes.py** - Filtra o dataset UCF101 mantendo apenas classes relevantes para detecção de violência
- **validate_data_structure.py** - Valida a estrutura de dados do projeto (datasets, dados processados, etc.)
- **compare_baseline_multimodal.py** - Compara modelos baseline com multimodal
- **train_quick_test.py** - Script para teste rápido de treinamento

## Como Usar

### Filtrar UCF101

```bash
python tools/filter_ucf101_classes.py
```

### Validar Estrutura de Dados

```bash
python tools/validate_data_structure.py
```

### Teste Rápido de Treinamento

```bash
python tools/train_quick_test.py
```

### Comparar Modelos

```bash
python tools/compare_baseline_multimodal.py
```

## Nota

Estes são scripts utilitários. Para treinamento completo, use `train_pipeline.py` na raiz do projeto.

