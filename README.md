# Fake News & Misinformation Detection

A study of three ML variants; **Classical ML**, **Deep Learning**, and **Transformers** for fake-news / misinformation detection

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

 Name | Task | ~Size | Source |
------|------|-------|--------|
| ISOT Fake News | Binary (fake/real) | 44,898 articles | Kaggle (clmentbisaillon) |
| WELFake | Binary (fake/real) | 72,134 articles | Kaggle (saurabhshahane) |
| Combined | Binary (ISOT + WELFake, deduped) | — | derived |
| NELA-GT | 3-class (reliable / questionable / conspiracy) | sampled 100k / 500k | NELA-GT |
| LIAR | 6-class (pants-fire → true) | 12,836 claims | PolitiFact |
| PHEMEPlus | Cross-dataset eval target only | — | PHEMEPlus |

**ISOT, WELFake, Combined** — processed automatically on first training run (raw → `data/processed/`). No manual step required.

Note: Download dataset and place under data/raw/<datatsetName> so that the pre-processing step are run before training.

## Processing the datasets

**NELA-GT** — must be preprocessed explicitly before training:

```bash
# Complete
python -m src.nela.dataset --preprocess

# Subsampled variants
python -m src.nela.dataset --preprocess --sample 500000 --output-suffix sampled_500k
```

## Training

### Classical ML (TF-IDF + LR / LinearSVC / RandomForest / XGBoost)

```bash
python -m src.classical.train --dataset isot --model all
python -m src.classical.train --dataset welfake --model lr xgb --tune random
```

### Deep Learning (BiLSTM / TextCNN, GloVe frozen & fine-tuned)

```bash
python -m src.deep_learning.train --dataset isot --model all --freeze both
python -m src.deep_learning.train --dataset welfake --model bilstm --freeze finetuned --epochs 15 --batch-size 64
```

### Transformers (BERT / RoBERTa)

```bash
python -m src.transformers_.train --dataset isot --model all --freeze finetuned
python -m src.transformers_.train --dataset welfake --model roberta --epochs 4 --batch-size 16
```

### LIAR (6-class)

```bash
python -m src.liar.train --tier classical --model all
python -m src.liar.train --tier deep_learning --model bilstm textcnn
python -m src.liar.train --tier transformer --model bert roberta --freeze finetuned
```

### Cross-dataset generalization probe

```bash
python -m src.cross_probe --tier transformer --direction combined_to_welfake
python -m src.cross_probe --tier transformer --direction combined_to_isot
```

### NELA / PHEME evaluation

```bash
python -m src.nela.eval --tier all --train-dataset nela_sampled_500k --pheme pheme
```
