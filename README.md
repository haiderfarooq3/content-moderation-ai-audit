# Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety

FAST-NUCES **Responsible & Explainable AI** — Assignment 2.

End-to-end audit of a production-style DistilBERT toxicity classifier trained on the Jigsaw Unintended Bias in Toxicity Classification dataset. The repo contains five notebooks + one pipeline module.

## Contents

| File | Part | What it does |
|---|---|---|
| `part1.ipynb` | 1 | Fine-tune `distilbert-base-uncased` on 100 k stratified rows; evaluate on 20 k; justify operating threshold. |
| `part2.ipynb` | 2 | Bias audit: high-black vs. reference cohort; TPR/FPR/FNR/precision, Disparate-Impact ratio, aif360 statistical-parity and equal-opportunity differences. |
| `part3.ipynb` | 3 | Two from-scratch adversarial attacks (character-level evasion + label-flipping poisoning). |
| `part4.ipynb` | 4 | Three mitigation techniques (aif360 Reweighing, fairlearn ThresholdOptimizer, oversampling); Pareto plot; base-rate theorem discussion. |
| `part5.ipynb` | 5 | Demonstrates the `ModerationPipeline` (regex → calibrated model → review queue) on 1 000 comments with full layer-distribution and threshold-sensitivity analysis. |
| `pipeline.py` | 5 | Stand-alone `ModerationPipeline` class with `.predict(text) -> dict` — the three-layer production guardrail. |

## Headline results

| Stage | Metric | Value |
|---|---|---|
| Baseline | macro-F1 / AUC-ROC (threshold 0.5) | **0.819 / good** |
| Bias audit (threshold 0.5) | FPR high-black / reference | **0.154 / 0.105** |
| Bias audit | Disparate-Impact ratio (FPR) | **1.47×** (high-black over-flagged) |
| Evasion attack | ASR on 500 confidently-toxic comments | **0.98** |
| Poisoning attack | ΔFNR (toxic) after 5 % label flip | **+0.7 pts** |
| Mitigation (best = fairlearn ThresholdOptimizer) | FPR gap 0.049 → **0.009** | EOD +0.037 → +0.078 |
| Guardrail pipeline (1 000 eval comments) | auto-action F1 / review rate | **0.62 / 2 %** |

## Environment

- **Python 3.12.13** (project venv at `.venv/`).
- **Device used for training:** Apple M-series GPU via PyTorch MPS backend.
  Training wall times actually observed:
  - Baseline (Part 1): **~54 min**
  - Poisoned retrain (Part 3): **~58 min**
  - Reweighed retrain (Part 4, WeightedTrainer custom compute_loss): **~95 min**
  - Oversampled retrain (Part 4, 102 k rows): **~57 min**
- Google Colab T4 GPU also works with the identical `requirements.txt`.
- Compatibility note: `pandas<3.0` is pinned (fairlearn 0.13 + pandas 3.x clash when `ThresholdOptimizer` mixes probabilities into a float32 column).

## Reproducing from a clean checkout

```bash
# 1. venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=jigsaw-py312 --display-name="Jigsaw (Py3.12)"

# 2. Dataset (needs a Kaggle account + rules accepted):
#    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules
export KAGGLE_API_TOKEN=<your KGAT_... token>
mkdir -p data
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f train.csv -p data
unzip -o data/train.csv.zip -d data
mv data/train.csv data/jigsaw-unintended-bias-train.csv

# 3. Execute notebooks in order (wall times on M-series MPS):
#    part1  ~55 min (training dominates)
#    part2   ~1 min (no retrain)
#    part3  ~60 min (one retrain for poisoning)
#    part4  ~95 min + 60 min = ~2.5 h (two retrains)
#    part5   ~2 min (calibrator + 1 000 demo predictions)
for n in part1 part2 part3 part4 part5; do
    jupyter nbconvert --to notebook --execute "$n.ipynb" \
        --output "$n.ipynb" --ExecutePreprocessor.timeout=14400 \
        --ExecutePreprocessor.kernel_name=jigsaw-py312
done
```

## File / folder layout

```
.
├── assignment_2.pdf              # assignment spec
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── pipeline.py                   # production guardrail module
├── part1.ipynb … part5.ipynb     # executed notebooks with outputs
├── data/                         # dataset + persisted train/eval parquet  (git-ignored)
├── models/                       # trained checkpoints                      (git-ignored)
│   ├── baseline/
│   ├── poisoned/
│   ├── reweighed/
│   ├── oversampled/
│   └── calibrator.pkl
├── outputs/                      # summary CSVs / PNG plots / probability caches
└── src/                          # notebook-builder scripts + verify script
```

## Submission checklist (mapping to the PDF)

- [x] `part1.ipynb` + saved model checkpoint, threshold sweep, ROC+PR plots
- [x] `part2.ipynb` with cohort sizes printed (162 hb, 178 ref), summary table, grouped bar chart, two confusion matrices, aif360 SPD/EOD
- [x] `part3.ipynb` with both attacks implemented from scratch, ASR table (0.98), before/after comparison table
- [x] `part4.ipynb` with all three mitigations, comparison table (4 rows), Pareto plot, best-model saved to `outputs/best_mitigated.json` (= fairlearn ThresholdOptimizer)
- [x] `part5.ipynb` demonstrating the pipeline on 1 000 comments with layer-distribution plot, auto-action F1/precision/recall, review-queue toxic/non-toxic breakdown, threshold-sensitivity table
- [x] `pipeline.py` implementing `ModerationPipeline.predict` with regex prefilter (26 patterns across 5 categories)
- [x] `requirements.txt` with pinned versions
- [x] `README.md` (this file)
- [x] `.gitignore` excluding *.csv, *.pt, *.bin, saved_model/, *.safetensors
- [x] Commit history: 6 incremental commits, one per phase
