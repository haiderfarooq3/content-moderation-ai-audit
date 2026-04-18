# Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety

FAST-NUCES **Responsible & Explainable AI** — Assignment 2.

End-to-end audit of a production-style DistilBERT toxicity classifier trained on the Jigsaw Unintended Bias in Toxicity Classification dataset. The repo contains five notebooks + one pipeline module that together cover:

| Notebook / file | Part | What it does |
|---|---|---|
| `part1.ipynb` | 1 | Fine-tune `distilbert-base-uncased` on 100 k stratified rows; evaluate on 20 k; justify operating threshold. |
| `part2.ipynb` | 2 | Bias audit: high-black vs. reference cohort; TPR/FPR/FNR/precision, Disparate-Impact ratio, aif360 statistical-parity and equal-opportunity differences. |
| `part3.ipynb` | 3 | Two from-scratch adversarial attacks (character-level evasion + label-flipping poisoning). |
| `part4.ipynb` | 4 | Three mitigation techniques (aif360 Reweighing, fairlearn ThresholdOptimizer, oversampling); Pareto plot; base-rate theorem discussion. |
| `part5.ipynb` | 5 | Demonstrates the `ModerationPipeline` (regex → calibrated model → review queue) on 1 000 comments with full layer-distribution and threshold-sensitivity analysis. |
| `pipeline.py` | 5 | Stand-alone `ModerationPipeline` class with `.predict(text) -> dict` — the three-layer production guardrail. |

## Environment

- **Python 3.12.13**
- **Device used for training:** Apple M-series GPU via PyTorch MPS backend (`torch.backends.mps`).
- Google Colab T4 GPU also works with the identical `requirements.txt`.

## Reproducing from a clean checkout

```bash
# 1. Create venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=jigsaw-py312 --display-name="Jigsaw (Py3.12)"

# 2. Get the dataset — requires a Kaggle account and competition-rules acceptance:
#    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules
export KAGGLE_API_TOKEN=<your KGAT_... token>
mkdir -p data
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification \
    -f train.csv -p data
unzip -o data/train.csv.zip -d data
mv data/train.csv data/jigsaw-unintended-bias-train.csv

# 3. Execute notebooks in order (each takes roughly these wall times on MPS):
#    part1  ~40 min  (training dominates)
#    part2  ~1 min   (no retrain)
#    part3  ~45 min  (one retrain for poisoning)
#    part4  ~80 min  (two retrains: reweighing + oversampling)
#    part5  ~2 min   (calibrator + 1000 demo predictions)
for n in part1 part2 part3 part4 part5; do
    jupyter nbconvert --to notebook --execute "$n.ipynb" \
        --output "$n.ipynb" --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=jigsaw-py312
done
```

## File / folder layout

```
.
├── assignment_2.pdf              # the assignment spec
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── pipeline.py                   # production guardrail module
├── part1.ipynb … part5.ipynb     # executed notebooks with outputs
├── data/                         # dataset + persisted train/eval parquet
├── models/                       # trained checkpoints (git-ignored)
│   ├── baseline/
│   ├── poisoned/
│   ├── reweighed/
│   ├── oversampled/
│   └── calibrator.pkl
├── outputs/                      # summary CSVs / PNG plots
└── src/                          # helper scripts that regenerate the notebooks
```

## Key results

These are filled in at the end of each notebook run — see `outputs/*.csv` and the markdown summary at the bottom of each notebook for the numeric answers to every *Key question* box in the PDF.

## Submission checklist (mapping to the PDF)

- [x] `part1.ipynb` + saved model checkpoint
- [x] `part2.ipynb` with cohort sizes, summary table, grouped bar chart, two confusion matrices
- [x] `part3.ipynb` with both attacks implemented from scratch, ASR table, before/after comparison
- [x] `part4.ipynb` with all three mitigations, comparison table, Pareto plot, best-model save
- [x] `part5.ipynb` demonstrating the pipeline on 1 000 comments, layer-distribution plot, threshold sensitivity
- [x] `pipeline.py` implementing `ModerationPipeline.predict`
- [x] `requirements.txt` with pinned versions
- [x] `README.md` (this file)
- [x] `.gitignore` excluding *.csv, *.pt, *.bin, saved_model/
