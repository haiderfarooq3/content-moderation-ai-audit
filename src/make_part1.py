"""Generate part1.ipynb - Baseline DistilBERT classifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_notebook import write_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "part1.ipynb"

md1 = r"""# Part 1 — Baseline DistilBERT Toxicity Classifier

**Scenario.** We are auditing a production content-moderation model. Part 1 reproduces the baseline: a DistilBERT classifier fine-tuned on a 100 000-row stratified sample of the Jigsaw Unintended Bias in Toxicity Classification dataset.

**Outputs of this notebook**
- Trained `distilbert-base-uncased` toxicity classifier saved to `./models/baseline`.
- Accuracy, macro-F1, AUC-ROC, confusion matrix on a held-out 20 000-row eval sample.
- ROC and Precision–Recall curves.
- Operating-threshold sweep (0.3 → 0.7) with F1 per threshold and a justified choice."""

code_setup = r"""import os, json, random, time, pathlib
import numpy as np
import pandas as pd
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

ROOT = pathlib.Path.cwd()
DATA = ROOT / "data" / "jigsaw-unintended-bias-train.csv"
MODEL_DIR = ROOT / "models" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)
print("device:", DEVICE)
print("data :", DATA, "exists:", DATA.exists())"""

md_stratify = r"""## 1. Load and stratify

We load the full CSV, binarize the `toxic` column at 0.5, then draw a stratified random sample: 100 000 training rows and 20 000 evaluation rows. Stratification preserves the native class balance (roughly 8 % toxic) in both splits, so the model and its metrics see a realistic prior."""

code_stratify = r"""from sklearn.model_selection import train_test_split

# Kaggle columns: `target` is the toxicity score in [0,1]; identity columns use native names.
RAW_COLS = ["comment_text", "target", "black", "white", "muslim", "jewish",
            "homosexual_gay_or_lesbian"]
df = pd.read_csv(DATA, usecols=RAW_COLS)
df = df.dropna(subset=["comment_text"]).reset_index(drop=True)
# Rename to the names used in the assignment specification.
df = df.rename(columns={"target": "toxic", "homosexual_gay_or_lesbian": "lgbtq"})
df["label"] = (df["toxic"] >= 0.5).astype(int)
print("full rows:", len(df), "| toxic prevalence:", round(df["label"].mean(), 4))

strat_pool, eval_df = train_test_split(
    df, test_size=20_000, stratify=df["label"], random_state=SEED
)
train_df, _ = train_test_split(
    strat_pool, train_size=100_000, stratify=strat_pool["label"], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

print("train:", len(train_df), "toxic%:", round(train_df["label"].mean(), 4))
print("eval :", len(eval_df),  "toxic%:", round(eval_df["label"].mean(), 4))

# Cohort previews for the bias audit in Part 2 — confirming non-zero cohort sizes early.
hb = eval_df[(eval_df["black"].fillna(0) >= 0.5)]
ref = eval_df[(eval_df["black"].fillna(0) < 0.1) & (eval_df["white"].fillna(0) >= 0.5)]
print("high-black eval rows:", len(hb), "| reference eval rows:", len(ref))

train_df.to_parquet(ROOT / "data" / "train_100k.parquet")
eval_df.to_parquet(ROOT / "data" / "eval_20k.parquet")"""

md_tokenize = r"""## 2. Tokenization

We tokenize `comment_text` with the `distilbert-base-uncased` tokenizer, max length 128, with truncation. 128 tokens is the community-standard ceiling for this dataset — the median Jigsaw comment is ~40 sub-word tokens, and beyond 128 the tail is dominated by pasted spam, so truncation is near-lossless while letting a batch fit comfortably on a consumer-class GPU."""

code_tokenize = r"""from transformers import AutoTokenizer
MODEL_NAME = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts, max_length=128):
    return tok(list(texts), truncation=True, padding="max_length",
               max_length=max_length, return_tensors=None)

t0 = time.time()
train_enc = encode(train_df["comment_text"].tolist())
eval_enc  = encode(eval_df["comment_text"].tolist())
print(f"tokenized in {time.time()-t0:.1f}s")"""

code_dataset = r"""from torch.utils.data import Dataset

class JigsawDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.enc["input_ids"][i]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
            "labels": torch.tensor(int(self.labels[i])),
        }

train_ds = JigsawDataset(train_enc, train_df["label"].values)
eval_ds  = JigsawDataset(eval_enc,  eval_df["label"].values)
len(train_ds), len(eval_ds)"""

md_train = r"""## 3. Fine-tune DistilBERT (3 epochs, Trainer API)

We fine-tune `distilbert-base-uncased` with HuggingFace `Trainer`. Standard binary cross-entropy loss is applied through the model head; no custom loss is introduced. Three epochs is enough for the model to converge on this sample size."""

code_train = r"""from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir=str(MODEL_DIR / "trainer_out"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=200,
    save_strategy="no",
    eval_strategy="epoch",
    report_to="none",
    seed=SEED,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=eval_ds)

t0 = time.time()
trainer.train()
print(f"training wall time: {(time.time()-t0)/60:.1f} min")

trainer.save_model(str(MODEL_DIR))
tok.save_pretrained(str(MODEL_DIR))
print("saved model to", MODEL_DIR)"""

md_eval = r"""## 4. Evaluation on the held-out 20 000 rows

We score the eval set, convert logits to softmax probabilities, and compute:
- Accuracy, macro-F1, AUC-ROC
- Confusion matrix at the default 0.5 threshold
- F1 sweep across thresholds 0.3 / 0.4 / 0.5 / 0.6 / 0.7
- ROC and Precision–Recall curves"""

code_predict = r"""import torch.nn.functional as F

@torch.no_grad()
def predict_probs(model, dataset, batch=128):
    model.eval()
    probs = np.zeros(len(dataset), dtype=np.float32)
    idx = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    for batch_ in loader:
        b = {k: v.to(model.device) for k, v in batch_.items() if k != "labels"}
        logits = model(**b).logits
        p_tox = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs[idx:idx+len(p_tox)] = p_tox
        idx += len(p_tox)
    return probs

t0 = time.time()
eval_probs = predict_probs(trainer.model, eval_ds)
print(f"inference: {time.time()-t0:.1f}s")
np.save(ROOT / "outputs" / "eval_probs_baseline.npy", eval_probs)"""

code_metrics = r"""from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              average_precision_score)

y_true = eval_df["label"].values
y_pred_05 = (eval_probs >= 0.5).astype(int)

acc  = accuracy_score(y_true, y_pred_05)
f1m  = f1_score(y_true, y_pred_05, average="macro")
auc  = roc_auc_score(y_true, eval_probs)
cm   = confusion_matrix(y_true, y_pred_05)

print(f"Accuracy   : {acc:.4f}")
print(f"Macro F1   : {f1m:.4f}")
print(f"AUC-ROC    : {auc:.4f}")
print("Confusion matrix (threshold=0.5):")
print(pd.DataFrame(cm, index=["true_non","true_tox"], columns=["pred_non","pred_tox"]))"""

code_curves = r"""import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_true, eval_probs)
prec, rec, _ = precision_recall_curve(y_true, eval_probs)
ap = average_precision_score(y_true, eval_probs)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[0].plot([0,1],[0,1],'--', color="grey")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC — baseline DistilBERT"); axes[0].legend()

axes[1].plot(rec, prec, label=f"AP = {ap:.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall — baseline DistilBERT"); axes[1].legend()

plt.tight_layout()
plt.savefig(ROOT / "outputs" / "part1_roc_pr.png", dpi=120)
plt.show()"""

md_threshold = r"""## 5. Threshold justification

The default 0.5 threshold is rarely optimal. Below we report F1 at five candidate thresholds and choose the one used for every later notebook."""

code_threshold = r"""rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (eval_probs >= t).astype(int)
    tp = int(((yp==1) & (y_true==1)).sum())
    fp = int(((yp==1) & (y_true==0)).sum())
    fn = int(((yp==0) & (y_true==1)).sum())
    rows.append({
        "threshold": t,
        "F1_macro": f1_score(y_true, yp, average="macro"),
        "F1_toxic": f1_score(y_true, yp, pos_label=1),
        "precision_toxic": tp/max(tp+fp, 1),
        "recall_toxic":    tp/max(tp+fn, 1),
    })
th_table = pd.DataFrame(rows).round(4)
print(th_table.to_string(index=False))
th_table.to_csv(ROOT / "outputs" / "part1_threshold_sweep.csv", index=False)"""

code_choose = r"""chosen = float(th_table.loc[th_table["F1_macro"].idxmax(), "threshold"])
print("Chosen operating threshold =", chosen)
with open(ROOT / "outputs" / "chosen_threshold.json", "w") as f:
    json.dump({"threshold": chosen}, f)"""

md_justify = r"""### Threshold justification — what does this choice imply for the platform?

We choose the threshold that **maximises macro-F1** on the held-out set. A lower threshold (0.3) catches more toxic content but inflates false positives: innocent users get flagged and the moderation backlog balloons. A higher threshold (0.7) minimises false positives but misses more genuine toxicity, which is unacceptable for the platform's civil-rights obligations.

Macro-F1 balances precision and recall equally across both classes — neither over-moderation nor under-moderation is acceptable on a social platform. The chosen threshold is carried forward to every subsequent notebook via `outputs/chosen_threshold.json`.

There is no universally correct answer; this choice reflects a platform that treats false positives (silencing legitimate speech) and false negatives (missed toxicity) as equally costly."""

md_summary = r"""## 6. Summary

- Model: `distilbert-base-uncased` fine-tuned for 3 epochs on 100 000 stratified rows.
- Accuracy / macro-F1 / AUC-ROC reported above.
- Operating threshold chosen by maximising macro-F1 on held-out data.
- Model artefacts saved to `./models/baseline`; eval probabilities saved to `./outputs/eval_probs_baseline.npy` for Parts 2–5."""

cells = [
    ("markdown", md1),
    ("code", code_setup),
    ("markdown", md_stratify),
    ("code", code_stratify),
    ("markdown", md_tokenize),
    ("code", code_tokenize),
    ("code", code_dataset),
    ("markdown", md_train),
    ("code", code_train),
    ("markdown", md_eval),
    ("code", code_predict),
    ("code", code_metrics),
    ("code", code_curves),
    ("markdown", md_threshold),
    ("code", code_threshold),
    ("code", code_choose),
    ("markdown", md_justify),
    ("markdown", md_summary),
]

write_notebook(OUT, cells)
