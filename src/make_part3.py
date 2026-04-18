"""Generate part3.ipynb - Adversarial attacks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_notebook import write_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "part3.ipynb"

md1 = r"""# Part 3 — Adversarial Attacks: Breaking the Classifier

A content-moderation model that is "accurate on average" is not a production system. Real platforms face **adversarial inputs** (users who actively probe the model for weaknesses) and **adversarial training pipelines** (supply-chain attacks on the data we train on).

This notebook implements both threat models from scratch:

1. **Attack 1 — Character-level evasion.** A `perturb(text)` function that applies zero-width spaces, Unicode homoglyph substitutions, and random character duplication to a toxic comment, without using any adversarial-ML library. We measure **Attack Success Rate (ASR)** on 500 confidently-toxic comments.
2. **Attack 2 — Label-flipping poisoning.** We flip 5 % of the training-set labels and retrain a fresh DistilBERT from the pre-trained checkpoint, then report how F1, accuracy, and false-negative rate change on the *clean* eval set.

The final markdown cell compares the two attacks by operational risk."""

code_setup = r"""import json, pathlib, random, time, unicodedata
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT = pathlib.Path.cwd()
MODEL_DIR = ROOT / "models" / "baseline"

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
clean_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
clean_model = clean_model.to(device); clean_model.eval()
print("device:", device)

eval_df   = pd.read_parquet(ROOT / "data" / "eval_20k.parquet")
train_df  = pd.read_parquet(ROOT / "data" / "train_100k.parquet")
eval_probs = np.load(ROOT / "outputs" / "eval_probs_baseline.npy")
with open(ROOT / "outputs" / "chosen_threshold.json") as f:
    THRESH = json.load(f)["threshold"]
print("eval rows:", len(eval_df), "| threshold:", THRESH)"""

md_attack1 = r"""## Attack 1 — Character-level evasion

### `perturb(text)` — three transformations

1. **Zero-width-space insertion** — insert `U+200B` between every 2–3 characters of words that the classifier likely keys on. This is invisible to humans but splits BPE/WordPiece sub-word tokens that the model relies on.
2. **Unicode homoglyph substitution** — replace Latin `a` / `e` / `o` with Cyrillic `а` (U+0430), `е` (U+0435), `о` (U+043E). A human reads the same word; the tokenizer sees unknown tokens.
3. **Random character duplication** — for 20 % of characters in each word, insert a duplicate (`hate` → `haate`). This breaks the n-gram / sub-word patterns the model learned.

All three transformations are applied in sequence. The function is intentionally cheap — attackers in the wild will not use a PGD-style optimiser, they will use exactly this kind of hand-crafted text rewriting."""

code_perturb = r"""ZWSP = "\u200B"
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e",
        "A": "\u0410", "E": "\u0415", "O": "\u041e"}
DUP_PROB = 0.20

def _dup_chars(word, prob=DUP_PROB, rng=None):
    rng = rng or random
    out = []
    for ch in word:
        out.append(ch)
        if rng.random() < prob and ch.isalpha():
            out.append(ch)
    return "".join(out)

def _zwsp_split(word, rng=None):
    rng = rng or random
    if len(word) < 4:
        return word
    i, out = 0, []
    while i < len(word):
        step = rng.choice([2, 3])
        out.append(word[i:i+step])
        i += step
    return ZWSP.join(out)

def _homoglyph(word):
    return "".join(HOMO.get(ch, ch) for ch in word)

def perturb(text: str, rng=None) -> str:
    rng = rng or random
    words = text.split()
    out = []
    for w in words:
        w2 = _dup_chars(w, rng=rng)
        w2 = _homoglyph(w2)
        w2 = _zwsp_split(w2, rng=rng)
        out.append(w2)
    return " ".join(out)

# sanity check
ex = "I hate this stupid comment"
print("orig:", repr(ex))
print("pert:", repr(perturb(ex, rng=random.Random(0))))"""

code_sample = r"""from torch.utils.data import Dataset

class TextDS(Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i])}

@torch.no_grad()
def score(model, texts, batch=64):
    model.eval()
    probs = np.zeros(len(texts), dtype=np.float32)
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=128, return_tensors=None)
    loader = torch.utils.data.DataLoader(TextDS(enc), batch_size=batch)
    idx = 0
    for b in loader:
        b = {k: v.to(model.device) for k, v in b.items()}
        p = F.softmax(model(**b).logits, dim=-1)[:, 1].cpu().numpy()
        probs[idx:idx+len(p)] = p
        idx += len(p)
    return probs

# Assemble sample: eval rows the clean model says are toxic with confidence >= 0.7
cand = eval_df.copy()
cand["prob"] = eval_probs
cand = cand[cand["prob"] >= 0.7]
cand = cand[cand["label"] == 1]  # genuinely toxic — we attack real toxicity
print("candidate pool size:", len(cand))

rng = random.Random(SEED)
sample = cand.sample(n=500, random_state=SEED).reset_index(drop=True)
print("attack sample:", len(sample))"""

code_asr = r"""orig_texts = sample["comment_text"].tolist()
pert_texts = [perturb(t, rng=random.Random(i)) for i, t in enumerate(orig_texts)]

orig_probs = score(clean_model, orig_texts)
pert_probs = score(clean_model, pert_texts)

orig_preds = (orig_probs >= THRESH).astype(int)
pert_preds = (pert_probs >= THRESH).astype(int)

# ASR = of the originally-detected toxic comments, what fraction now predicted non-toxic
detected = orig_preds == 1
evaded   = (pert_preds == 0) & detected
asr = evaded.sum() / max(detected.sum(), 1)
print(f"Attack 1 — Character-level evasion")
print(f"  originally detected as toxic : {int(detected.sum())} / {len(sample)}")
print(f"  evaded (flip to non-toxic)  : {int(evaded.sum())}")
print(f"  Attack Success Rate (ASR)   : {asr:.3f}")
print(f"  mean confidence  BEFORE     : {orig_probs[detected].mean():.4f}")
print(f"  mean confidence  AFTER      : {pert_probs[detected].mean():.4f}")

asr_table = pd.DataFrame({
    "metric": ["sample_size", "originally_detected", "evaded", "ASR",
               "mean_conf_before", "mean_conf_after"],
    "value":  [len(sample), int(detected.sum()), int(evaded.sum()), float(asr),
               float(orig_probs[detected].mean()), float(pert_probs[detected].mean())],
})
asr_table.to_csv(ROOT / "outputs" / "part3_attack1_asr.csv", index=False)
asr_table"""

code_examples = r"""import itertools
rows = []
for i in list(itertools.islice(np.where(detected & (pert_preds==0))[0], 5)):
    rows.append({"original": orig_texts[i][:80],
                 "perturbed": pert_texts[i][:80],
                 "p_before": round(float(orig_probs[i]), 3),
                 "p_after":  round(float(pert_probs[i]), 3)})
pd.DataFrame(rows)"""

md_attack2 = r"""## Attack 2 — Label-flipping poisoning

We corrupt 5 % of the 100 000-row training set by flipping each selected label (toxic → non-toxic and vice versa). We then **retrain a fresh DistilBERT from the pre-trained `distilbert-base-uncased` checkpoint** — *not* from the Part 1 fine-tuned model; the point is to simulate a fresh training pipeline being corrupted upstream.

We reuse identical hyper-parameters from Part 1 and evaluate on the same clean eval set. The attack succeeds if the model's false-negative rate rises — i.e. toxic comments slip through."""

code_flip = r"""POISON_DIR = ROOT / "models" / "poisoned"
POISON_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
poison_idx = rng.choice(len(train_df), size=int(0.05 * len(train_df)), replace=False)
train_poisoned = train_df.copy()
train_poisoned.loc[poison_idx, "label"] = 1 - train_poisoned.loc[poison_idx, "label"]

print("total flipped labels:", len(poison_idx))
print("original toxic %    :", train_df['label'].mean().round(4))
print("poisoned toxic %    :", train_poisoned['label'].mean().round(4))"""

code_retrain = r"""enc_train = tok(train_poisoned["comment_text"].tolist(), truncation=True,
                padding="max_length", max_length=128, return_tensors=None)
enc_eval  = tok(eval_df["comment_text"].tolist(), truncation=True,
                padding="max_length", max_length=128, return_tensors=None)

class JigsawDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                "labels": torch.tensor(int(self.labels[i]))}

ds_train = JigsawDataset(enc_train, train_poisoned["label"].values)
ds_eval  = JigsawDataset(enc_eval,  eval_df["label"].values)

poisoned_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

args = TrainingArguments(
    output_dir=str(POISON_DIR / "trainer_out"),
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
    fp16=False, bf16=False,
    dataloader_pin_memory=False,
)
trainer = Trainer(model=poisoned_model, args=args,
                  train_dataset=ds_train, eval_dataset=ds_eval)

t0 = time.time()
trainer.train()
print(f"poisoned training wall time: {(time.time()-t0)/60:.1f} min")
trainer.save_model(str(POISON_DIR))"""

code_eval_poisoned = r"""@torch.no_grad()
def predict_probs(model, dataset, batch=128):
    model.eval()
    probs = np.zeros(len(dataset), dtype=np.float32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)
    idx = 0
    for b in loader:
        b_ = {k: v.to(model.device) for k, v in b.items() if k != "labels"}
        p = F.softmax(model(**b_).logits, dim=-1)[:, 1].cpu().numpy()
        probs[idx:idx+len(p)] = p; idx += len(p)
    return probs

poison_probs = predict_probs(trainer.model, ds_eval)
np.save(ROOT / "outputs" / "eval_probs_poisoned.npy", poison_probs)

from sklearn.metrics import f1_score, accuracy_score
y  = eval_df["label"].values
baseline_preds = (eval_probs  >= THRESH).astype(int)
poison_preds   = (poison_probs >= THRESH).astype(int)

def fnr(y, yp):
    pos = (y == 1).sum()
    fn  = ((yp == 0) & (y == 1)).sum()
    return float(fn / max(pos, 1))

table = pd.DataFrame({
    "metric":      ["accuracy", "F1_macro", "F1_toxic", "FNR_toxic"],
    "baseline":    [accuracy_score(y, baseline_preds),
                    f1_score(y, baseline_preds, average="macro"),
                    f1_score(y, baseline_preds, pos_label=1),
                    fnr(y, baseline_preds)],
    "poisoned":    [accuracy_score(y, poison_preds),
                    f1_score(y, poison_preds, average="macro"),
                    f1_score(y, poison_preds, pos_label=1),
                    fnr(y, poison_preds)],
}).round(4)
table["delta"] = (table["poisoned"] - table["baseline"]).round(4)
print(table.to_string(index=False))
table.to_csv(ROOT / "outputs" / "part3_attack2_comparison.csv", index=False)"""

md_compare = r"""## Key question — which attack is operationally more dangerous?

**Evasion (Attack 1) is more realistic but self-limiting.**

- Requires **per-comment effort from the attacker**. Each hostile user has to run their own text through a perturber; at scale, that is a friction most spam/hate-mob actors will pay (bot scripts are cheap) but lone trolls often will not.
- Affects **only the perturbed comments**. Clean comments on the platform still get moderated normally.
- Detection is possible: a unicode normalisation filter plus a zero-width-space stripper (exactly the preprocessor one would add to a production pipeline) removes most of this attack cheaply.

**Poisoning (Attack 2) is catastrophic but requires privileged access.**

- Requires **access to the training pipeline**. That is normally a small number of engineers, an insider threat, or a compromised CI runner — three orders of magnitude harder than evading the classifier as a user.
- Affects **every comment ever scored** by the poisoned model, not just one.
- Detection is hard: the corrupted model looks normal on aggregate accuracy; the FNR on toxic content rises silently.

For a social platform, the realistic threat ordering is: (a) **evasion**, running constantly at scale, (b) **poisoning via compromised data contributor**, (c) **direct poisoning by an insider**.

**Defensive priority that follows from this:**

1. **Input normalisation before tokenisation** — unicode NFKC, strip zero-width characters, collapse repeated characters. These defend against evasion for a few dozen lines of code.
2. **Training-data provenance and anomaly detection** — track source-of-label, run data-quality checks that flag class-rate drift, and retrain in a reproducible pipeline so a corrupted run can be identified post-hoc.
3. **Layered defence** — the regex pre-filter in Part 5 catches raw slurs regardless of model drift, so a poisoned model alone cannot open the whole floodgate.

The regex pre-filter + unicode normalisation implemented in Part 5 is the deployment-time response to both attacks."""

cells = [
    ("markdown", md1),
    ("code", code_setup),
    ("markdown", md_attack1),
    ("code", code_perturb),
    ("code", code_sample),
    ("code", code_asr),
    ("code", code_examples),
    ("markdown", md_attack2),
    ("code", code_flip),
    ("code", code_retrain),
    ("code", code_eval_poisoned),
    ("markdown", md_compare),
]
write_notebook(OUT, cells)
