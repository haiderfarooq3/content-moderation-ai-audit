"""ModerationPipeline — 3-layer production guardrail around the mitigated classifier.

Layer 1: Regex input filter (BLOCKLIST with 5 categories, >=20 patterns).
Layer 2: Calibrated DistilBERT model (isotonic) — block >= 0.6, allow <= 0.4.
Layer 3: Human review queue for the 0.4 - 0.6 uncertainty band.

.predict(text) -> dict with keys {"decision", "layer", "confidence", optional "category"}.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Layer 1 — Regex blocklist. Categories, pattern counts, and intent follow the
# assignment spec. All patterns use word boundaries and re.IGNORECASE.
# ---------------------------------------------------------------------------
BLOCKLIST: dict[str, list[re.Pattern]] = {
    "direct_threat": [
        # subject + variable threat verb + object, with common conjugations of "going to".
        re.compile(r"\bi(?:'m| am| will| will not|'ll| gonna| am going to| will be)?\s+(?:going\s+to\s+|gonna\s+)?(kill|murder|shoot|stab|hurt|beat|strangle|choke)\s+(you|him|her|them|u)\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are| will be| are gonna be)\s+(?:going\s+to\s+)?(?:die|dead|get\s+killed|be\s+killed|meet\s+your\s+maker)\b", re.IGNORECASE),
        re.compile(r"\bi(?:'ll| will| am going to| gonna)\s+find\s+where\s+(?:you|u)\s+live\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(kill|shoot|stab|murder|hurt|silence)\s+(you|him|her|them)\b", re.IGNORECASE),
        re.compile(r"\bi\s+hope\s+(?:you|u|he|she|they)\s+(?:die|get\s+killed|are\s+murdered|burn(?:\s+in\s+hell)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:i|we)(?:'ll| will| should)\s+(?:end|terminate|take\s+out)\s+(you|him|her|them)\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        # second-person subject + self-harm verb; must not match "I thought about hurting myself".
        re.compile(r"\byou\s+(?:should|ought\s+to|need\s+to|have\s+to|must)\s+(?:go\s+)?(?:kill|hang|harm|hurt)\s+your?self\b", re.IGNORECASE),
        re.compile(r"\bgo\s+(?:kill|hang|off)\s+your?self\b", re.IGNORECASE),
        re.compile(r"\bkys\b|\bkysw\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+would\s+(?:miss|care\s+about)\s+(?:you|u)\s+if\s+(?:you|u)\s+(?:died|killed\s+your?self|were\s+dead)\b", re.IGNORECASE),
        re.compile(r"\b(?:do|would)\s+everyone\s+a\s+(?:favou?r|service)\s+and\s+(?:disappear|die|kill\s+your?self)\b", re.IGNORECASE),
        re.compile(r"\bwhy\s+don(?:'t|t)\s+you\s+just\s+(?:die|disappear|kill\s+your?self)\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi(?:'ll|\s+will|\s+am\s+going\s+to|\s+gonna)\s+(?:post|publish|leak|share|dox)\s+(?:your|ur)\s+(?:address|phone|home|real\s+name|photos?|workplace)\b", re.IGNORECASE),
        re.compile(r"\bi\s+know\s+where\s+(?:you|u)\s+(?:live|work|go\s+to\s+school)\b", re.IGNORECASE),
        re.compile(r"\bi\s+found\s+(?:your|ur)\s+(?:real\s+name|home|address|phone|workplace|family)\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+(?:will|is\s+about\s+to|'ll|'s\s+going\s+to)\s+know\s+who\s+(?:you|u)\s+really\s+are\b", re.IGNORECASE),
        re.compile(r"\b(?:posting|sharing|leaking)\s+(?:your|ur)\s+(?:info|details|address|phone|photos?)\b", re.IGNORECASE),
    ],
    "dehumanization": [
        # non-capturing (?:human|people|person) keeps pattern auditable.
        re.compile(r"\b(?:they|those|these|all)\s+(?:are|r)\s+not\s+(?:human|people|person)s?\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those|these|all)\s+(?:are|r)\s+(?:animals|vermin|rats|cockroaches|parasites|subhuman)\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those|these|all)\s+should\s+be\s+(?:exterminated|eliminated|purged|wiped\s+out|removed)\b", re.IGNORECASE),
        re.compile(r"\b(?:they|those|these|all)\s+(?:are|r)\s+(?:a\s+)?(?:disease|plague|cancer|virus|infection)\b", re.IGNORECASE),
        re.compile(r"\b(?:these|those)\s+people\s+(?:don(?:'t|t)|do\s+not)\s+deserve\s+to\s+(?:live|exist|breathe)\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        # plural/collective subject + aggressive verb; lookahead used in at least one pattern.
        re.compile(r"\beveryone\s+(?=.*?\b(?:report|dox|attack|swarm|raid)\b)", re.IGNORECASE),
        re.compile(r"\blet(?:'s|s)\s+all\s+(?:go\s+after|pile\s+on|attack|report)\s+(?:this|that|them|@?\w+)\b", re.IGNORECASE),
        re.compile(r"\braid\s+(?:their|his|her|this)\s+(?:profile|page|account|dms?)\b", re.IGNORECASE),
        re.compile(r"\bmass\s+(?:report|flag|downvote)\s+(?:this|that|them|@?\w+)\b", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> dict[str, Any] | None:
    """Return a block decision dict if any category matches, else None."""
    for category, patterns in BLOCKLIST.items():
        for pat in patterns:
            if pat.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


# ---------------------------------------------------------------------------
# Layer 2 — Calibrated model.
# ---------------------------------------------------------------------------
@dataclass
class IsotonicCalibrator:
    """Isotonic-regression calibrator persisted as a pickled object."""
    x_: np.ndarray
    y_: np.ndarray

    def predict(self, p: np.ndarray) -> np.ndarray:
        # piecewise-linear interpolation on the sorted (x, y) pairs.
        return np.interp(np.clip(p, 0, 1), self.x_, self.y_)


class ModerationPipeline:
    """Three-layer guardrail. Call .predict(text) for a decision dict."""

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        calibrator_path: str | pathlib.Path,
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = pathlib.Path(model_dir)
        self.calibrator_path = pathlib.Path(calibrator_path)
        self.block_threshold = float(block_threshold)
        self.allow_threshold = float(allow_threshold)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(device)
        self.model.eval()

        with open(self.calibrator_path, "rb") as f:
            self.calibrator: IsotonicCalibrator = pickle.load(f)

    @torch.no_grad()
    def _raw_prob(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        p = F.softmax(logits, dim=-1)[0, 1].item()
        return float(p)

    def predict(self, text: str) -> dict[str, Any]:
        layer1 = input_filter(text)
        if layer1 is not None:
            return layer1

        raw = self._raw_prob(text)
        calibrated = float(self.calibrator.predict(np.array([raw]))[0])
        if calibrated >= self.block_threshold:
            decision = "block"
        elif calibrated <= self.allow_threshold:
            decision = "allow"
        else:
            decision = "review"
        return {
            "decision": decision,
            "layer": "model",
            "confidence": calibrated,
            "raw_confidence": raw,
        }


__all__ = ["BLOCKLIST", "input_filter", "IsotonicCalibrator", "ModerationPipeline"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibrator", required=True)
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()
    pipe = ModerationPipeline(args.model, args.calibrator)
    for t in args.text:
        print(t, "->", json.dumps(pipe.predict(t)))
