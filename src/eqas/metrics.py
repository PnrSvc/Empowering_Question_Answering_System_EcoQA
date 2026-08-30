import re
import string
from collections import Counter

def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))

def token_f1(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())

    if same == 0:
        return 0.0

    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def aggregate(predictions, references):
    assert len(predictions) == len(references)
    n = len(predictions)
    return {
        "n": n,
        "EM": 100 * sum(exact_match(p, r) for p, r in zip(predictions, references)) / max(n, 1),
        "F1": 100 * sum(token_f1(p, r) for p, r in zip(predictions, references)) / max(n, 1),
    }


def mean_latency_seconds(rows, latency_key="latency_seconds"):
    if not rows:
        return 0.0
    values = [float(row[latency_key]) for row in rows]
    if any(v < 0 for v in values):
        raise ValueError("latency cannot be negative")
    return sum(values) / len(values)

def fallback_rate(rows, fallback_key="used_fallback"):
    if not rows:
        return 0.0
    return 100.0 * sum(bool(row[fallback_key]) for row in rows) / len(rows)
