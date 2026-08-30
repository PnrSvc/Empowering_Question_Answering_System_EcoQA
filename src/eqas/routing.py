from enum import Enum

class Route(str, Enum):
    DIRECT_QA = "direct_qa"
    KG = "kg"
    RETRIEVAL = "retrieval"
    GENERATIVE = "generative"

def normalize_confidence(value):
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    return value

def select_route(
    confidence,
    threshold=0.90,
    kb=None,
    kg_evidence_available=False,
    retrieval_available=False,
):
    confidence = normalize_confidence(confidence)
    threshold = normalize_confidence(threshold)

    if confidence >= threshold and (kb is True or kb is None):
        return Route.DIRECT_QA
    if kg_evidence_available:
        return Route.KG
    if retrieval_available:
        return Route.RETRIEVAL
    return Route.GENERATIVE
