from dataclasses import dataclass

@dataclass(frozen=True)
class HallucinationAnnotation:
    response_id: str
    total_atomic_claims: int
    unsupported_atomic_claims: int

    def __post_init__(self):
        if self.total_atomic_claims < 0:
            raise ValueError("total_atomic_claims must be >= 0")
        if self.unsupported_atomic_claims < 0:
            raise ValueError("unsupported_atomic_claims must be >= 0")
        if self.unsupported_atomic_claims > self.total_atomic_claims:
            raise ValueError("unsupported_atomic_claims cannot exceed total_atomic_claims")

def response_level_rate(rows):
    if not rows:
        return 0.0
    hallucinated = sum(r.unsupported_atomic_claims > 0 for r in rows)
    return 100.0 * hallucinated / len(rows)

def claim_level_rate(rows):
    total_claims = sum(r.total_atomic_claims for r in rows)
    unsupported = sum(r.unsupported_atomic_claims for r in rows)
    if total_claims == 0:
        return 0.0
    return 100.0 * unsupported / total_claims
