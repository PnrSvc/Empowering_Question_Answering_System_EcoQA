SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for home-appliance information. "
    "Use the supplied context and evidence for factual claims. "
    "Do not introduce unsupported product facts."
)

def qa_prompt(context, question):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

def grounded_prompt(context, question, structured_evidence="", retrieved_evidence=""):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Source context:\n{context}\n\n"
        f"Structured evidence:\n{structured_evidence}\n\n"
        f"Retrieved evidence:\n{retrieved_evidence}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
