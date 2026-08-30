def chunk_text(text, chunk_size=1000, chunk_overlap=0):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid chunk_overlap")
    step = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start+chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks

def unique_contexts(dataset, splits=("train",)):
    seen = {}
    for split in splits:
        for row in dataset[split]:
            seen[row["title"]] = row["context"]
    return [{"title":title,"context":context} for title,context in seen.items()]
