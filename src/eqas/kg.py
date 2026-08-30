import re
from collections import defaultdict

EXPECTED_COUNTS = {
    "product_entity_nodes": 1111,
    "question_nodes": 10000,
    "answer_nodes": 10000,
    "total_nodes": 21111,
    "product_question_edges": 10000,
    "question_answer_edges": 10000,
    "total_edges": 20000,
}

def build_structural_graph(dataset):
    products = {}
    questions = {}
    answers = {}
    product_question_edges = []
    question_answer_edges = []

    product_id_by_title = {}
    product_no = 0
    question_no = 0

    for split in ("train","validation","test"):
        for row in dataset[split]:
            title = row["title"]
            if title not in product_id_by_title:
                product_no += 1
                pid = f"P{product_no:04d}"
                product_id_by_title[title] = pid
                products[pid] = {"title":title,"split":split}
            pid = product_id_by_title[title]

            question_no += 1
            qid = f"Q{question_no:05d}"
            aid = f"A{question_no:05d}"
            questions[qid] = {
                "dataset_id":str(row["id"]),
                "text":row["question"],
                "product_id":pid,
                "split":split,
            }
            answer = row["answers"]["text"][0] if row["answers"]["text"] else ""
            answers[aid] = {"text":answer,"question_id":qid,"split":split}
            product_question_edges.append((pid,qid))
            question_answer_edges.append((qid,aid))

    stats = {
        "product_entity_nodes":len(products),
        "question_nodes":len(questions),
        "answer_nodes":len(answers),
        "total_nodes":len(products)+len(questions)+len(answers),
        "product_question_edges":len(product_question_edges),
        "question_answer_edges":len(question_answer_edges),
        "total_edges":len(product_question_edges)+len(question_answer_edges),
    }
    assert stats == EXPECTED_COUNTS, (stats, EXPECTED_COUNTS)
    return {
        "products":products,
        "questions":questions,
        "answers":answers,
        "product_question_edges":product_question_edges,
        "question_answer_edges":question_answer_edges,
        "stats":stats,
    }

def context_to_structured_facts(title, context):
    facts = [f"Product: {title}"]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    for sentence in sentences:
        if any(k in sentence.lower() for k in (
            "capacity","energy","efficiency","noise","dimension","weight",
            "feature","technology","warranty","program","litre","liter","kg","db"
        )):
            facts.append(sentence)
    return "\n".join(facts)
