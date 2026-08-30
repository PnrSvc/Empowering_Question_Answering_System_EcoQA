import json
import requests
from datasets import Dataset, DatasetDict

_URL = "https://raw.githubusercontent.com/Gokcimen/Home_Appliance_Dataset/main/"

_URLS = {
    "train": _URL + "train.json",
    "test": _URL + "test.json",
    "dev": _URL + "dev.json",
}

EXPECTED = {
    "train": 8000,
    "validation": 1000,
    "test": 1000,
    "total": 10000,
    "products": 1111,
}

def _download_json(url, timeout=60):
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()

def _flatten(raw):
    rows = []
    for article in raw["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                rows.append({
                    "id": str(qa["id"]),
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in answers],
                        "answer_start": [int(a["answer_start"]) for a in answers],
                        "answer_end": [
                            int(a["answer_end"]) if "answer_end" in a else int(a["answer_start"]) + len(a["text"]) - 1
                            for a in answers
                        ],
                    },
                })
    return rows

def load_dataset_from_github(validate=True):
    train = _flatten(_download_json(_URLS["train"]))
    dev = _flatten(_download_json(_URLS["dev"]))
    test = _flatten(_download_json(_URLS["test"]))

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(dev),
        "test": Dataset.from_list(test),
    })

    if validate:
        validate_dataset(ds)

    return ds

def validate_dataset(ds):
    counts = {
        "train": len(ds["train"]),
        "validation": len(ds["validation"]),
        "test": len(ds["test"]),
    }
    assert counts == {"train":8000, "validation":1000, "test":1000}, counts
    assert sum(counts.values()) == 10000

    split_ids = {s:set(map(str, ds[s]["id"])) for s in counts}
    assert not split_ids["train"] & split_ids["validation"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["validation"] & split_ids["test"]

    all_ids = set().union(*split_ids.values())
    assert len(all_ids) == 10000
    assert {int(x) for x in all_ids} == set(range(1,10001))

    split_titles = {s:set(ds[s]["title"]) for s in counts}
    assert not split_titles["train"] & split_titles["validation"]
    assert not split_titles["train"] & split_titles["test"]
    assert not split_titles["validation"] & split_titles["test"]

    all_titles = set().union(*split_titles.values())
    assert len(all_titles) == 1111, len(all_titles)

    return {
        "train":8000,
        "validation":1000,
        "test":1000,
        "total":10000,
        "unique_ids":10000,
        "id_min":1,
        "id_max":10000,
        "products":1111,
        "qa_overlap":0,
        "product_overlap":0,
    }
