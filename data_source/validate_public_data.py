from HomeAppliancesDataset import load_dataset_from_github
import json

ds = load_dataset_from_github(validate=True)

titles = {s:set(ds[s]["title"]) for s in ("train","validation","test")}
report = {
    "train": len(ds["train"]),
    "validation": len(ds["validation"]),
    "test": len(ds["test"]),
    "total": sum(len(ds[s]) for s in ("train","validation","test")),
    "unique_products": len(set().union(*titles.values())),
    "product_overlap_train_validation": len(titles["train"] & titles["validation"]),
    "product_overlap_train_test": len(titles["train"] & titles["test"]),
    "product_overlap_validation_test": len(titles["validation"] & titles["test"]),
}
print(json.dumps(report, indent=2))
