"""Evaluate original-token recovery on a small demonstration set."""
import json
from pathlib import Path

from predictor import MaskPredictor

ROOT = Path(__file__).parent


def main():
    dataset = json.loads((ROOT / "examples" / "evaluation_set.json").read_text())
    model = MaskPredictor()
    model.load()
    records = []
    for sample in dataset:
        target = model._pipeline.tokenizer(sample["expected"], add_special_tokens=False)["input_ids"]
        if len(target) != 1:
            raise ValueError("Evaluation targets must be single vocabulary tokens.")
        result = model.predict(sample["text"], 5)
        candidates = [p["token"] for p in result["predictions"]]
        records.append({**sample, "top1_correct": candidates[0] == target[0],
                        "top5_correct": target[0] in candidates, "result": result})
    report = {"scope": "12 hand-authored demonstration sentences; exact target-token recovery, not a benchmark",
              "count": len(records), "top1_accuracy": sum(r["top1_correct"] for r in records) / len(records),
              "top5_accuracy": sum(r["top5_correct"] for r in records) / len(records), "records": records}
    (ROOT / "examples" / "sample_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
