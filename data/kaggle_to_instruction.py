"""
Convert Kaggle 10k QA dataset to instruction JSONL for QLoRA
Input: data/kaggle/legal_qa.csv (10002 rows, question+answer+case_name)
Output: data/processed/train.jsonl, val.jsonl, test.jsonl
"""
import csv
import json
import random
from pathlib import Path

SRC = Path("data/kaggle/legal_qa.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def convert():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r["question"].strip()
            a = r["answer"].strip()
            case = r["case_name"].strip()
            if len(q) < 20 or len(a) < 20:
                continue
            rows.append({
                "instruction": "Answer the Indian legal question based on the Supreme Court case. Be concise and accurate.",
                "input": f"Case: {case}\nQuestion: {q}",
                "output": a,
                "case_name": case,
                "source": "kaggle: yogeshm01/indian-legal-qa-dataset-10k-questions"
            })
    
    print(f"Loaded {len(rows)} QA pairs from {SRC}")
    
    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(rows)
    
    # For D3 demo, use 1000, but keep full 10k available
    # Take 1000 for quick fine-tune, or 8000 for full
    # Here we create 1000 split: 800/100/100
    n = min(1000, len(rows))
    subset = rows[:n]
    print(f"Using {n} for fine-tuning (subset of {len(rows)})")
    
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = subset[:n_train]
    val = subset[n_train:n_train+n_val]
    test = subset[n_train+n_val:]
    
    for name, data in [("train", train), ("val", val), ("test", test)]:
        out = OUT_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {name}: {len(data)} -> {out}")
    

    full_out = OUT_DIR / "full_10k.jsonl"
    with open(full_out, "w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved full: {len(rows)} -> {full_out}")
    
    # Stats
    avg_q = sum(len(r["input"]) for r in subset) / len(subset)
    avg_a = sum(len(r["output"]) for r in subset) / len(subset)
    print(f"Avg input: {avg_q:.0f} chars, Avg output: {avg_a:.0f} chars")
    print(f"\nSample train:")
    print(json.dumps(train[0], indent=2, ensure_ascii=False)[:800])
    print(f"\nNext: Fine-tune with QLoRA on Colab/Kaggle T4")
    print(f"  python train.py --config configs/qlora.yaml")

if __name__ == "__main__":
    convert()
