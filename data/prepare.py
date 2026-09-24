"""
Prepare instruction dataset from raw crawled judgments
Cleans, formats, and splits into train/val/test for QLoRA
"""
import json
import re
import hashlib
from pathlib import Path
from collections import Counter

RAW_FILE = Path("data/raw/indian_legal_raw.jsonl")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """Clean judgment text for training"""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    # Remove markdown artifacts
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove premium/ads leftover
    text = re.sub(r'Unlock Advanced Research.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Premium.*', '', text)
    return text.strip()

def create_instruction(example: dict) -> dict:
    """Convert raw judgment to instruction format"""
    text = clean_text(example["text"])
    # Truncate input for training (keep 800-1200 chars for context)
    input_text = text[:1200].strip()
    if len(input_text) < 400:
        return None
    
    # For demo, create synthetic summary (in production, use LLM or human summaries)
    # Here we extract first 2 sentences as pseudo-summary for pipeline test
    sentences = re.split(r'(?<=[.!?])\s+', input_text)
    if len(sentences) < 2:
        return None
    
    # Create structured summary template
    output = f"""- Facts: {example['title']} before {example['court']}. Matter concerning {sentences[0][:100]}...
- Issues: Whether the court should grant relief under the relevant legal provisions.
- Holdings: The court held as per the reasoning in the judgment, disposing of the matter accordingly.
- Reasoning: The court considered the facts and circumstances, heard counsel, and applied relevant precedents.
- Order: Disposed of in terms of the above, with directions as per the judgment text."""
    
    return {
        "instruction": "Summarize the Indian legal judgment in 5 bullets (facts, issues, holdings, reasoning, order).",
        "input": input_text,
        "output": output,
        "source": example["url"],
        "court": example["court"],
        "title": example["title"]
    }

def deduplicate(examples: list[dict]) -> list[dict]:
    """Deduplicate by input hash"""
    seen = set()
    deduped = []
    for ex in examples:
        h = hashlib.sha256(ex["input"].encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            deduped.append(ex)
    return deduped

def length_filter(examples: list[dict]) -> list[dict]:
    """Filter by length"""
    filtered = []
    for ex in examples:
        if 400 <= len(ex["input"]) <= 1500 and 300 <= len(ex["output"]) <= 800:
            filtered.append(ex)
    return filtered

def main():
    if not RAW_FILE.exists():
        print(f"Raw file not found: {RAW_FILE}")
        print("Run: python data/crawl_legal.py --count 50")
        return
    
    # Load raw
    raw = []
    for line in open(RAW_FILE, encoding="utf-8"):
        try:
            raw.append(json.loads(line))
        except:
            continue
    print(f"Loaded {len(raw)} raw judgments from {RAW_FILE}")
    
    # Convert to instruction format
    instructions = []
    for r in raw:
        inst = create_instruction(r)
        if inst:
            instructions.append(inst)
    
    print(f"Converted to {len(instructions)} instruction examples")
    if not instructions:
        print("No valid examples — check raw data")
        return
    
    # Deduplicate
    instructions = deduplicate(instructions)
    print(f"After dedup: {len(instructions)}")
    
    # Length filter
    instructions = length_filter(instructions)
    print(f"After length filter: {len(instructions)}")
    
    if len(instructions) < 10:
        print(f"Warning: Only {len(instructions)} examples after filtering. Need 1000+ for D3.")
        print("-> Crawl more: python data/crawl_legal.py --count 200")
    
    # Shuffle with fixed seed for reproducibility
    import random
    random.seed(42)
    random.shuffle(instructions)
    
    # Split: 80% train, 10% val, 10% test
    n = len(instructions)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train = instructions[:n_train]
    val = instructions[n_train:n_train+n_val]
    test = instructions[n_train+n_val:]
    
    # Save
    for split, data in [("train", train), ("val", val), ("test", test)]:
        out = PROCESSED_DIR / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {split}: {len(data)} examples -> {out}")
    
    # Save held-out test separately for final eval (never looked at during training)
    heldout = PROCESSED_DIR / "heldout.jsonl"
    with open(heldout, "w", encoding="utf-8") as f:
        for ex in test:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Stats
    courts = Counter(ex["court"] for ex in instructions)
    print(f"\nCourt distribution: {dict(courts.most_common(5))}")
    avg_in = sum(len(ex["input"]) for ex in instructions) / len(instructions)
    avg_out = sum(len(ex["output"]) for ex in instructions) / len(instructions)
    print(f"Avg input: {avg_in:.0f} chars, Avg output: {avg_out:.0f} chars")
    
    print(f"\nDataset ready in {PROCESSED_DIR}/")
    print(f"Next: Fine-tune with QLoRA")
    print(f"  python train.py --config configs/qlora.yaml  # on Colab/Kaggle T4")
    
    # Sample preview
    if train:
        print(f"\nSample train example:")
        print(json.dumps(train[0], indent=2, ensure_ascii=False)[:1000])

if __name__ == "__main__":
    main()
