# D3 Fine-Tune a Small Language Model with LoRA / QLoRA

> Track D / Intermediate-Advanced / 3-4 weeks. Narrow domain, 1k+ curated instructions, QLoRA 4-bit on free GPU, evaluate vs base, serve.

## What it is
Take a 7B open-weight model, build a domain instruction dataset, QLoRA-tune on Colab/Kaggle T4, compare to base on held-out with a rubric, merge adapter, serve via vLLM + Gradio.

## Build order (vault D3:33-34 — do in order)
1. Pick narrow domain — SQL for my schema, Indian legal summaries, or SAP support (narrow beats general).
2. Build dataset (70% of work): 1k+ examples, dedup, length filter, format validation. Document source + cleaning.
3. Split train/val/test **before** looking again.
4. Run base on test, record score — the comparison point.
5. QLoRA 4-bit on tiny subset to prove loop, then full.
6. Watch train/val loss curves — stop if val rises (overfit).
7. Evaluate vs base on held-out with rubric (wins/losses/ties; LLM-as-judge + 20 human).
8. Merge adapter, quantize, serve via vLLM + Gradio, publish model + dataset cards.

## Repo layout
```
data/           raw + cleaned JSONL, prepare.py
configs/        qlora.yaml (rank, alpha, LR, epochs)
train.py        QLoRA loop
eval/           held-out set + rubric
serve.py        vLLM/FastAPI + Gradio
```

## Interview gate (must answer before resume)
1. When to fine-tune vs RAG vs better prompt?
2. What does LoRA change inside the model, why does it save memory?
3. Validation loss climbed at epoch 3 — what and what did you do?

## Resume bullet template
Fine-tuned 7B open-weight model with QLoRA 4-bit on free-tier GPU over 1.2k curated domain examples; evaluated vs base on held-out with defined rubric and served merged quantised model via vLLM + Gradio.

## Next
Pick domain, create empty repo `finetune-lora`, commit this SKILL.md, start dataset curation. Ask me for domain pros/cons if torn.
