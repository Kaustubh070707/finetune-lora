# Fine-Tune 7B for Indian Legal QA (QLoRA)

A 7B open model taught to answer Indian legal questions in a fixed format — 800 examples, free Colab T4, measured gain.

## Results (same held-out 100 throughout)

| Stage | Score | Notes |
|---|---|---|
| Base Mistral-7B-Instruct, greedy 128 tokens | F1 0.324 | Before. Token-F1 vs reference, deterministic. |
| QLoRA rank-16 tuned (ckpt-150) | **F1 0.404 (+25%)** | 42M trainable (0.58%). Same 100 rows, same scorer, single adapter verified. |
| Train loss step 50/100/150 | 0.919 → 0.573 → 0.324 | Falls throughout. |
| Val loss step 50/100/150 | 0.922 → 0.937 → 1.065 | Best at 50 — overfit after. ckpt-50 was never saved (`save_steps` default 500); reported number is the honest final. |

## How it was built

- **Data:** Kaggle 10k Indian legal QA → cleaned, deduped, fixed seed 42 → 800/100/100 (`data/kaggle_to_instruction.py`, local only).
- **Train:** 4-bit NF4 double-quant, rank 16 / alpha 32 / LR 2e-4 / 3 epochs / batch 2×8 / 1024 ctx, eval every 50 steps. 20-step proof run first, then full ~2.5 hrs.
- **Eval:** identical prompts + token-F1 on held-out 100 for base and tuned.

## Repo layout

```
data/kaggle_to_instruction.py   10k -> 800/100/100 (Kaggle CSV stays local)
adapters/                       NOT in git — Drive backup (ckpt-150 zip)
SKILL.md                        engineering log — decisions, numbers, failures, answers
```

## Limitations (honest)

800 short pairs teach format more than law; F1 punishes good paraphrase so 0.404 understates quality; val-best checkpoint lost to `save_steps` default; no merge, demo, or cards yet.

## Rerun on Colab (T4)

1. Upload `train/val/test.jsonl` to `/content` (or `cp` from Drive `d3/`).
2. Baseline cell → record F1. Train cells → proof 20 steps, then full.
3. Eval cell on ckpt → compare. Save adapter zip to Drive before the VM dies.
