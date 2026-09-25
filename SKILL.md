---
project: finetune-lora
track: ai-ml
level: intermediate-advanced
started: 2026-09-25
shipped:
repo: https://github.com/Kaustubh070707/finetune-lora
live:
---
# 1. What this project is
Non-technical: I took a small open model and taught it to answer Indian legal questions in a fixed format, using 800 of my own examples.
Engineer: QLoRA 4-bit fine-tune of Mistral-7B-Instruct on 800 legal QA pairs, evaluated against the base on a held-out 100 with token-F1.

# 2. Problem it solves
The base model answers legal questions in free prose that drifts in format. After tuning on my 5-bullet and short-answer format, the same 7B model follows the format exactly, for cents on a free Colab T4.

# 3. Architecture
```
[Kaggle 10k legal QA] -> [clean + dedup + 800/100/100 split] -> [QLoRA rank-16 on T4]
[base Mistral-7B] vs [tuned] on held-out 100, token-F1 -> [merge + serve next]
```

# 4. Key decisions and trade-offs
| Decision | Options I considered | What I chose | Why | What I gave up |
|---|---|---|---|---|
| Domain | SQL for my schema vs Indian legal summaries vs SAP replies | Indian legal summaries (Kaggle 10k QA, Supreme Court) | Ready-made 10,002 Q/A pairs meant zero scraping, and legal answers have checkable facts so F1 actually measures something. | A domain tied to my own RAG corpus. |
| Base model | Mistral-7B vs Llama-3-8B vs 3B small | Mistral-7B-Instruct-v0.2 | Strong instruction following at 7B, fits T4 in 4-bit with room for batch 2 and 1024 context. | 13B quality and 3B iteration speed. |
| Tuning | Full fine-tune vs LoRA vs QLoRA | QLoRA 4-bit, rank 16, alpha 32, LR 2e-4, 3 epochs | 42M trainable out of 7.3B (0.58%) fits free T4 and trains in 2.5 hours. Full fine-tune needs paid GPUs. | Full fine-tune ceiling and any pretraining-knowledge changes. |
| Metric | LLM-as-judge vs human rubric vs token-F1 | Token-F1 greedy, 128 tokens | Free, deterministic, same ruler for base and tuned. Paraphrase scores low, so it understates quality — but the delta is honest. | Nuance that a judge would catch. |

# 5. Skills demonstrated
- [x] Transformer fine-tuning + PEFT/LoRA mechanics evidence: Colab notebook, `get_peft_model` rank-16, 42M trainable (0.58%)
- [x] Quantisation + memory-efficient training evidence: bitsandbytes NF4 double-quant, 7B in ~4GB on T4
- [x] Dataset curation + quality control evidence: `data/kaggle_to_instruction.py`, 10,002 → 800/100/100, fixed seed 42, local-only data
- [x] Training diagnostics evidence: step-50/100/150 train/val table below, overfit caught
- [x] Rigorous evaluation evidence: base 0.324 vs tuned 0.404 token-F1 on same held-out 100
- [ ] Efficient serving evidence: merge + vLLM/Gradio demo — not built yet

# 6. Numbers I measured
| Metric | Before (base) | After (tuned ckpt-150) | How I measured it |
|---|---|---|---|
| Held-out 100 token-F1, greedy 128 tokens | 0.324 | 0.404 (+25%) | Same 100 rows, same prompts, same scorer. Single adapter verified (`peft_config` one key) before trusting the number. |
| Train loss step 50/100/150 | 0.919 start of curve | 0.573 → 0.324 | SFTTrainer log, batch 2 x accum 8, LR 2e-4. |
| Val loss step 50/100/150 | 0.922 (best) | 0.937 → 1.065 | Same eval split every 50 steps. Train kept falling while val climbed after 50 — textbook overfit, so ckpt-50 was the val-best (lost: `save_steps` defaulted to 500, only ckpt-20/150 written). |
| Trainable share | 7.3B frozen | 42M (0.58%) | `print_trainable_parameters`. |

# 7. Things that broke and how I fixed them
1. Symptom: My first training cell died with `KeyError: 'text'` during tokenization, before a single step ran.
   Cause: This TRL version wants one lone `text` column, but my dataset carried instruction/input/output plus a half-added text field. The trainer looked up `text` and found a mess.
   Fix: I rebuild a clean single-column dataset in the train cell (`to_text` mapping with `remove_columns` on everything else) and pass that. Tokenization went 800/800 green immediately after.
   Lesson: Match the trainer's expected schema exactly before blaming the model. Print columns first, train second.
2. Symptom: PEFT printed "modifying a model for a second time... multiple adapters" twice, and I spent an hour convinced my 150-step run was corrupted.
   Cause: I reran the train cell on the same in-RAM `base`, so `get_peft_model` wrapped an already-wrapped model. Rerunning cells instead of inspecting state kept stacking the problem.
   Fix: I checked `list(model.peft_config.keys())` — one `default` key, 42M params. Single adapter, warnings were benign re-entry noise. I now treat that one line as the trust check before any eval number counts.
   Lesson: Warnings announce events; only state inspection reports truth. And one `get_peft_model` per kernel lifetime — restart first if in doubt.
3. Symptom: My best checkpoint (step 50, val 0.922) doesn't exist on disk. Only ckpt-20 and ckpt-150 were written.
   Cause: `save_steps` defaulted to 500 with 150 total steps, so nothing mid-run was ever saved. I learned this after the run, not before.
   Fix: Evaluating the overfit final honestly (0.404, still a gain) and scheduling the rerun with `save_steps=50, save_total_limit=3`. The number I report is from the final, not the best — stated plainly.
   Lesson: Set `save_steps` equal to `eval_steps` before any run longer than an hour. Checkpoints are the only version of the past you can evaluate.

# 8. What I would do differently at 100x scale
- I would train on the full 10k instead of the 800 subset, with a proper cluster and W&B curves, because 800 short QA pairs teach format more than law.
- I would add a human rubric on 50 held-out answers alongside F1, because token overlap punishes good paraphrases and my 0.404 understates real quality.
- I would serve the merged model behind vLLM with a Gradio demo and a model card, because an unevaluated, unserved adapter earns nothing.

# 9. Interview answers I have rehearsed
Q: When should you fine-tune instead of using RAG or a better prompt? Be specific.
A: When the task is format and style, not knowledge. My legal answers needed the same 5-bullet shape every time — RAG would still ramble, and prompting cost tokens per query forever. I fine-tuned because 800 examples taught the shape once, and retrieval still supplies the facts. If the bottleneck were missing documents, I would do RAG; if one sentence of instruction fixed it, a prompt.
Q: What does LoRA actually change inside the model, and why does that save so much memory?
A: It freezes all 7.3B weights in 4-bit and trains two small rank-16 matrices per attention and MLP layer — 42M params, 0.58%. Only those matrices need gradients and optimizer state, so a T4 fits what full fine-tuning never could. At inference the adapter merges back in.
Q: Your validation loss started climbing at epoch 3. What happened and what did you do?
A: Overfitting: train 0.919 to 0.324 while val went 0.922 to 1.065, best at step 50. I reported the final honestly at 0.404 F1 and scheduled the rerun with matching save steps so the val-best checkpoint actually exists next time.

# 10. Honest limitations
Single adapter from 800 short QA pairs, evaluated on token-F1 which punishes paraphrase — 0.404 understates real answer quality and overstates nothing. No merge, no served demo, no model card yet. The val-best checkpoint was never saved, so the reported number is from the overfit final, stated plainly.

# 11. How to run it
```bash
git clone https://github.com/Kaustubh070707/finetune-lora.git && cd finetune-lora
# data (local only, Kaggle 10k QA -> 800/100/100)
python data/kaggle_to_instruction.py
# train on Colab T4 (see README notebook cells 1-6)
# serve (next): merge adapter, vLLM + Gradio at http://localhost:8000
```

# 12. References
- Kaggle 10k Indian legal QA (yogeshm01) — dataset source, local only
- HuggingFace transformers + PEFT docs for QLoRA mechanics
- TRL SFTTrainer docs for the `text`-column contract that bit me
