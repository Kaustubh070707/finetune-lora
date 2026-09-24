---
project: finetune-lora
track: ai-ml
level: intermediate-advanced
started: 2026-09-25
shipped:
repo:
live:
---
# 1. What this project is
Non-technical: Take a small open model and teach it to do one job well on my own examples.
Engineer: QLoRA 4-bit fine-tune of a 7B open-weight model on a 1k+ instruction dataset, evaluated vs base, served via vLLM/Gradio.

# 2. Problem it solves
Base models are generalists — they need many tokens and still hallucinate on narrow tasks (e.g., SQL for my schema, Indian legal summaries). A 1k-example LoRA adapter makes the same 7B model follow my format exactly, for cents on a free GPU.

# 3. Architecture
```
[raw data] -> [clean + dedup + format -> instruction JSONL] -> [train/val/test split]
[base 7B] + [QLoRA 4-bit] -> [adapter] -> [merge + quantize] -> [vLLM/FastAPI + Gradio]
[eval: base vs tuned on held-out + LLM-as-judge]
```

# 4. Key decisions and trade-offs
| Decision | Options | Chose | Why | Gave up |
|---|---|---|---|---|
| Base model | 7B vs 3B vs 13B | 7B TBD after VRAM test on Colab T4 | Fits free tier with QLoRA 4-bit, strong enough for instruction following | 13B quality, 3B speed |
| Tuning | Full FT vs LoRA vs QLoRA | QLoRA 4-bit | Fits 16GB VRAM, trains in hours not days | Full FT ceiling |
| Dataset | 1k vs 5k | 1k curated narrow domain first | 70% of work is data quality; narrow beats general | Broad coverage |

# 5. Skills demonstrated
- [ ] Transformer fine-tuning + PEFT/LoRA mechanics evidence: `train.py` + adapter
- [ ] Quantisation + memory-efficient training evidence: bitsandbytes 4-bit + QLoRA config
- [ ] Dataset curation + quality control evidence: `data/` + cleaning notebook
- [ ] Training diagnostics evidence: loss curves (W&B/TensorBoard)
- [ ] Rigorous evaluation evidence: base vs tuned on held-out with rubric
- [ ] Efficient serving evidence: vLLM or FastAPI wrapper + Gradio demo

# 6. Numbers I measured
| Metric | Before (base) | After (tuned) | How I measured it |
|---|---|---|---|
| held-out rubric win rate | TBD | TBD | LLM-as-judge + human 20-sample |
| training loss | — | — | W&B |
| inference latency | — | — | vLLM vs baseline |

# 7. Things that broke and how I fixed them
1. Symptom:
   Cause:
   Fix:
   Lesson:

# 8. What I would do differently at 100x scale
- TBD
- TBD
- TBD

# 9. Interview answers I have rehearsed
Q: When should you fine-tune instead of using RAG or a better prompt? Be specific.
A:
Q: What does LoRA actually change inside the model, and why does that save so much memory?
A:
Q: Your validation loss started climbing at epoch 3. What happened and what did you do?
A:

# 10. Honest limitations
 Narrow domain, 1k examples, single GPU, no RLHF — not a general assistant.

# 11. How to run it
```bash
git clone <repo> && cd finetune-lora
cp .env.example .env
# data prep
python data/prepare.py
# train on Colab/Kaggle
python train.py --config configs/qlora.yaml
# serve
python serve.py  # FastAPI + Gradio at http://localhost:8000
```

# 12. References
- https://github.com/mlabonne/llm-course
- https://github.com/mlabonne/llm-datasets
- https://github.com/karpathy/nanoGPT
- https://github.com/microsoft/generative-ai-for-beginners
