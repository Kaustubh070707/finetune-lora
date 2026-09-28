"""Serve merged legal model: FastAPI + Gradio.
Needs GPU VRAM for 7B (Colab T4 ok, Railway free is not).
Model: https://huggingface.co/Kaustubh070707/mistral-7B-legal-qlora-5bullet
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = os.getenv("D3_MODEL_ID", "Kaustubh070707/mistral-7B-legal-qlora-5bullet")

app = FastAPI(title="D3 Legal Summarizer — QLoRA 7B")


class SummarizeRequest(BaseModel):
    case_text: str
    max_new_tokens: int = 256


_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline

        _pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            device_map="auto",
            max_new_tokens=256,
        )
    return _pipe


PROMPT = (
    "Summarize the Indian legal judgment in 5 bullets "
    "(facts, issues, holdings, reasoning, order).\n\nInput:\n{case_text}\n\nSummary:\n"
)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    text = PROMPT.format(case_text=req.case_text[:2000])
    out = get_pipe()(text, max_new_tokens=req.max_new_tokens, do_sample=False)[0][
        "generated_text"
    ]
    summary = out[len(text):].strip()
    return {"summary": summary, "model": MODEL_ID}


def gradio_ui():
    import gradio as gr

    def run(case_text: str):
        text = PROMPT.format(case_text=(case_text or "")[:2000])
        out = get_pipe()(text, max_new_tokens=256, do_sample=False)[0][
            "generated_text"
        ]
        return out[len(text):].strip()

    demo = gr.Interface(
        fn=run,
        inputs=gr.Textbox(lines=10, label="Judgment excerpt"),
        outputs=gr.Textbox(lines=12, label="5-bullet summary"),
        title="Indian Legal Summarizer (QLoRA 7B)",
        description="Fine-tuned Mistral-7B rank-16 on 800 legal QA — F1 0.324 to 0.386 merged.",
    )
    return demo


if __name__ == "__main__":
    import sys

    if "--share" in sys.argv:
        gradio_ui().launch(share=True)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
