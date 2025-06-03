# modal_setup.py

import modal

# Register the Modal App under a unique name in your Modal account
app = modal.App("agent-lifecycle-retrain")


# ───── Drift Embedding Function (CPU-based) ─────
@app.function(
    gpu=None,  # CPU only
    image=modal.Image.debian_slim().pip_install("sentence-transformers"),
    timeout=900,  # 15 minutes max
)
def compute_drift_score(log_texts: list[str], baseline_texts: list[str]) -> float:
    """
    Compute embeddings for baseline_texts and log_texts using all-MiniLM-L6-v2,
    calculate average pairwise cosine similarity, and return a drift score = 1 - similarity.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from numpy.linalg import norm

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_recent = model.encode(log_texts, convert_to_numpy=True)
    emb_baseline = model.encode(baseline_texts, convert_to_numpy=True)

    def pairwise_cosine_mean(a, b):
        sims = []
        for x in a:
            for y in b:
                sims.append(np.dot(x, y) / (norm(x) * norm(y)))
        return np.mean(sims)

    similarity = pairwise_cosine_mean(emb_baseline, emb_recent)
    drift_score = round(1.0 - similarity, 2)

    return drift_score


# ───── Optional Retraining Function (GPU-based) ─────
@app.function(
    gpu="A100",  # Use GPU if available under your $250 credits
    image=modal.Image.debian_slim().pip_install(
        "transformers",
        "datasets",
        "torch",
        "sentence-transformers",
        "bitsandbytes"
    ),
    timeout=3600,  # 1 hour max
)
def trigger_retraining_job(base_version: str) -> dict:
    """
    1. Download the base model (e.g., Flan-T5-Small) from Hugging Face.
    2. Gather recent logs.
    3. Fine-tune a small model on that data.
    4. Push the new model to HF Hub or S3.
    5. Return {"new_version": "<new_version_id>"}.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # (Actual fine-tuning steps would go here; omitted for brevity.)

    new_version = "v1.1.0-open"
    return {"new_version": new_version}
