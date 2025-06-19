import numpy as np
import torch
from .loader import model, tokenizer, device

id2label = model.config.id2label  # This will now have correct 27 keys


def predict_emotions(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = 1 / (1 + np.exp(-logits.cpu().numpy()[0]))

    return [(id2label[i], float(p)) for i, p in enumerate(probs) if p >= threshold]
