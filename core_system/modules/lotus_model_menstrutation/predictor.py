import numpy as np
import torch
from .loader import model, tokenizer, device

id2label = model.config.id2label  

#print(f"Loaded model with {len(id2label)} labels: {id2label}")
def predict_emotions_menstrutation(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = 1 / (1 + np.exp(-logits.cpu().numpy()[0]))
   
    # Convert i to string key to match id2label dictionary
    return [(id2label.get(str(i), model.config.id2label[i]), float(p)) for i, p in enumerate(probs) if p >= threshold]

