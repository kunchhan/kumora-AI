import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(base_dir, "../../../models/lotus_general_emotion/lotus_final_model/"))
best_weights_path = os.path.abspath(os.path.join(base_dir, "../../../models/lotus_general_emotion/lotus_best_model_epoch3.pt"))


# model_path = "../../../models/lotus_general_emotion/lotus_final_model/"
# best_weights_path = "../../../models/lotus_general_emotion/lotus_best_model_epoch3.pt"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)


# 2. Load your best weights
state_dict = torch.load(best_weights_path, map_location="cpu")  # use map_location to avoid GPU mismatch
model.load_state_dict(state_dict)
model.eval()

# Detect MPS or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
