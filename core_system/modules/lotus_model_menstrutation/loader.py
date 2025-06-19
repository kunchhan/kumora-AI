from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "../../../models/lotus_menstrual_emotion/lotus_menstrual_emotion_model_v1/"    
best_weights_path = "../../../models/lotus_menstrual_emotion/lotus_menstrual_emotion_classifier_v1.pt"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)

# 2. Load your best weights
model.load_state_dict(torch.load(best_weights_path))  # your saved .pt file
model.eval()

# Detect MPS or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
