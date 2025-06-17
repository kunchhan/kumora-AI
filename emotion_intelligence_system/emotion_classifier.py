"""
Kumora Emotion Intelligence Module
Multi-Label Emotion Classifier Implementation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report
)
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class EmotionConfig:
    """Configuration for Emotion Classifier"""
    model_name: str = "distilbert-base-uncased"
    test_size: float = 0.2
    val_size: float = 0.1
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 27 emotion labels
    emotion_labels: List[str] = None
    
    def __post_init__(self):
        self.emotion_labels = [
            'Mood swings', 'Irritability', 'Anxiety', 'Sadness', 'Tearfulness',
            'Anger or frustration', 'Emotional sensitivity', 'Feeling overwhelmed',
            'Low self-esteem', 'Loneliness or Isolation', 'Restlessness',
            'Sensitivity to rejection', 'Physical discomfort', 'Improved mood',
            'Hopefulness', 'Renewed energy', 'Optimism', 'Productivity',
            'Clarity', 'Feeling in control', 'Confidence', 'High energy',
            'Sociability', 'Attractiveness', 'Empowerment', 'Sexual drive',
            'Motivation'
        ]
        self.num_labels = len(self.emotion_labels)


class EmotionDataset(Dataset):
    """Custom Dataset for Multi-Label Emotion Classification"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }


class MultiLabelEmotionClassifier(nn.Module):
    """Multi-Label Emotion Classifier using Transformer backbone"""
    
    def __init__(self, config: EmotionConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)
        self.transformer_config = AutoConfig.from_pretrained(config.model_name)
        
        # Classification layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(
            self.transformer_config.hidden_size, 
            config.num_labels
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hidden_size, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, num_labels)
        # )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output or mean of last hidden states
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling
            last_hidden_states = outputs.last_hidden_state
            pooled_output = self.mean_pooling(last_hidden_states, attention_mask)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def mean_pooling(self, last_hidden_states, attention_mask):
        """Mean pooling considering attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class EmotionClassifierTrainer:
    """Trainer for Multi-Label Emotion Classifier"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize model
        self.model = MultiLabelEmotionClassifier(config).to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Best thresholds for each label
        self.best_thresholds = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders from dataframe"""
        # Extract texts and labels
        texts = df['preprocessed_text'].tolist()
        
        # Extract emotion labels as multi-label array
        labels = np.zeros((len(df), self.config.num_labels))
        for i, emotion in enumerate(self.config.emotion_labels):
            if emotion in df.columns:
                labels[:, i] = df[emotion].values
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=self.config.test_size, random_state=42, stratify=labels.sum(axis=1)
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.val_size/(1-self.config.test_size), random_state=42, stratify=y_temp.sum(axis=1)
        )
        
        # Create datasets
        train_dataset = EmotionDataset(X_train, y_train, self.tokenizer, self.config.max_length)
        val_dataset = EmotionDataset(X_val, y_val, self.tokenizer, self.config.max_length)
        test_dataset = EmotionDataset(X_test, y_test, self.tokenizer, self.config.max_length)
        
        print("Size of Train, Val and Test Sets after splits:\n")
        print(f"Length of Training Set:\t\t{len(train_dataset)}")
        print(f"Length of Validation Set:\t{len(val_dataset)}")
        print(f"Length of Testing Set:\t\t{len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        # Calculate class weights for imbalanced data
        self.calculate_class_weights(y_train)
        
        return train_loader, val_loader, test_loader
    
    def calculate_class_weights(self, labels: np.ndarray):
        """Calculate class weights for handling imbalanced data"""
        pos_counts = labels.sum(axis=0)
        neg_counts = len(labels) - pos_counts
        
        # Calculate positive weights
        pos_weights = neg_counts / (pos_counts + 1e-5)
        
        # Convert to tensor
        self.pos_weights = torch.FloatTensor(pos_weights).to(self.device)
        
        # Update loss function with weights
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader, optimize_threshold: bool = False) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Apply sigmoid to get probabilities
        all_probs = torch.sigmoid(all_logits).numpy()
        
        # Optimize thresholds if requested
        if optimize_threshold:
            self.best_thresholds = self.optimize_thresholds(all_probs, all_labels)
            thresholds = self.best_thresholds
        else:
            thresholds = self.best_thresholds if self.best_thresholds is not None else [self.config.threshold] * self.config.num_labels
        
        # Apply thresholds
        all_predictions = (all_probs > thresholds).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_predictions)
        
        return metrics
    
    def optimize_thresholds(self, probs: np.ndarray, labels: np.ndarray) -> List[float]:
        """Optimize threshold for each emotion label"""
        best_thresholds = []
        
        for i in range(self.config.num_labels):
            best_threshold = 0.5
            best_f1 = 0
            
            # Try different thresholds
            for threshold in np.arange(0.2, 0.8, 0.05):
                preds = (probs[:, i] > threshold).astype(int)
                f1 = f1_score(labels[:, i], preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            best_thresholds.append(best_threshold)
            print(f"{self.config.emotion_labels[i]}: Best threshold = {best_threshold:.2f}, F1 = {best_f1:.3f}")
        
        return best_thresholds
    
    def calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        # Overall metrics
        micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Create metrics dictionary
        metrics = {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'per_label_metrics': {}
        }
        
        for i, emotion in enumerate(self.config.emotion_labels):
            metrics['per_label_metrics'][emotion] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        patience = 0
        max_patience = 5
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch+1)
            # print(f"Training Loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, optimize_threshold=(epoch == 0))
            # print(f"Validation Micro F1: {val_metrics['micro_f1']:.4f}")
            # print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"Train Loss: {train_loss:.4f}, Val F1 Micro: {val_metrics['micro_f1']:.4f}, Val F1 Macro: {val_metrics['macro_f1']:.4f}")
            
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                patience = 0
                
                self.save_model(f"best_emotion_model_epoch_{epoch + 1}")
                print(f"Saved best model with F1 Macro: {best_val_f1:.4f}!")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
        print(f"\nTraining completed. Best Validation F1: {best_val_f1:.4f}")
    
    def save_model(self, path: str):
        """Save model and configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_thresholds': self.best_thresholds
        }, f"{path}.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """Load saved model"""
        checkpoint = torch.load(f"{path}.pt", weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_thresholds = checkpoint['best_thresholds']
        self.config = checkpoint['config']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)


class EmotionIntelligenceModule:
    """Main Emotion Intelligence Module for Kumora"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = EmotionConfig()
        self.trainer = EmotionClassifierTrainer(self.config)
        
        if model_path:
            self.trainer.load_model(model_path)
            self.model = self.trainer.model
            self.tokenizer = self.trainer.tokenizer
            self.thresholds = self.trainer.best_thresholds
        else:
            self.model = None
            self.tokenizer = None
            self.thresholds = [0.5] * self.config.num_labels
    
    def analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions in given text"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.trainer.device)
        attention_mask = encoding['attention_mask'].to(self.trainer.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply thresholds
        predictions = (probs > self.thresholds).astype(int)
        
        # Prepare results
        detected_emotions = []
        emotion_scores = {}
        
        for i, (emotion, prob, pred) in enumerate(zip(self.config.emotion_labels, probs, predictions)):
            emotion_scores[emotion] = float(prob)
            if pred == 1:
                detected_emotions.append(emotion)
        
        # Identify primary emotion (highest probability)
        primary_emotion_idx = np.argmax(probs)
        primary_emotion = self.config.emotion_labels[primary_emotion_idx]
        
        # Calculate emotional intensity (average of detected emotion probabilities)
        if detected_emotions:
            intensity = np.mean([emotion_scores[e] for e in detected_emotions])
        else:
            intensity = 0.0
        
        # Categorize emotions
        negative_emotions = ['Mood swings', 'Irritability', 'Anxiety', 'Sadness', 'Tearfulness',
                           'Anger or frustration', 'Emotional sensitivity', 'Feeling overwhelmed',
                           'Low self-esteem', 'Loneliness or Isolation', 'Restlessness',
                           'Sensitivity to rejection', 'Physical discomfort']
        
        positive_emotions = ['Improved mood', 'Hopefulness', 'Renewed energy', 'Optimism',
                           'Productivity', 'Clarity', 'Feeling in control', 'Confidence',
                           'High energy', 'Sociability', 'Attractiveness', 'Empowerment',
                           'Sexual drive', 'Motivation']
        
        emotional_valence = self._calculate_valence(detected_emotions, negative_emotions, positive_emotions)

        emotional_confidence = self._calculate_confidence(probs, predictions, np.array(self.thresholds))
        
        # Add logic to get emotional_confidence = self._calculate_confidence()
        return {
            'detected_emotions': detected_emotions,
            'emotion_scores': emotion_scores,
            'primary_emotion': primary_emotion,
            'emotional_intensity': float(intensity),
            'emotional_valence': emotional_valence,
            'emotional_confidence': emotional_confidence,
            'total_emotions': len(detected_emotions),
            'raw_probabilities': probs.tolist()
        }
    
    def _calculate_valence(self, detected_emotions: List[str], 
                          negative_emotions: List[str], 
                          positive_emotions: List[str]) -> str:
        """Calculate overall emotional valence"""
        neg_count = sum(1 for e in detected_emotions if e in negative_emotions)
        pos_count = sum(1 for e in detected_emotions if e in positive_emotions)
        
        if neg_count > pos_count:
            return "negative"
        elif pos_count > neg_count:
            return "positive"
        else:
            return "mixed"
    
    # def _calculate_confidence(self, probs: np.ndarray, predictions: np.ndarray, thresholds: List[float]) -> float:
    #     """
    #     Calculates a confidence score based on the model's output probabilities.
    #     The confidence is the average probability of the detected emotions, penalized
    #     by the proximity of undetected emotions to their thresholds.
    #     """
    #     detected_indices = np.where(predictions == 1)[0]
        
    #     if len(detected_indices) == 0:
    #         # Confidence in "no emotion" is higher when the max prob is far below its threshold.
    #         max_prob = np.max(probs)
    #         return float(np.clip(1.0 - max_prob, 0.0, 1.0))

    #     # Base confidence is the average probability of detected emotions.
    #     detected_probs = probs[detected_indices]
    #     confidence = np.mean(detected_probs)
        
    #     # Penalize confidence if other emotions were "close calls".
    #     undetected_indices = np.where(predictions == 0)[0]
    #     if len(undetected_indices) > 0:
    #         undetected_probs = probs[undetected_indices]
    #         # Using a fixed threshold of 0.5 for simplicity in nearness calculation
    #         nearness = np.mean([p for p in undetected_probs if p > 0.3]) # Avg prob of "near misses"
    #         if not np.isnan(nearness):
    #             # Dampen confidence by how close other labels were to being predicted
    #             confidence *= (1.0 - nearness * 0.25) # Dampening factor
                
    #     return float(np.clip(confidence, 0.0, 1.0))
    def _calculate_confidence(self, probs: np.ndarray, predictions: np.ndarray, thresholds: np.ndarray) -> float:
        """
        Calculates a confidence score based on the model's output probabilities.
        The confidence is the average probability of the detected emotions, penalized
        by the proximity of undetected emotions to their own specific thresholds.
        """
        detected_indices = np.where(predictions == 1)[0]
        
        if len(detected_indices) == 0:
            # Confidence in "no emotion" is higher when the max prob is far below its threshold.
            max_prob_idx = np.argmax(probs)
            max_prob = probs[max_prob_idx]
            threshold_for_max = thresholds[max_prob_idx]
            # Confidence is how far the max probability is from its own threshold
            return float(np.clip(1.0 - (max_prob / threshold_for_max), 0.0, 1.0))

        # Base confidence is the average probability of detected emotions.
        detected_probs = probs[detected_indices]
        confidence = np.mean(detected_probs)
        
        # Penalize confidence if other emotions were "close calls".
        undetected_indices = np.where(predictions == 0)[0]
        if len(undetected_indices) > 0:
            # Calculate how close each undetected prob was to its specific threshold
            # Ratios close to 1.0 are "near misses".
            nearness_ratios = probs[undetected_indices] / thresholds[undetected_indices]
            
            # Consider only those that are reasonably close (e.g., > 75% of the way to the threshold)
            close_calls = nearness_ratios[nearness_ratios > 0.75]
            
            if len(close_calls) > 0:
                # Average the "closeness" of the near misses
                avg_nearness = np.mean(close_calls)
                # Dampen confidence by how close other labels were to being predicted
                confidence *= (1.0 - (avg_nearness - 0.75) * 0.5) # Penalize more as nearness approaches 1.0

        return float(np.clip(confidence, 0.0, 1.0))

    def get_emotion_context(self, emotion_analysis: Dict) -> Dict:
        """Generate context for response generation"""
        context = {
            'primary_emotion': emotion_analysis['primary_emotion'],
            # 'secondary_emotions': [e for e in emotion_analysis['detected_emotions'] 
            #                      if e != emotion_analysis['primary_emotion']],
            'detected_emotions': emotion_analysis['detected_emotions'],
            'intensity': emotion_analysis['emotional_intensity'],
            'valence': emotion_analysis['emotional_valence'],
            # 'support_type': self._determine_support_type(emotion_analysis)
            'confidence': emotion_analysis['emotional_confidence']
        }
        
        return context
    
    def _determine_support_type(self, emotion_analysis: Dict) -> str:
        """Determine the type of support needed based on emotions"""
        detected = set(emotion_analysis['detected_emotions'])
        
        # Crisis emotions
        crisis_emotions = {'Feeling overwhelmed', 'Low self-esteem', 'Loneliness or Isolation'}
        if len(detected & crisis_emotions) >= 2:
            return "crisis_support"
        
        # Validation needed
        validation_emotions = {'Sadness', 'Tearfulness', 'Anxiety', 'Anger or frustration'}
        if len(detected & validation_emotions) >= 1:
            return "emotional_validation"
        
        # Growth support
        growth_emotions = {'Hopefulness', 'Motivation', 'Empowerment', 'Confidence'}
        if len(detected & growth_emotions) >= 2:
            return "growth_encouragement"
        
        # Default support
        return "general_support"


# Example usage and training script
def main():
    """Example training and inference"""
    # Load your preprocessed data
    df = pd.read_csv('kumora_emotion_dataset.csv')
    
    # Initialize config and trainer
    config = EmotionConfig()
    trainer = EmotionClassifierTrainer(config)
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(df)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest Set Performance:")
    print(f"Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Save final model
    trainer.save_model("kumora_emotion_model_final")
    
    # Example inference
    emotion_module = EmotionIntelligenceModule("kumora_emotion_model_final")
    
    # Test examples
    test_texts = [
        "I feel so overwhelmed and anxious about everything happening in my life",
        "Today was amazing! I feel so confident and motivated to tackle my goals",
        "I'm crying again and I don't know why. Everything just feels too much",
        "Finally starting to feel like myself again. The clarity is refreshing"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        analysis = emotion_module.analyze_emotions(text)
        print(f"Primary Emotion: {analysis['primary_emotion']}")
        print(f"Detected Emotions: {', '.join(analysis['detected_emotions'])}")
        print(f"Emotional Intensity: {analysis['emotional_intensity']:.2f}")
        print(f"Support Type: {emotion_module.get_emotion_context(analysis)['support_type']}")


if __name__ == "__main__":
    main()