import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

class KoBERTClassifier:
    def __init__(self, model_name='monologg/kobert', num_labels=2, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.model.eval()

    def predict_prob(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            return probs  # [real_prob, fake_prob] 또는 [fake_prob, real_prob] 구조에 주의

# 전역 객체 생성 및 편의 함수 제공
_classifier = KoBERTClassifier()

def predict_prob(text):
    return _classifier.predict_prob(text)
