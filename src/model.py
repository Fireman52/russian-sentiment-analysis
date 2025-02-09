import torch
import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):
    """
    BERT-based model for sentiment classification.
    """

    def __init__(self, bert_model_name: str, hidden_dim: int = 256, num_classes: int = 3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
