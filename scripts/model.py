import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class TransformerModel(nn.Module):

    def __init__(self, num_class):
        super(TransformerModel, self).__init__()
        self.base_model = RobertaModel.from_pretrained("roberta-base")

        self.FC1 = nn.Linear(768, 256)
        self.drpt = nn.Dropout(0.2)
        self.FC2 = nn.Linear(256, num_class)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, label):
        encoded_output = self.base_model(input_ids, attention_mask)
        seq_output = encoded_output[0][:, 0]
        hidden_output = self.FC1(self.drpt(F.gelu(seq_output)))
        logits = self.intent_FC2(self.drpt(F.gelu(hidden_output)))

        loss = self.loss_fn(logits, label)
        label_pred = torch.argmax(nn.Softmax(dim=1)(logits), axis=1)

        return {
            "loss": loss,
            "label_pred": label_pred
        }