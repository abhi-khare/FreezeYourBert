import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class TransformerModel(nn.Module):

    def __init__(self, num_class):
        super(TransformerModel, self).__init__()
        self.base_model = RobertaModel.from_pretrained("roberta-base")

        self.fc = nn.Sequential(nn.Linear(768, 256),
                                nn.Dropout(0.2),
                                nn.Linear(256, num_class))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, label):
        encoded_output = self.base_model(input_ids, attention_mask)
        seq_output = encoded_output[0][:, 0]
        logits = self.fc(seq_output)

        loss = self.loss_fn(logits, label)
        label_pred = torch.argmax(nn.Softmax(dim=1)(logits))

        return {
            "loss": loss,
            "label_pred": label_pred
        }