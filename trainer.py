import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import RobertaTokenizerFast
from scripts.dataset import get_yelp_review_dataset
from scripts.model import TransformerModel

torch.set_float32_matmul_precision('medium')


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)

    def prepare_data(self):
        self.train_ds, self.val_ds, self.test_ds = get_yelp_review_dataset(self.tokenizer)

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=32)


# define the LightningModule
class Model(pl.LightningModule):
    def __init__(self, num_class):
        super().__init__()
        self.model = TransformerModel(num_class)

    def training_step(self, batch, batch_idx):
        input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        out = self.model(input_ids, attn_mask, labels)
        loss = out['loss']
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        out = self.model(input_ids, attn_mask, labels)
        loss = out['loss']
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
        labels = batch['label']

        out = self.model(input_ids, attn_mask, labels)
        loss = out['loss']
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


model = Model(10)
dl = DataModule(args)

trainer = pl.Trainer(max_epochs=5,
                     gpus=-1,
                     precision=16,
                     default_root_dir="bins/")

trainer.fit(model=model, train_dataloaders=dl)
