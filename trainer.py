import torch
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from scripts.dataset import get_dataloader
from scripts.model import TransformerModel
from arguments import arguments


# setting platform specific params
torch.set_float32_matmul_precision('medium')
# training arguments
args = arguments()
# tensorboard logger setup
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

train_dl, val_dl, test_dl, num_class = get_dataloader(args)


# define the LightningModule
class Model(pl.LightningModule):
    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
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
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer


model = Model(args, num_class)

trainer = pl.Trainer(
    gpus=-1,
    precision=args.precision,
    max_epochs=args.max_epoch,
    deterministic=args.deterministic,
    logger=tb_logger)

trainer.fit(model=model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl)

trainer.test(model=model,
             dataloaders=test_dl)
