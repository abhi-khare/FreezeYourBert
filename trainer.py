import torch
from torch import optim
import pytorch_lightning as pl
from scripts.dataset import get_dataloader
from scripts.model import TransformerModel
from arguments import arguments

# setting platform specific params
torch.set_float32_matmul_precision('medium')

# training arguments
args = arguments()

train_ds, val_ds, test_ds, num_class = get_dataloader(args)


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


model = Model(args)

# logger = tb_logger, callbacks = [checkpoint_callback]

trainer = pl.Trainer(
    gpus=-1,
    deterministic=args.deterministic,
    precision=args.precision,
    max_epochs=args.max_epoch,
    check_val_every_n_epoch=1,
    default_root_dir="bins/")

trainer.fit(model=model,
            train_dataloaders=train_ds,
            val_dataloaders=val_ds)

trainer.test(model=model,
             dataloaders=test_ds)
