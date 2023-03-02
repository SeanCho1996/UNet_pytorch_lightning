import logging
import os
from argparse import ArgumentParser

import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from torch import nn

from lightning_dataset import SegDataModule
from unet import UNet


class UNetSegmentation(pl.LightningModule):
    def __init__(self, input_channels:int = 3, num_classes:int = 2, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super().__init__()
        self.model = UNet(input_channels=input_channels, num_classes=num_classes)
        
        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

    def forward(self, inputs):
        """
        :param inputs: Input image tensor
        :return: output - segmentation mask feature
        """
        output = self.model(inputs)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented argument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=4,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            metavar="LR",
            help="learning rate (default: 0.0001)",
        )
        return parser
    
    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function
        :return: output - Initialized cross entropy loss function
        """
        return F.cross_entropy(logits, labels, ignore_index=255)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :param batch_idx: Batch indices
        :return: output - Training loss
        """
        inputs = train_batch[0].to(self.device)
        masks = train_batch[1].to(self.device)

        output = self.forward(inputs)
        loss = self.cross_entropy_loss(output.squeeze(1), masks.squeeze(1))
        self.log("loss", loss)

        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Testing accuracy
        """
        inputs = test_batch[0].to(self.device)
        masks = test_batch[1].to(self.device)

        output = self.forward(inputs)

        predict = torch.argmax(nn.Softmax(dim=1)(output), dim=1)
        pure_mask = masks.masked_select(masks.ne(255))
        pure_predict = predict.masked_select(masks.ne(255))
        acc = pure_mask.eq(pure_predict).sum()/len(pure_mask)

        return {"test_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - valid step loss
        """

        inputs = val_batch[0].to(self.device)
        masks = val_batch[1].to(self.device)

        output = self.forward(inputs)

        predict = torch.argmax(nn.Softmax(dim=1)(output), dim=1)
        pure_mask = masks.masked_select(masks.ne(255))
        pure_predict = predict.masked_select(masks.ne(255))
        acc = pure_mask.cpu().eq(pure_predict.cpu()).sum()/len(pure_mask)
        
        loss = self.cross_entropy_loss(output, masks)
        return {"val_step_loss": loss,
                "val_step_acc": acc}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_step_acc"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_acc", avg_acc, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]
    

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog UNet Example")

    # Early stopping parameters
    parser.add_argument(
        "--es_monitor", type=str, default="val_loss", help="Early stopping monitor parameter"
    )
    parser.add_argument(
        "--es_mode", type=str, default="min", help="Early stopping mode parameter"
    )
    parser.add_argument(
        "--es_verbose", type=bool, default=True, help="Early stopping verbose parameter"
    )
    parser.add_argument(
        "--es_patience", type=int, default=3, help="Early stopping patience parameter"
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = UNetSegmentation.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    if "strategy" in dict_args:
        if dict_args["strategy"] == "None":
            dict_args["strategy"] = None

    if "devices" in dict_args:
        if dict_args["devices"] == "None":
            dict_args["devices"] = None

    model = UNetSegmentation(input_channels=3, num_classes=2, **dict_args)

    sd = SegDataModule(data_dir=f"./PNG", **dict_args)
    sd.prepare_data()
    sd.setup(stage="train")

    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_acc", mode="max"
    )

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping, checkpoint_callback]
    )

    # It is safe to use `mlflow.pytorch.autolog` in DDP training, as below condition invokes
    # autolog with only rank 0 gpu.
    with mlflow.start_run() as mlrun:
        # For CPU Training
        devices_list = [int(x.strip()) for x in dict_args["devices"].split(",") if len(x) > 0]
        if dict_args["devices"] is None or 0 in devices_list:
            mlflow.pytorch.autolog()
        elif 0 not in devices_list and trainer.global_rank == 0:
            # In case of multi gpu training, the training script is invoked multiple times,
            # The following condition is needed to avoid multiple copies of mlflow runs.
            # When one or more gpus are used for training, it is enough to save
            # the model and its parameters using rank 0 gpu.
            mlflow.pytorch.autolog()
        else:
            # This condition is met only for multi-gpu training when the global rank is non zero.
            # Since the parameters are already logged using global rank 0 gpu, it is safe to ignore
            # this condition.
            logging.info("Active run exists.. ")

        trainer.fit(model, sd)
        trainer.test(datamodule=sd, ckpt_path="best")