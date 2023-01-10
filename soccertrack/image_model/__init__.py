import torch
import torchmetrics
from torchvision import transforms, models
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class ImageClassificationData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        im_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.im_size = im_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((im_size, im_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.dims = (3, im_size, im_size)
        self.num_classes = 2

    def setup(self, stage=None):
        self.trainset = ImageFolder(
            self.data_dir / "trainset", transform=self.transform
        )
        self.valset = ImageFolder(self.data_dir / "valset", transform=self.transform)
        self.testset = ImageFolder(self.data_dir / "testset", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ImageEmbedder(pl.LightningModule):
    def __init__(self, num_classes, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # We take in input dimensions as parameters and use those to dynamically build model.
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # init a pretrained resnet
        backbone = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        # add a layer with `hidden_size` units
        self.feature_extractor = nn.Sequential(
            *layers, nn.Flatten(), nn.Linear(num_filters, hidden_size)
        )

        # add a layer with `num_classes` units
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.feature_extractor(x)
        if return_embeddings:
            return x
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        return self.forward(x, return_embeddings=False).argmax(dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
