from functools import partial
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose


def single_agent_collate_fn(batch):
    x = torch.Tensor([seq for item in batch for seq in rearrange(item[0], "L N D ->  N L D")])
    y = torch.Tensor([seq for item in batch for seq in rearrange(item[1], "L N D ->  N L D")])
    # x, y = default_collate(batch)

    # B: batch size, L: sequence length, N: number of agents, D: dimension
    # x = rearrange(x, "B L N D -> (B N) L D")
    # y = rearrange(y, "B L N D -> (B N) L D")

    return x, y


def multi_agent_collate_fn(batch, max_num_agents, dummy_value=-1000):
    # pad by dummy values
    x_len = batch[0][0].shape[0]
    y_len = batch[0][1].shape[0]
    x = np.full((len(batch), x_len, max_num_agents, 2), dummy_value)
    y = np.full((len(batch), y_len, max_num_agents, 2), dummy_value)

    for i, (x_seq, y_seq) in enumerate(batch):
        num_agents = x_seq.shape[1]
        x[i, :, :num_agents, :] = x_seq[:, :max_num_agents, :]
        y[i, :, :num_agents, :] = y_seq[:, :max_num_agents, :]
    return torch.Tensor(x), torch.Tensor(y)


def smooth_sequence(sequence, window_size=21):
    sequence = np.array(sequence)
    smoothed_sequence = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_sequence[i] = np.mean(sequence[start:end], axis=0)
    return smoothed_sequence


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, transform=None, flatten=False, split=50):
        self.flatten = flatten
        self.transform = transform
        self.files = self.get_files(data_dir)
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.load_data(self.files[idx])
        if not self.flatten:
            data = data.reshape(data.shape[0], -1, 2)  # (seq_len, num_agents, 2)

        if self.transform:
            data = self.transform(data)

        out_data = data[: self.split]
        out_label = data[self.split :]
        return out_data, out_label

    def load_data(self, path):
        data = np.loadtxt(path, delimiter=",")
        return data

    def get_files(self, data_dir):
        files = []
        for file in data_dir.glob("*.txt"):
            files.append(file)
        return files


def random_ordering(data):
    # randomize and flatten the agent axis
    num_agents = data.shape[1]
    data = data[:, torch.randperm(num_agents), :]
    return data


def smooth(data):
    return data


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        pin_memory: bool = False,
        num_workers: int = 1,
        shuffle: bool = True,
        single_agent=False,
        smooth=True,
        split=96,
        max_num_agents=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.check_data_dir()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.single_agent = single_agent
        self.split = split
        self.max_num_agents = max_num_agents

        transforms = [torch.Tensor]
        if smooth:
            transforms.append(smooth_sequence)
        if not single_agent:
            transforms.append(random_ordering)

        self.transform = Compose(transforms)

    def check_data_dir(self):
        # do necessary checks (datafolder structure etc.) here
        pass

    def setup(self, stage: str = "both"):
        data_dir = self.data_dir
        train_data_dir = data_dir / "train"
        val_data_dir = data_dir / "val"
        test_data_dir = data_dir / "test"

        if stage == "fit" or stage == "both":
            self.trainset = TrajectoryDataset(train_data_dir, transform=self.transform, split=self.split)
            self.valset = TrajectoryDataset(val_data_dir, transform=self.transform, split=self.split)
            self.testset = TrajectoryDataset(test_data_dir, transform=self.transform, split=self.split)
        if stage == "test" or stage == "both":
            self.testset = TrajectoryDataset(test_data_dir, transform=self.transform, split=self.split)

    def train_dataloader(self):
        collate_fn = (
            single_agent_collate_fn
            if self.single_agent
            else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        )
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        collate_fn = (
            single_agent_collate_fn
            if self.single_agent
            else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        )
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        collate_fn = (
            single_agent_collate_fn
            if self.single_agent
            else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        )
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
