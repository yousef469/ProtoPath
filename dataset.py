import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from config import CANCER_TO_IDX

IMG_SIZE = 224


class TCGADataset(Dataset):
    def __init__(self, split="train", transform=None, max_samples=None):
        self.dataset = load_dataset(
            "dakomura/tcga-ut",
            "internal",
            split=split,
            streaming=False,
        )
        self.labels = CANCER_TO_IDX
        self.class_names = list(CANCER_TO_IDX.keys())

        if max_samples is not None:
            indices = list(range(min(max_samples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)

        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["jpg"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        label = self.labels[sample["json"]["label"]]
        return image, label


class TCGADataModule:
    def __init__(self, batch_size=128, num_workers=4, max_samples=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self._init_datasets()

    def _init_datasets(self):
        self.train_dataset = TCGADataset(
            split="train",
            max_samples=self.max_samples,
        )
        self.val_dataset = TCGADataset(
            split="valid",
            max_samples=self.max_samples // 4 if self.max_samples else None,
        )
        self.test_dataset = TCGADataset(
            split="test",
            max_samples=self.max_samples // 4 if self.max_samples else None,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
