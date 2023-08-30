import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader


def load_transformed_data(img_size, batch_size):
    ext_img_size = img_size + img_size // 4
    data_transforms = [
        T.Resize((ext_img_size, ext_img_size)),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Scales data into [0,1]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    data_transform = T.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(
        root=".",
        download=True,
        transform=data_transform
    )
    test = torchvision.datasets.StanfordCars(
        root=".",
        download=True,
        transform=data_transform,
        split='test'
    )
    data = torch.utils.data.ConcatDataset([train, test])
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader
