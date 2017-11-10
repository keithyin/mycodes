from torch.utils.data import DataLoader
from torchvision import transforms
from .CustomDataset import CustomDataset
from .CustomDataset import CustomDatasetSeq
from .CustomDataset import CustomDatasetVal


def get_dataset(root, txt_file, train=True):
    # transform
    transform = transforms.Compose([
        transforms.Scale(size=(299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(root=root, txt_file=txt_file, train=train, transform=transform)
    return dataset


def get_dataset_seq(root, txt_file, seq_len, train=True):
    # transform
    transform = transforms.Compose([
        transforms.Scale(size=(299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDatasetSeq(root=root, txt_file=txt_file,
                               train=train, transform=transform,
                               seq_len=seq_len)
    return dataset


def get_dataset_val(root, txt_file, seq_len):
    transform = transforms.Compose([
        transforms.Scale(size=(299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    queries_dataset = CustomDatasetVal(root=root, txt_file=txt_file,
                                       transform=transform,
                                       seq_len=seq_len, is_query=True)
    gallery_dataset = CustomDatasetVal(root=root, txt_file=txt_file,
                                       transform=transform,
                                       seq_len=seq_len, is_query=False)
    return queries_dataset, gallery_dataset


def get_loader(dataset, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=drop_last)

    return loader
