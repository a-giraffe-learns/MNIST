
# region packages

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from sklearn.model_selection import train_test_split
from mypyutils.utils_dl import *
# endregion


# -----------------

def mnist_dataloaders(data_path, transforms_train, transforms_test, batch_size, vectorize=False):
    dataset_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    dataset_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if vectorize:
        n_samples_train = dataset_train.data.shape[0]
        n_samples_test = dataset_test.data.shape[0]
        dataset_train = ImageDataset(transforms_train(dataset_train.data.view(n_samples_train, -1)), dataset_train.targets,
                                     None, xpath=False)
        samples_test = ImageDataset(transforms_test(dataset_test.data.view(n_samples_test, -1)), dataset_test.targets,
                                    None, xpath=False)

    else:
        dataset_train = ImageDataset(transforms_train(dataset_train.data[:, None, :, :]), dataset_train.targets,
                                     None, xpath=False)
        samples_test = ImageDataset(transforms_test(dataset_test.data[:, None, :, :]), dataset_test.targets,
                                    None, xpath=False)


    samples_train, samples_val = train_test_split(dataset_train, stratify=dataset_train.image_labels, test_size=0.2)

    dataloader_train = DataLoader(samples_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(samples_val, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(samples_test, batch_size=batch_size, shuffle=False, drop_last=False)

    return dataloader_train, dataloader_val, dataloader_test
