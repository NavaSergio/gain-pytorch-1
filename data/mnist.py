from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


class MNISTInvase(MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTInvase, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.view(-1)
        # Below -1 is due to G being undefined
        return img, target, -1


def one_hot(arr):
    temp = torch.zeros((arr.shape[0], arr.max() + 1))
    temp[torch.arange(arr.shape[0]), arr] = 1
    return temp


def get_mnist(args):
    base_path = "./data-dir"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.batch_size if args.batch_size else 512
    num_workers = args.workers if args.workers else 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_data = MNISTInvase(base_path, train=True, download=True,
                             transform=transform)
    train_data.means = (0.1307,)
    train_data.stds = (0.3081,)
    train_data.bounds = [0, 1]
    train_data.input_size = 784
    train_data.output_size = 10

    train_data.targets = one_hot(train_data.targets)

    test_data = MNISTInvase(base_path, train=False,
                            transform=transform)
    test_data.targets = one_hot(test_data.targets)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class MNIST_GAIN(MNIST):
    def __init__(self, miss_rate, *args, **kwargs):
        super(MNIST_GAIN, self).__init__(*args, **kwargs)
        self.miss_rate = torch.tensor(miss_rate)
        self.mask_sampler = torch.distributions.Uniform(low=0, high=1)
        self.input_size = 784
        self.output_size = 10
        self.masks = None
        self._init_masks()
        self.targets = torch.tensor(one_hot(self.targets.cpu().numpy()))

    def _init_masks(self):
        torch.manual_seed(1)
        all_binary = []
        for i in range(len(self.data)):
            uniform = self.mask_sampler.sample([self.input_size])
            binary_mat = (uniform < (1-self.miss_rate)).float().view(1, self.input_size)
            all_binary.append(binary_mat)
        self.masks = torch.cat(all_binary, dim=0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        mask = self.masks[index]
        img = img.view(-1)
        img_miss = img.clone()
        img_miss[mask == 0] = 0.
        return img, img_miss, target, mask


def get_mnist_gain(args):
    base_path = "./data-dir"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.batch_size if args.batch_size else 512
    num_workers = args.workers if args.workers else 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_data = MNIST_GAIN(root=base_path, miss_rate=args.miss_rate, train=True,
                            download=True, transform=transform)

    test_data = MNIST_GAIN(root=base_path, miss_rate=args.miss_rate, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                                              shuffle=False, num_workers=num_workers)
    print(f"Generating min/max for {args.data_type}")
    all_imgs = []
    for img, _, _, _ in tqdm(train_loader):
        all_imgs.append(img.cpu().numpy())

    imgs = np.concatenate(all_imgs, axis=0)
    min_tensor = np.nanmin(imgs, axis=0)
    imgs = imgs - min_tensor
    max_tensor = np.nanmax(imgs, axis=0)
    min_tensor = torch.tensor(min_tensor)
    max_tensor = torch.tensor(max_tensor)

    transform_params = {"min": min_tensor, "max": max_tensor}
    return train_loader, test_loader, transform_params
