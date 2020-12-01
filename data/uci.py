import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class UCI(Dataset):
    def __init__(self, data_type, root, miss_rate, train):
        super(UCI, self).__init__()
        self.output_size = None
        self.load_dataset(data_type, root, train)
        self.miss_rate = torch.tensor(miss_rate)
        self.mask_sampler = torch.distributions.Uniform(low=0, high=1)
        self.input_size = self.x.shape[1]
        self.masks = None
        self._init_masks()

    def _init_masks(self):
        torch.manual_seed(1)
        all_binary = []
        for i in range(len(self.x)):
            uniform = self.mask_sampler.sample([self.input_size])
            binary_mat = (uniform < (1-self.miss_rate)).float().view(1, self.input_size)
            all_binary.append(binary_mat)
        self.masks = torch.cat(all_binary, dim=0)

    def one_hot(self, arr):
        temp = torch.zeros((arr.shape[0], arr.max() + 1))
        temp[torch.arange(arr.shape[0]), arr] = 1
        return temp

    def load_dataset(self, data_type, base_path, train):
        file_name = os.path.join(base_path, 'uci-data/', data_type + '.txt')
        data = pd.read_csv(file_name)
        data = data.to_numpy()
        #data = np.genfromtxt(file_name, delimiter=",")
        if data_type == "spam":
            target = data[:, -1]
            data = data[:, :-1].astype(np.float)
        elif data_type == "letter":
            target = data[:, 0]
            data = data[:, 1:].astype(np.float)
        else:
            raise NotImplementedError(data_type)
        target = target.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(target)
        target = encoder.transform(target)

        #N = data.shape[0]
        #np.random.seed(1)
        #train_pick = np.random.choice(N, int(0.8 * N), replace=False)
        #train_idx = np.zeros((N), dtype=bool)
        #train_idx[train_pick] = True
        #if train:
        #    idx = train_idx
        #else:
        #    idx = ~train_idx

        #self.x = torch.tensor(data[idx]).float()
        #self.y = torch.tensor(target[idx])
        #self.output_size = self.y.shape[1]

        self.x = torch.tensor(data).float()
        self.y = torch.tensor(target)
        self.output_size = self.y.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_batch = self.x[idx]
        mask_batch = self.masks[idx]
        x_batch_miss = x_batch.clone()
        x_batch_miss[mask_batch == 0] = np.nan
        y_batch = self.y[idx]
        return x_batch, x_batch_miss, y_batch, mask_batch


def get_uci(args):
    base_path = "./data"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.batch_size if args.batch_size else 512
    num_workers = args.workers if args.workers else 4

    train_data = UCI(args.data_type, root=base_path, miss_rate=args.miss_rate, train=True)
    test_data = UCI(args.data_type, root=base_path, miss_rate=args.miss_rate, train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                             shuffle=False, num_workers=num_workers)
    print(f"Generating min/max for {args.data_type}")
    all_imgs= []
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

