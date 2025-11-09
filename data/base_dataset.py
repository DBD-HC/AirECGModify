import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def to_tensor(d, ecg, ref_ecg):
    return torch.from_numpy(d).type(torch.float32), torch.from_numpy(ecg).type(torch.float32), torch.from_numpy(ref_ecg).type(torch.float32)


class RadarDataset(Dataset):
    def __init__(self, radar_filenames, ref_filenames, transform=None, data_root=None, rand_ref=False):
        if radar_filenames is None:
            return
        self.len = len(radar_filenames)
        self.radar_filenames = np.array(radar_filenames)
        self.ref_filenames = np.array(ref_filenames)
        self.data_root = data_root
        self.transform = base_transform if transform is None else transform
        self.rand_ref = rand_ref
        self.user_ecg_file = [[] for _ in range(20)]
        for fn in ref_filenames:
            inf = fn.split('_')
            user = int(inf[1][1:])
            self.user_ecg_file[user].append(fn)

    def preprocessing_radar(self, radar_data):
        return radar_data

    def preprocessing_ref(self, ref_ref):
        return ref_ref

    def get_ref_ecg(self, index):
        return self.ref_filenames[index]

    @staticmethod
    def resample(data, size=1024):
        data = data.unsqueeze(1)  # (C, 1, D)
        data = F.interpolate(data, size=size, mode='linear', align_corners=False)
        data = data.squeeze(1)  # (C, 1024)
        data = data.view(-1, 32, 32)
        return data

    def __getitem__(self, index):
        d = np.load(os.path.join(self.data_root, self.radar_filenames[index]))

        ecg_filename = self.get_ref_ecg(index)
        ecg = np.load(os.path.join(self.data_root, ecg_filename))
        d = self.preprocessing_radar(d)
        ecg = self.preprocessing_ref(ecg)

        if self.rand_ref:

            inf = ecg_filename.split('_')
            user = int(inf[1][1:])
            while True:
                ref_ecg_filename = np.random.choice(self.user_ecg_file[user])
                if ref_ecg_filename != ecg_filename:
                    break
            # ecg_u{user}_st{status}_s{sample}.npy
            ref_ecg = np.load(os.path.join(self.data_root, ref_ecg_filename))
        else:
            ref_ecg = np.load(os.path.join(self.data_root, self.ref_filenames[index]))
        ref_ecg = self.preprocessing_ref(ref_ecg)
        d, ref_ecg = self.transform(d, ref_ecg)
        d, ref_ecg, ecg = to_tensor(d, ref_ecg, ecg)
        d = self.resample(d)
        ref_ecg = self.resample(ref_ecg)
        ecg = self.resample(ecg)
        return d, ecg, ref_ecg

    def __len__(self):
        return self.len


def base_transform(radar_data, ref_data):
    return radar_data, ref_data


def rand_transform(radar_data, ref_data):
    radar_data, ref_data = base_transform(radar_data, ref_data)
    rand_idx = np.arange(radar_data.shape[0])
    np.random.shuffle(rand_idx)
    radar_data_shuffled = radar_data[rand_idx]
    return radar_data_shuffled, ref_data


def get_dataloader(data_set, shuffle, batch_size, collate_fn, sw=None, num_workers=8, drop_last=False):
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers,
                                         worker_init_fn=sw,
                                         pin_memory=True,
                                         collate_fn=collate_fn, drop_last=drop_last)
    return loader


class DataSpliter:
    def __init__(self, data_root=None, rand_ref=False, train_transform=base_transform, val_transform=base_transform,
                 train_ratio=0.8, num_domain=4, n_fold=5):
        self.data_root = data_root
        if self.data_root is None:
            return
        self.pre_domain = -1
        self.radar_filenames = []
        self.ref_filenames = []
        self.radar_data_fold = []
        self.ref_data_fold = []
        self.rand_ref = rand_ref
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_ratio = train_ratio
        self.num_domain = num_domain
        self.num_fold = n_fold

    @staticmethod
    def split_list(lst, n_parts=5):
        length = len(lst)
        if length == 0:
            return [[] for _ in range(n_parts)]
        k, m = divmod(length, n_parts)  # k=每份基础长度, m=余数
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

    def organize_data(self, domain):
        pass

    def get_dataset(self, index):
        radar_data = []
        ref_data = []
        radar_data.extend(self.radar_filenames[index])
        ref_data.extend(self.ref_filenames[index])
        return RadarDataset(radar_data, ref_data, self.val_transform, self.rand_ref)

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        self.organize_data(domain)
        train_radar_data, val_radar_data, test_radar_data = [], [], []
        train_ref_data, val_ref_data, test_ref_data = [], [], []

        for i in train_idx:
            train_radar_data.extend(self.radar_data_fold[i])
            train_ref_data.extend(self.ref_data_fold[i])
        for i in test_idx:
            test_radar_data.extend(self.radar_data_fold[i])
            test_ref_data.extend(self.ref_data_fold[i])
        if need_val:
            train_data_len = int(len(train_radar_data) * self.train_ratio)
            rand_idx = np.arange(len(train_radar_data))
            np.random.shuffle(rand_idx)
            train_radar_data = np.array(train_radar_data)
            train_ref_data = np.array(train_ref_data)
            val_radar_data.extend(train_radar_data[rand_idx[train_data_len:]])
            val_ref_data.extend(train_ref_data[rand_idx[train_data_len:]])
            train_radar_data = train_radar_data[rand_idx[:train_data_len]]
            train_ref_data = train_ref_data[rand_idx[:train_data_len]]

        return train_radar_data, train_ref_data, val_radar_data, val_ref_data, test_radar_data, test_ref_data
