import os.path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset



def to_tensor(d, ref):
    return torch.from_numpy(d).type(torch.float32), torch.from_numpy(ref).type(torch.float32)


class RadarDataset(Dataset):
    def __init__(self, filenames, transform=None, data_root=None, rand_ref=False, sample2file_info=None, n_select=0.2,
                 user2samples=None, need_align=False):
        if filenames is None:
            return
        self.len = len(filenames)
        self.filenames = np.array(filenames)
        self.data_root = data_root
        self.transform = base_transform if transform is None else transform
        self.rand_ref = rand_ref
        self.sample2file_info = sample2file_info
        self.adjust_filenames = np.random.choice(self.filenames, int(n_select * self.len))
        self.user2samples=user2samples
        self.need_align = need_align

    def preprocessing_radar(self, radar_data):
        return radar_data

    def preprocessing_ref(self, ref):
        return ref

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.len


def base_transform(radar_data, rand_radar_data, ref_data):
    return radar_data, rand_radar_data, ref_data


def rand_scale(radar_data, scale_range=(0.5, 2.), noise_std=1):
    """
    radar_data: np.array, shape [C, T]
    scale_range: 缩放比例范围，例如(0.7, 1.3)
    noise_std: 高斯噪声标准差
    """
    C, T = radar_data.shape
    scale = np.random.uniform(*scale_range)
    new_len = max(1, int(T * scale))

    # 缩放每个通道
    scaled = np.zeros((C, new_len))
    for c in range(C):
        scaled[c] = zoom(radar_data[c], new_len / T, order=1)  # 一维线性插值

    # 根据缩放结果填充或裁剪
    if new_len < T:
        # 缩短：填充高斯噪声
        noise = np.random.normal(0, 1, (C, T - new_len))
        radar_scaled = np.concatenate([noise, scaled], axis=-1)
    elif new_len > T:
        # 放大：裁剪回原长度
        radar_scaled = scaled[:, :T]
    else:
        radar_scaled = scaled

    return radar_scaled


def random_time_shift(x, shift_range=(0, 200)):
    """
    x: (C, T)
    max_shift_ratio: 最大平移比例（相对T）
    """
    C, T = x.shape
    shift = np.random.randint(shift_range[0], shift_range[1])

    if shift == 0:
        return x

    shifted = np.zeros_like(x)

    if shift > 0:
        # 向右移
        shifted[:, shift:] = x[:, :-shift]
    else:
        # 向左移
        shifted[:, :shift] = x[:, -shift:]

    return shifted


def rand_transform(radar_data, rand_radar_data, ref_data, zero_ratio=0.2):
    radar_data, rand_radar_data, ref_data = base_transform(radar_data, rand_radar_data, ref_data)
    rand_idx = np.arange(radar_data.shape[0])
    np.random.shuffle(rand_idx)
    radar_data = radar_data[rand_idx]
    np.random.shuffle(rand_idx)
    rand_radar_data = rand_radar_data[rand_idx]
    C, T = radar_data.shape

    # if np.random.uniform(0, 1) < 0.5:
    #     radar_data = random_time_shift(radar_data)
    #     # ref_data = random_time_shift(ref_data)

    if np.random.uniform(0, 1) < 0.:
        zero_len = np.random.randint(int(T * zero_ratio))

        # 随机选一个起点
        start_idx = np.random.randint(0, T - zero_len + 1)
        end_idx = start_idx + zero_len

        # 将这段时间的雷达信号置零
        rand_radar_data[:, start_idx:end_idx] = np.random.normal(0, 0.01, (C, zero_len))
    # rand_radar_data = rand_scale(rand_radar_data)

    return radar_data, rand_radar_data, ref_data


def get_dataloader(data_set, shuffle, batch_size, collate_fn, sw=None, num_workers=8, drop_last=False):
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers,
                                         worker_init_fn=sw,
                                         pin_memory=True,
                                         collate_fn=collate_fn, drop_last=drop_last)
    return loader


class DataSpliter:
    def __init__(self, data_root=None, trail_root=None, rand_ref=False, train_transform=rand_transform, val_transform=base_transform,
                 train_ratio=0.8, num_domain=4, n_fold=5, need_align=False):
        self.data_root = data_root
        if self.data_root is None:
            return
        self.trail_root = trail_root
        self.pre_domain = -1
        self.sample_fold = []
        self.rand_ref = rand_ref
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_ratio = train_ratio
        self.num_domain = num_domain
        self.num_fold = n_fold
        self.sample2file_info = np.load(os.path.join(self.data_root, 'radar_ecg_sample_info.npy'),
                                        allow_pickle=True).item()
        self.need_align = need_align

    @staticmethod
    def split_list(lst, n_parts=5):
        length = len(lst)
        if length == 0:
            return [[] for _ in range(n_parts)]
        k, m = divmod(length, n_parts)  # k=每份基础长度, m=余数
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

    def organize_data(self, domain):
        pass

    def get_trails(self, domain, index):
        pass

    def get_dataset(self, index):
        samples = []
        samples.extend(self.sample_fold[index])
        return RadarDataset(samples, self.val_transform, self.trail_root, self.rand_ref, self.sample2file_info)

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        self.organize_data(domain)
        train_data, val_data, test_data = [], [], []

        for i in train_idx:
            train_data.extend(self.sample_fold[i])
        for i in test_idx:
            test_data.extend(self.sample_fold[i])
        if need_val:
            train_data_len = int(len(train_data) * self.train_ratio)
            rand_idx = np.arange(len(train_data))
            np.random.shuffle(rand_idx)
            train_data = np.array(train_data)
            val_data.extend(train_data[rand_idx[train_data_len:]])
            train_data = train_data[rand_idx[:train_data_len]]

        return train_data, val_data, test_data
