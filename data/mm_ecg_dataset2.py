import os
import time

import numpy as np
import scipy.io as sio
import h5py
import struct

import torch
from tqdm import tqdm
import torch.nn.functional as F
from data.base_dataset import RadarDataset, DataSpliter, rand_transform, base_transform

raw_data_root = r"/root/autodl-tmp/dataset/mmecg/finalPartialPublicData20221108"
FILE_FORMAT = 'u{user}_st{status}_{sample}'
FILE_KEY_FORMAT = 'u{user}_st{status}'
RADAR_FILE_FORMAT = 'radar_u{user}_st{status}_s{sample}.npy'
REF_FILE_FORMAT = 'ecg_u{user}_st{status}_s{sample}.npy'
POS_FILE_FORMAT = 'pos_u{user}_st{status}_s{sample}.npy'
samples_root = r'/root/autodl-tmp/dataset/mmecg/samples'
trails_root = r'/root/autodl-tmp/dataset/mmecg/trails'
RADAR_TRAIL_FORMAT = 'radar_u{user}_st{status}_id{id}.npy'
REF_TRAIL_FORMAT = 'ecg_u{user}_st{status}_id{id}.npy'
POS_TRAIL_FORMAT = 'pos_u{user}_st{status}_id{id}.npy'
USER = [i + 1 for i in range(11)]
STATUS = [i + 1 for i in range(4)]


def load_mat_auto(path):
    """
    自动检测 MATLAB .mat 文件版本并读取
    - v7 及以下：使用 scipy.io.loadmat
    - v7.3 (HDF5 格式)：使用 h5py
    """
    # 先读取文件头前 128 个字节，里面包含版本信息
    with open(path, 'rb') as f:
        header = f.read(128)

    # MATLAB v7.3 的文件头里包含 "MATLAB 7.3 MAT-file"
    if b'MATLAB 7.3' in header:
        print("[INFO] Detected MATLAB v7.3 (HDF5) format, using h5py...")
        data = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                data[k] = f[k][()]
        return data
    else:
        print("[INFO] Detected MATLAB v7 or lower format, using scipy.io.loadmat...")
        return sio.loadmat(path)


def split_raw_data():
    # filenames = os.listdir(raw_data_root)
    sample_rate = 200
    sample_len = 5 * sample_rate
    step = 1 * sample_rate
    uid_set = {}
    status_set = {}
    sample_count_set = {}
    user_count = 0
    status_count = 0
    file_sample_info = {}
    for f_id in tqdm(range(1, 92)):
        mat_map = load_mat_auto(os.path.join(raw_data_root, f'{f_id}.mat'))
        data_struct = mat_map['data']
        radar_data = data_struct['RCG'][0, 0]
        ref_data = data_struct['ECG'][0, 0]
        user_id = data_struct['id'][0, 0][0, 0]
        status = data_struct['physistatus'][0, 0][0]
        pos = data_struct['posXYZ'][0, 0]
        if user_id not in uid_set:
            user_count += 1
            uid_set[user_id] = user_count
        if status not in status_set:
            status_count += 1
            status_set[status] = status_count
        time.sleep(0.1)

        radar_trail_name = RADAR_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=f_id)
        ref_trail_name = REF_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=f_id)
        pos_trail_name = POS_TRAIL_FORMAT.format(user=uid_set[user_id], status=status_set[status], id=f_id)
        np.save(os.path.join(trails_root, radar_trail_name), radar_data)
        np.save(os.path.join(trails_root, ref_trail_name), ref_data)
        np.save(os.path.join(trails_root, pos_trail_name), pos)

        num_samples = (len(radar_data) - sample_len) // step + 1

        sample_key = f"u{uid_set[user_id]}_st{status_set[status]}"
        sample_count = 0
        if sample_key not in sample_count_set:
            sample_count_set[sample_key] = 0
            file_sample_info[sample_key] = []
        else:
            print(f'sample dup {sample_key}')
            sample_count = sample_count_set[sample_key]

        for i in tqdm(range(num_samples)):
            sample_index = i + sample_count
            start = i * step
            end = start + sample_len
            radar_sample = radar_data[start:end]
            ref_sample = ref_data[start:end]
            pos_sample = pos
            # 'radar_u{user}_p{pose}_s{sample}.npy'

            # radar_filename = RADAR_FILE_FORMAT.format(user=uid_set[user_id], status=status_set[status], sample=sample_index)
            # ref_filename = REF_FILE_FORMAT.format(user=uid_set[user_id], status=status_set[status], sample=sample_index)
            # pos_filename = POS_FILE_FORMAT.format(user=uid_set[user_id], status=status_set[status], sample=sample_index)
            file_sample_info[sample_key].append({
                'radar_fn': radar_trail_name,
                'pos_fn': pos_trail_name,
                'ecg_fn': ref_trail_name,
                's': start,
                't': end
            })
            # np.save(os.path.join(samples_root, radar_filename), radar_sample)
            # np.save(os.path.join(samples_root, ref_filename), ref_sample)
            # np.save(os.path.join(samples_root, pos_filename), pos_sample)
            # time.sleep(0.1)
        sample_count_set[sample_key] = sample_count + num_samples
    np.save(os.path.join(samples_root, 'radar_ecg_sample_info.npy'), file_sample_info)
    print(f'status {status_set}')
    print(f'user {uid_set}')


def norm(data):
    max_data = np.max(data, axis=0, keepdims=True)
    min_data = np.min(data, axis=0, keepdims=True)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    data = data * 2 - 1
    # data = (data - np.mean(data, axis=0, keepdims=True)) / (
    #         np.std(data, axis=0, keepdims=True) + 1e-9)
    return data


def preprocessing_radar(radar_data):
    data = (radar_data - np.mean(radar_data, keepdims=True)) / (
            np.std(radar_data, keepdims=True) + 1e-9)
    # radar_data = self.norm(radar_data)
    # return self.norm(radar_data)
    return data


def preprocessing_ref(ref):
    ref = norm(ref)
    return ref


class MMECGDataset(RadarDataset):
    def __init__(self, filenames, transform=None, data_root=None, rand_ref=False, sample2file_info=None,
                 need_align=False):
        super(MMECGDataset, self).__init__(filenames, transform, data_root, rand_ref, sample2file_info, need_align)
        print(f'need align {need_align}')

    @staticmethod
    def up_sample(data, size=1024):
        data = data.unsqueeze(1)  # (C, 1, D)
        data = F.interpolate(data, size=size, mode='linear', align_corners=False)
        data = data.squeeze(1)  # (C, 1024)
        data = data.view(-1, 32, 32)
        return data

    def get_inf(self, index=None, filename=None):
        if filename is None:
            filename = self.filenames[index]
        info = filename.split('_')
        filename_key = info[0] + '_' + info[1]
        sample2file = self.sample2file_info[filename_key][int(info[-1])]
        start, to = sample2file['s'], sample2file['t']
        radar_filename = sample2file['radar_fn']
        pos_filename = sample2file['pos_fn']
        if self.need_align:
            ecg_filename = sample2file['ecg_align_fn']
        else:
            ecg_filename = sample2file['ecg_fn']
        return radar_filename, pos_filename, ecg_filename, start, to

    @staticmethod
    def random_fill_channels(x, total_channels=100):
        C, D = x.shape
        # 1) 随机选择放置位置
        idx = np.random.permutation(total_channels)[:C]

        # 2) 初始化噪声数组
        out = np.random.randn(total_channels, D)

        # 3) 覆盖真实数据
        out[idx, :] = x

        return out

    def __getitem__(self, index):
        radar_filename, pos_filename, ecg_filename, start, to = self.get_inf(index)
        d = np.load(os.path.join(self.data_root, radar_filename))[start:to]
        ref = np.load(os.path.join(self.data_root, ecg_filename))[start:to]
        pos = np.load(os.path.join(self.data_root, pos_filename))
        d, ref = preprocessing_radar(d), preprocessing_ref(ref)
        d = np.transpose(d, (1, 0))
        ref = np.transpose(ref, (1, 0))
        d, d_shuffled, ref = self.transform(d, d, ref)
        d = torch.from_numpy(d).type(torch.float32)
        pos = torch.from_numpy(pos).type(torch.float32)
        ref = torch.from_numpy(ref).type(torch.float32)
        d = self.up_sample(d)
        ref = self.up_sample(ref)
        return d, ref, torch.tensor(0.0)


class MMECGDataSpliter(DataSpliter):
    def __init__(self, data_root=samples_root, trail_root=trails_root, rand_ref=False, train_transform=rand_transform,
                 val_transform=base_transform, train_ratio=0.8, num_domain=4, n_fold=5, need_align=False):
        super().__init__(data_root, trail_root, rand_ref, train_transform, val_transform, train_ratio, num_domain,
                         n_fold, need_align=need_align)

        self.sample_fold.extend([[] for _ in range(max(n_fold, len(USER), len(STATUS)))])

    def organize_data(self, domain):
        if self.pre_domain == domain:
            return
        else:
            self.pre_domain = domain
        cur_domain_idx = [0 for _ in range(self.num_domain)]

        for u_id, u in enumerate(USER):
            cur_domain_idx[1] = u_id
            for s_id, s in enumerate(STATUS):
                cur_domain_idx[2] = s_id
                temp_filenames = []
                i = 0
                while True:
                    filename = FILE_FORMAT.format(user=u, status=s, sample=i)
                    sample_key = FILE_KEY_FORMAT.format(user=u, status=s)
                    if sample_key not in self.sample2file_info or len(self.sample2file_info[sample_key]) <= i:
                        break
                    temp_filenames.append(filename)
                    i += 1
                if len(temp_filenames) == 0:
                    continue
                temp_filenames = np.array(temp_filenames)
                if domain == 0:
                    rand_idx = np.arange(len(temp_filenames))
                    np.random.shuffle(rand_idx)
                    rand_idx = self.split_list(rand_idx, self.num_fold)
                    for i in range(self.num_fold):
                        self.sample_fold[i].extend(temp_filenames[rand_idx[i]])
                else:
                    self.sample_fold[cur_domain_idx[domain]].extend(temp_filenames)

    def get_trails(self, domain, index):
        filename_list = os.listdir(self.trail_root)
        num_files = len(filename_list) // 3
        cur_domain_idx = [0 for _ in range(self.num_domain)]
        radar_trails = []
        ref_trails = []
        pos_trails = []
        for u_id, u in enumerate(USER):
            cur_domain_idx[1] = u_id
            for s_id, s in enumerate(STATUS):
                cur_domain_idx[2] = s_id
                if cur_domain_idx[domain] == index:
                    trail_id = 0
                    while trail_id < num_files:
                        radar_filename = RADAR_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        ref_filename = REF_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        pos_filename = POS_TRAIL_FORMAT.format(user=u, status=s, id=trail_id)
                        trail_id += 1
                        if radar_filename not in filename_list:
                            continue
                        radar_trails.append(np.load(os.path.join(self.trail_root, radar_filename)))
                        ref_trails.append(np.load(os.path.join(self.trail_root, ref_filename)))
                        pos_trails.append(np.load(os.path.join(self.trail_root, pos_filename)))

        radar_trails = np.transpose(np.array(radar_trails), (0, 2, 1))
        ref_trails = np.transpose(np.array(ref_trails), (0, 2, 1))
        pos_trails = np.array(pos_trails)
        return radar_trails, ref_trails, pos_trails

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        data = super(MMECGDataSpliter, self).split_data(domain, train_idx, test_idx, need_val)
        tr, vl, te = data
        return MMECGDataset(tr, self.train_transform, self.trail_root, self.rand_ref, self.sample2file_info,
                            self.need_align), \
            MMECGDataset(vl, self.val_transform, self.trail_root, False, self.sample2file_info, self.need_align), \
            MMECGDataset(te, self.val_transform, self.trail_root, False, self.sample2file_info, self.need_align)


if __name__ == '__main__':
    # data_path = r"I:\dataset\MMECG202211\atrial_fibrillation\A_0001_1.mat"
    # data = load_mat_auto(data_path)
    # print(data)
    split_raw_data()
