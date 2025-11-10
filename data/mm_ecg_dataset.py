import os
import time

import numpy as np
import scipy.io as sio
import h5py
import struct

from tqdm import tqdm

from data.base_dataset import RadarDataset, DataSpliter, rand_transform, base_transform

raw_data_root = r"/root/autodl-tmp/dataset/mmecg/finalPartialPublicData20221108"
RADAR_FILE_FORMAT = 'radar_u{user}_st{status}_s{sample}.npy'
REF_FILE_FORMAT = 'ecg_u{user}_st{status}_s{sample}.npy'
curves_data_root = r'/root/autodl-tmp/dataset/mmecg/samples'

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
    filenames = os.listdir(raw_data_root)
    sample_rate = 200
    sample_len = 5 * sample_rate
    step = 2 * sample_rate
    uid_set = {}
    status_set = {}
    sample_count_set = {}
    user_count = 0
    status_count = 0
    for filename in filenames:
        mat_map = load_mat_auto(os.path.join(raw_data_root, filename))
        data_struct = mat_map['data']
        radar_data = data_struct['RCG'][0, 0]
        ref_data = data_struct['ECG'][0, 0]
        user_id = data_struct['id'][0, 0][0, 0]
        status = data_struct['physistatus'][0, 0][0]
        if user_id not in uid_set:
            user_count += 1
            uid_set[user_id] = user_count
        if status not in status_set:
            status_count += 1
            status_set[status] = status_count
        time.sleep(0.1)
        num_samples = (len(radar_data) - sample_len) // step + 1

        sample_key = f"{user_id}_{status_count}"
        sample_count = 0
        if sample_key not in sample_count_set:
            sample_count_set[sample_key] = 0
        else:
            print(f'sample dup {sample_key}')
            sample_count = sample_count_set[sample_key] + 1

        for i in tqdm(range(num_samples)):
            sample_count = i + sample_count
            start = i * step
            end = start + sample_len
            radar_sample = radar_data[start:end]
            ref_sample = ref_data[start:end]
            # 'radar_u{user}_p{pose}_s{sample}.npy'
            radar_filename = RADAR_FILE_FORMAT.format(user=uid_set[user_id], status=status_set[status], sample=sample_count)
            ref_filename = REF_FILE_FORMAT.format(user=uid_set[user_id], status=status_set[status], sample=sample_count)
            np.save(os.path.join(curves_data_root, radar_filename), radar_sample)
            np.save(os.path.join(curves_data_root, ref_filename), ref_sample)
            time.sleep(0.1)
        sample_count_set[sample_key] = sample_count

    print(f'status {status_set}')
    print(f'user {uid_set}')


class MMECGDataset(RadarDataset):
    def __init__(self, radar_filenames, ref_filenames, transform=None, data_root=None, rand_ref=False):
        super(MMECGDataset, self).__init__(radar_filenames, ref_filenames, transform, data_root, rand_ref)
        self.user = []
        self.user_ecg_map = {}
        # 'radar_u{user}_st{status}_s{sample}.npy'
        for fn in self.ref_filenames:
            info = fn.split('_')
            u = int(info[1][1:])
            self.user.append(u)
            if u not in self.user_ecg_map:
                self.user_ecg_map[u] = [fn]
            else:
                self.user_ecg_map[u].append(fn)

    def get_ref_ecg(self, index):
        u = self.user[index]
        ecg_for_u = self.user_ecg_map[u]
        rand_idx = np.random.randint(len(ecg_for_u))
        ref_ecg = ecg_for_u[rand_idx]
        return ref_ecg

    def preprocessing_radar(self, radar_data):
        radar_data = (radar_data - np.mean(radar_data, keepdims=True)) / (
                    np.std(radar_data, keepdims=True) + 1e-9)
        radar_data = radar_data.reshape(len(radar_data), -1)
        radar_data = np.transpose(radar_data, (1, 0))
        return radar_data

    def preprocessing_ref(self, ref_data):
        ref_data = (ref_data - np.mean(ref_data, keepdims=True)) / (
                np.std(ref_data, keepdims=True) + 1e-9)
        ref_data = np.transpose(ref_data, (1, 0))
        return ref_data


class MMECGDataSpliter(DataSpliter):
    def __init__(self, data_root=curves_data_root, rand_ref=False, train_transform=base_transform,
                 val_transform=base_transform,
                 train_ratio=0.8, num_domain=4, n_fold=5):
        super().__init__(data_root, rand_ref, train_transform, val_transform, train_ratio, num_domain, n_fold)

        self.radar_data_fold.extend([[] for _ in range(max(n_fold, len(USER), len(STATUS)))])
        self.ref_data_fold.extend([[] for _ in range(max(n_fold, len(USER), len(STATUS)))])

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
                temp_radar_filenames = []
                temp_ref_filenames = []
                i = 0
                while True:
                    radar_filename = RADAR_FILE_FORMAT.format(user=u, status=s, sample=i)
                    ref_filename = REF_FILE_FORMAT.format(user=u, status=s, sample=i)
                    if not os.path.exists(os.path.join(self.data_root, radar_filename)):
                        break
                    temp_radar_filenames.append(radar_filename)
                    temp_ref_filenames.append(ref_filename)
                    i += 1
                if len(temp_radar_filenames) == 0:
                    continue
                temp_radar_filenames = np.array(temp_radar_filenames)
                temp_ref_filenames = np.array(temp_ref_filenames)
                if domain == 0:
                    rand_idx = np.arange(len(temp_radar_filenames))
                    np.random.shuffle(rand_idx)
                    rand_idx = self.split_list(rand_idx, self.num_fold)
                    for i in range(self.num_fold):
                        self.radar_data_fold[i].extend(temp_radar_filenames[rand_idx[i]])
                        self.ref_data_fold[i].extend(temp_ref_filenames[rand_idx[i]])
                else:
                    self.radar_data_fold[cur_domain_idx[domain]].extend(temp_radar_filenames)
                    self.ref_data_fold[cur_domain_idx[domain]].extend(temp_ref_filenames)

    def split_data(self, domain, train_idx=(0, 1), test_idx=(0, 1), need_val=True):
        data = super(MMECGDataSpliter, self).split_data(domain, train_idx, test_idx, need_val)
        tr_radar, tr_ref, vl_radar, vl_ref, te_radar, te_ref = data
        return MMECGDataset(tr_radar, tr_ref, self.train_transform, self.data_root, self.rand_ref), \
               MMECGDataset(vl_radar, vl_ref, self.val_transform, self.data_root, self.rand_ref), \
               MMECGDataset(te_radar, te_ref, self.val_transform, self.data_root, self.rand_ref)


if __name__ == '__main__':
    # data_path = r"I:\dataset\MMECG202211\finalPartialPublicData20221108\2.mat"
    # data = load_mat_auto(data_path)
    # print(data)
    split_raw_data()
