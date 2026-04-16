import numpy as np
import torch
from tqdm import tqdm

from data.mm_ecg_dataset2 import norm
import torch.nn.functional as F

class LRWrapper:
    def __init__(self, model, diffusion, win_size=800, fs=200, overlap=1, need_history=False):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.need_history = need_history
        self.history_ecg = None
        self.win_size = win_size
        self.overlap_len = overlap * fs
        self.eps = 1e-8

    @staticmethod
    def up_sample(data, size=1024):
        data = F.interpolate(data, size=size, mode='linear', align_corners=False)
        data = data.view(..., 32, 32)
        return data

    @staticmethod
    def down_sample(data, size=1000):
        data = data.view(..., 1024)
        data = F.interpolate(data, size=size, mode='linear', align_corners=False)
        return data

    @staticmethod
    def norm_radar(data):
        data = (data - torch.mean(data, dim=(-2, -1), keepdim=True)) / (
                torch.std(data, dim=(-2, -1), keepdim=True) + 1e-9)
        return data

    @staticmethod
    def norm_ecg(data):
        max_data = torch.max(data, dim=-1, keepdim=True)[0]
        min_data = torch.min(data, dim=-1, keepdim=True)[0]
        data = (data - min_data) / (max_data - min_data + 1e-6)
        data = data * 2 - 1
        return data


    def radar2ecg(self, radar_data, pos_data=None, drop_last=True):
        B, C, T = radar_data.shape
        step = T // self.win_size
        result = []
        att_radar_list = []
        with torch.no_grad():
            for i in tqdm(range(step)):
                if drop_last and i == step - 1:
                    break
                # 0 - 800 100-700 600 - 1400 700-1300 1200 - 2000 1300-1900
                overlap_len = self.overlap_len
                s, t = i * self.win_size, (i + 1) * self.win_size + overlap_len
                temp_radar = radar_data[:, :, s:t]
                temp_radar = self.up_sample(temp_radar)
                temp_radar = self.norm_radar(temp_radar)
                param = {'radar': temp_radar, 'is_train': False}
                if pos_data is not None:
                    temp_pos = pos_data
                    param['position'] = temp_pos
                if self.need_history:
                    param['history_ecg'] = self.history_ecg

                z = torch.randn(temp_radar.size(0), 1, 32, 32, device=temp_radar.device)
                # Setup guidance:

                model_kwargs = dict(y1=temp_radar)
                # Sample images:
                samples = self.diffusion.p_sample_loop(
                    self.model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=temp_radar.device
                )
                samples = samples.reshape(z.size(0), 1, 1024)
                samples = self.down_sample(samples)
                # att_radar = self.model.result_temp['val_att_radar']
                result.append(self.norm_ecg(samples[:, :, overlap_len // 2: -overlap_len // 2]))
                # att_radar_list.append(self.norm(att_radar[:, :, overlap_len // 2: -overlap_len // 2]))
                self.history_ecg = samples
        return torch.cat(result, dim=-1), None # torch.cat(att_radar_list, dim=-1)
