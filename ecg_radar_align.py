import torch


def batch_max_pearson_corr(x: torch.Tensor, y: torch.Tensor, dim=-1, max_lag: int = None, eps: float = 1e-8):
    """
    批量计算每条时间序列的最大皮尔逊相关系数及对应 lag。

    参数：
        x, y      : [batch, T] Tensor
        max_lag   : int, 最大考虑的正负时延
        eps       : float, 防止除零

    返回：
        max_pcc   : [batch] Tensor, 每条序列的最大 PCC
        best_lag  : [batch] Tensor, 每条序列对应 lag
    """
    if x.shape != y.shape:
        raise ValueError("x 和 y 必须形状相同")
    batch_size, C, T = x.shape
    if max_lag is None:
        max_lag = T - 1

    max_pcc = torch.full((batch_size,), -2.0, device=x.device, dtype=x.dtype)
    best_lag = torch.zeros((batch_size,), device=x.device, dtype=torch.int)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[..., :T + lag]
            ys = y[..., -lag:]
        elif lag > 0:
            xs = x[..., lag:]
            ys = y[..., :T - lag]
        else:
            xs, ys = x, y

        xm = xs - xs.mean(dim=dim, keepdim=True)
        ym = ys - ys.mean(dim=dim, keepdim=True)

        r_num = torch.sum(xm * ym, dim=dim, keepdim=True)
        r_den = torch.sqrt(torch.sum(xm ** 2, dim=dim, keepdim=True) * torch.sum(ym ** 2, dim=dim, keepdim=True)) + eps
        pcc = r_num / r_den

        mask = pcc[:, 0, 0] > max_pcc
        max_pcc = torch.where(mask, pcc[:, 0, 0], max_pcc)
        best_lag = torch.where(mask, torch.full_like(best_lag, lag), best_lag)

    return max_pcc, best_lag


def align_ecg_radar(ecg, radar, pre_sample_lag):
    aligned_ecg = torch.zeros_like(ecg)
    aligned_radar = torch.zeros_like(radar)
    B, _, T = ecg.shape
    for i, lag in enumerate(pre_sample_lag):
        if lag > 0:
            aligned_ecg[i, :, :-lag] = ecg[i, :, lag:]
            aligned_radar[i, :, :T - lag] = radar[i, :, :T - lag]
        elif lag < 0:
            aligned_ecg[i, :, :T + lag] = ecg[i, :, :T + lag]
            aligned_radar[i, :, :lag] = radar[i, :, -lag:]
        else:
            aligned_ecg[i] = aligned_ecg[i]

    return aligned_ecg, aligned_radar
