"""
Training script for AirECG using PyTorch DDP.
"""
import numpy as np
import torch
import visdom


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm

from models import AirECG_model
from diffusion import create_diffusion

import matplotlib.pyplot as plt
from data.mm_ecg_dataset import MMECGDataSpliter

plt.rcParams['figure.figsize'] = 23, 15


def extract_model(model_name):
    """
    load a pre-trained AirECG model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find AirECG checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    return checkpoint


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # 检查是否已经初始化了进程组
    try:
        if dist.is_initialized():
            is_rank_zero = (dist.get_rank() == 0)
        else:
            # 如果没有初始化进程组，假设是单GPU模式，rank为0
            is_rank_zero = True
    except:
        # 如果检查过程中出现异常，也假设是单GPU模式
        is_rank_zero = True

    if is_rank_zero:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_dataloader(data_set, shuffle, batch_size, collate_fn=None):
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=8,
                                         # worker_init_fn=seed_worker,
                                         pin_memory=True,
                                         collate_fn=collate_fn)
    return loader


def sample_images(ref, y, x, samples, batchIdx, resultPath, isVal=True):
    """
    Saves generated signals from the validation set
    """

    current_img_dir = resultPath  # + '/%s_Val.png' % (batchIdx)

    fig, axes = plt.subplots(y.shape[0], 4)
    axes[0][0].set_title('Ref')
    axes[0][1].set_title('mmWave')
    axes[0][2].set_title('ECG GroundTruth')
    axes[0][3].set_title('Generated ECG')

    for idx, signal in enumerate(y):
        axes[idx][0].plot(ref[idx], color='c')
        axes[idx][1].plot(y[idx], color='c')
        axes[idx][2].plot(x[idx], color='m')
        axes[idx][3].plot(samples[idx], color='y')

    fig.canvas.draw()
    fig.savefig(current_img_dir)
    plt.close(fig)


from torch.utils.data import DataLoader, Dataset


def DataLoader_example(bs, personID=0, fold_idx=0, splitChannel=False, mmWaveNorm=True):
    # Load your data here
    train_mmwave = torch.randn(96, 8, 1024)
    train_ecg = torch.randn(96, 1024)

    test_mmwave = torch.randn(96, 8, 1024)
    test_ecg = torch.randn(96, 1024)

    ref_ecg = torch.randn(96, 1024)

    def patchingmmWave(inputSignal):
        B = inputSignal.shape[0]
        C = inputSignal.shape[1]
        inputSignal = inputSignal.reshape(B, C, 32, 32)
        return inputSignal

    def patchingECG(inputSignal):
        B = inputSignal.shape[0]
        inputSignal = inputSignal.reshape(B, 32, 32)
        inputSignal = inputSignal.unsqueeze(1)
        return inputSignal

    train_mmwave = patchingmmWave(train_mmwave)
    train_ecg = patchingECG(train_ecg)

    test_mmwave = patchingmmWave(test_mmwave)
    test_ecg = patchingECG(test_ecg)

    ref_ecg = patchingECG(ref_ecg)

    class DataSet(Dataset):
        def __init__(self, x: torch.Tensor, y: torch.Tensor, y_ref: torch.Tensor):
            self.x = x
            self.y = y
            self.y_ref = y_ref

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            return [self.x[index], self.y[index], self.y_ref[index]]

    trainECGDataSet = DataSet(train_mmwave.float(), train_ecg.float(), ref_ecg.float())
    testECGDataSet = DataSet(test_mmwave.float(), test_ecg.float(), ref_ecg.float())

    train_ecg_loader = DataLoader(trainECGDataSet, batch_size=bs, shuffle=False)
    test_ecg_loader = DataLoader(testECGDataSet, batch_size=bs, shuffle=False)

    return train_ecg_loader, test_ecg_loader


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


def visualize_gen_curves(gen_curves, ref_curves, viz, win='curves', title='Generated vs Reference'):
    """
    使用 visdom 可视化生成曲线和参考曲线

    参数:
        gen_curves : numpy.ndarray 或 torch.Tensor, shape [N] 或 [batch, N]
        ref_curves : numpy.ndarray 或 torch.Tensor, 同 shape
        viz        : visdom.Visdom 实例
        win        : str, 窗口名字
        title      : str, 窗口标题
    """
    if viz is None:
        return

    # 转 numpy
    if hasattr(gen_curves, "detach"):
        gen_curves = gen_curves.detach().cpu().numpy()
    if hasattr(ref_curves, "detach"):
        ref_curves = ref_curves.detach().cpu().numpy()

    # 如果是 batch，取第一条或平均
    if gen_curves.ndim > 1:
        gen_curves = gen_curves.mean(axis=0)
    if ref_curves.ndim > 1:
        ref_curves = ref_curves.mean(axis=0)

    x = np.arange(len(gen_curves))

    viz.line(
        X=np.column_stack([x, x]),
        Y=np.column_stack([gen_curves, ref_curves]),
        win=win,
        opts=dict(
            title=title,
            xlabel='Time step',
            ylabel='Value',
            legend=['Generated', 'Reference']
        )
    )

def main(args, train_dataloader, test_dataloader):
    """
    Trains AirECG model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    if torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")
        use_ddp = True
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        print("单GPU模式，跳过分布式初始化")
        use_ddp = False
        world_size = 1
        rank = 0

    # 修改断言，只在DDP模式下检查batch size整除性
    if use_ddp:
        assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    else:
        # 单GPU模式下，global_batch_size就是实际batch size
        pass

    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = 'AirECG'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    # Create model:
    latent_size = 32
    model = AirECG_model(
        input_size=latent_size,
        mm_channels=args.mmWave_channels,
    )
    ckpt_path = args.ckpt
    if ckpt_path != None:
        state_dict = extract_model(ckpt_path)
        model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if use_ddp:
        model = DDP(model.to(device), device_ids=[rank])
    else:
        model = model.to(device)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    in_channels = args.mmWave_channels

    if use_ddp:
        # DDP模式下，通过 model.module 访问原始模型
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        # 单GPU模式下，直接使用 model
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    viz = visdom.Visdom(env='cross domain', port=6006)
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        model.train()
        for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(train_dataloader)):
            n = ecg.shape[0]
            ecg = ecg.to(device)
            mmwave = mmwave.to(device)
            ref = ref.to(device)  # Reference ECG for calibration guidance

            t = torch.randint(0, diffusion.num_timesteps, (ecg.shape[0],), device=device)
            model_kwargs = dict(y1=mmwave, y2=ref)
            loss_dict = diffusion.training_losses(model, ecg, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if use_ddp:
                # DDP模式下，通过 model.module 访问原始模型
                update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
            else:
                # 单GPU模式下，直接使用 model
                update_ema(ema, model, decay=0)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if use_ddp:
                    # DDP模式下需要all_reduce
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    # 单GPU模式下直接使用
                    avg_loss = avg_loss.item()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if use_ddp:
                    dist.barrier()

        if (epoch + 1) % 30 == 0:
            model.eval()
            for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(test_dataloader)):
                mmwave = mmwave[0:16]
                ecg = ecg[0:16]
                ref = ref[0:16]
                n = ecg.shape[0]
                mmwave = mmwave.to(device)
                ref = ref.to(device)

                z = torch.randn(n, 1, latent_size, latent_size, device=device)
                # Setup guidance:

                model_kwargs = dict(y1=mmwave, y2=ref)
                # Sample images:
                if use_ddp:
                    model_forward = model.module.forward
                else:
                    model_forward = model.forward

                samples = diffusion.p_sample_loop(
                    model_forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )

                ecg = ecg.reshape(n, 1, 1024)
                samples = samples.reshape(n, 1, 1024)
                ref = ref.reshape(n, 1, 1024)
                visualize_gen_curves(samples[0, 0], ecg[0, 0], viz, win=f'gen_ecg_val', title=f'gen ecg val')
                break
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    pcc_list = []
    for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(test_dataloader)):
        mmwave = mmwave
        ecg = ecg
        ref = ref
        n = ecg.shape[0]
        mmwave = mmwave.to(device)
        ref = ref.to(device)
        ecg = ecg.to(device)

        z = torch.randn(n, 1, latent_size, latent_size, device=device)
        # Setup guidance:

        model_kwargs = dict(y1=mmwave, y2=ref)
        # Sample images:
        if use_ddp:
            model_forward = model.module.forward
        else:
            model_forward = model.forward

        samples = diffusion.p_sample_loop(
            model_forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )

        # mmwave = mmwave.reshape(n, in_channels, 1024)[:, 0, :]
        ecg = ecg.reshape(n, 1, 1024)
        samples = samples.reshape(n, 1, 1024)
        ref = ref.reshape(n, 1, 1024)
        visualize_gen_curves(samples[0, 0], ecg[0, 0], viz, win=f'gen_ecg_val',
                             title=f'gen ecg val')
        # mmwave = mmwave.cpu().numpy()
        # ecg = ecg.cpu().numpy()
        # samples = samples.cpu().numpy()
        # sample_images(ref, mmwave, ecg, samples, batch_idx, f"{checkpoint_dir}/{train_steps:07d}_test.jpg")
        pcc, _ = batch_max_pearson_corr(ecg, samples, dim=-1, max_lag=50)
    pcc_list.extend([x.item() for x in pcc])
    mean_pcc = torch.tensor(pcc_list).mean()
    print(pcc)
    logger.info("Done!")
    # cleanup()
    return mean_pcc

def cross_domain(args, domain, train_index, test_index, data_spliter):
    viz = visdom.Visdom(env='cross domain', port=6006)
    batch_size = int(args.global_batch_size)
    train_dataset, val_dataset, test_dataset = data_spliter.split_data(domain, train_index, test_index, need_val=True)
    train_loader = get_dataloader(train_dataset, shuffle=True, collate_fn=None, batch_size=batch_size)
    val_loader = get_dataloader(val_dataset, shuffle=False, collate_fn=None, batch_size=batch_size)
    test_loader = get_dataloader(test_dataset, shuffle=False, collate_fn=None, batch_size=batch_size)
    pcc = main(args, train_loader, test_loader)
    return pcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--mmWave-channels", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=322)
    parser.add_argument("--ckpt-every", type=int, default=19320)
    parser.add_argument("--ckpt", type=str, default=None, help="Load your checkpoint here.")
    args = parser.parse_args()
    user = [i for i in range(11)]
    pcc_list = []

    for test_user in user:
        train_users = [j for j in user if j != test_user]
        temp_pcc = cross_domain(args, domain=1, train_index=train_users, test_index=[test_user],
                                data_spliter=MMECGDataSpliter(rand_ref=True))
        pcc_list.append(temp_pcc)
        print(pcc_list)
