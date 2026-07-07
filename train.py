import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import torch
import math


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# # ============================================================
# # Patch NeighborhoodAttention1D to handle seq_len < kernel_size
# # 当序列长度小于 kernel_size*dilation 时，自动 pad 再裁剪回来
# # ============================================================
# import torch.nn.functional as F
# import natten

# # 尝试禁用 fused 后端（不同版本 API 名称不同，兜底处理）
# try:
#     natten.use_fused_na(False)
#     print("[NATTEN Patch] fused NA kernel 已禁用")
# except AttributeError:
#     try:
#         natten.use_gemm_na()
#         print("[NATTEN Patch] 已切换到 gemm NA")
#     except AttributeError:
#         print("[NATTEN Patch] WARNING: 当前 natten 版本无法通过 API 禁用 fused kernel，依赖 pad 方案")

# from natten import NeighborhoodAttention1D as _NA1D

# _orig_na1d_forward = _NA1D.forward

# def _safe_na1d_forward(self, x):
#     # x shape: (B, L, C)
#     ks  = self.kernel_size
#     dil = self.dilation

#     if isinstance(ks,  (tuple, list)): ks  = ks[0]
#     if isinstance(dil, (tuple, list)): dil = dil[0]
#     if dil is None or dil == 0:        dil = 1   # ← 修复 dilation=None 的情况

#     # fused kernel 要求 seq_len > ks * dil（严格大于）
#     # 所以 min_len 加 1，确保 pad 后满足严格不等式
#     min_len = ks * dil + 1              # ← 关键修改：+1
#     seq_len = x.shape[1]
#     pad_len = max(0, min_len - seq_len)

#     if pad_len > 0:
#         x = F.pad(x, (0, 0, 0, pad_len), mode="replicate")

#     out = _orig_na1d_forward(self, x)

#     if pad_len > 0:
#         out = out[:, :seq_len, :].contiguous()

#     return out

# _NA1D.forward = _safe_na1d_forward
# # ============================================================

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.debug_utils import backup_modules

def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def _get_len_safe(obj):
    try:
        return len(obj)
    except Exception:
        return None

def _eff_batches(total_batches, limit):
    if total_batches is None:
        return None
    if isinstance(limit, float):
        if limit <= 0:
            return 0
        # 至少 1（Lightning 通常也不会让它变成 0）
        return max(1, int(total_batches * limit))
    if isinstance(limit, int):
        return max(0, min(limit, total_batches))
    return total_batches  # None/未设置 -> 全量

def _summ(dl, limit, world_size=1):
    if dl is None:
        return {"dataset_len": None, "batch_size": None, "total_batches": None, "effective_batches": None}

    bs = getattr(dl, "batch_size", None)

    ds = getattr(dl, "dataset", None)
    ds_len = _get_len_safe(ds)

    # 最理想：len(dataloader) 就是 steps/epoch（每卡）
    dl_len = _get_len_safe(dl)

    # 如果 dataloader 没实现 __len__，尝试 sampler
    if dl_len is None:
        sampler = getattr(dl, "sampler", None)
        dl_len = _get_len_safe(sampler)

    # 再兜底：用 dataset_len/batch_size 估算（DDP 下按每卡样本估）
    if dl_len is None and ds_len is not None and bs is not None:
        per_rank = math.ceil(ds_len / max(1, world_size))
        dl_len = math.ceil(per_rank / bs)

    total_batches = dl_len
    eff_batches = _eff_batches(total_batches, limit)
    return {"dataset_len": ds_len, "batch_size": bs, "total_batches": total_batches, "effective_batches": eff_batches}

def print_data_info(datamodule, conf):
    # 只做 fit 阶段，避免 test/validate 触发额外构建
    try:
        datamodule.setup("fit")
    except Exception as e:
        print(f"[print_data_info] datamodule.setup('fit') failed: {e}")

    # 只拿 train/val，不碰 test
    train_dl = None
    val_dl = None
    try:
        train_dl = datamodule.train_dataloader()
    except Exception as e:
        print(f"[print_data_info] train_dataloader() failed: {e}")
    try:
        val_dl = datamodule.val_dataloader()
    except Exception as e:
        print(f"[print_data_info] val_dataloader() failed: {e}")

    world_size = int(conf.gpus) if isinstance(conf.gpus, int) else 1

    train_info = _summ(train_dl, conf.limit_train_batches, world_size=world_size)
    val_info   = _summ(val_dl, conf.limit_val_batches, world_size=world_size)

    print("\n========== DataModule Summary ==========")
    print(f"- cfg batch_size           = {conf.batch_size}")
    print(f"- dataloader.batch_size    = {train_info['batch_size']}")
    print(f"- train dataset_len        = {train_info['dataset_len']}")
    print(f"- train total_batches      = {train_info['total_batches']}")
    print(f"- train effective_batches  = {train_info['effective_batches']} (limit_train_batches={conf.limit_train_batches})")
    print(f"- val dataset_len          = {val_info['dataset_len']}")
    print(f"- val total_batches        = {val_info['total_batches']}")
    print("========================================\n")

@hydra.main(version_base=None, config_path="conf", config_name="config_train_ttt")
def main(conf):
    print_cuda_info()

    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    backup_modules(conf, __file__, output_dir)

    if conf.wandb != "disable":
        logger = WandbLogger(
            project="Forecast-MAE",
            name=conf.output,
            mode=conf.wandb,
            log_model="all",
            resume=conf.checkpoint is not None,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    print('==============================')
    print(logger._save_dir.split('/')[-1])
    print('==============================')

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false" if conf.gpus > 1 else None,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    datamodule = instantiate(conf.datamodule)

    # print_data_info(datamodule, conf)

    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)


if __name__ == "__main__":
    main()
