from typing import List, Set, Optional
from copy import deepcopy
import datetime
from pathlib import Path
import importlib

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from metrics import MR, minADE, minFDE
from utils.optim import WarmupCosLR

from .model_ttt_peragent import ModelTTT
from model import layers


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        lr2: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        weight_decay2: float = 1e-4,
        # MAE
        decoder_depth=4,
        actor_mask_ratio=0.5,
        lane_mask_ratio=0.5,
        loss_weight: List[float] = [1.0, 1.0],
        forecast_loss_weight: List[float] = [1.0, 1.0, 1.0],
        mae_loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr2 = lr2
        self.weight_decay2 = weight_decay2
        self.save_hyperparameters()
        # self.submission_handler = SubmissionAv2()

        self.loss_weight = loss_weight
        self.forecast_loss_weight = forecast_loss_weight
        self.mae_loss_weight = mae_loss_weight

        self.net = ModelTTT(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            actor_mask_ratio=actor_mask_ratio,
            lane_mask_ratio=lane_mask_ratio,
            history_steps=historical_steps,
            future_steps=future_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")

        self.historical_steps = historical_steps
        self.future_steps = future_steps

        # ----- Anti-forgetting / long-horizon TTT add-ons -----
        # Frozen teacher copy used as a source-anchor for LwF distillation.
        # Lazily instantiated by ``setup_lwf_teacher`` to keep ``__init__`` light.
        self._lwf_teacher: Optional[nn.Module] = None
        # Toggles populated from config by ``setup_lwf_teacher`` /
        # ``configure_long_horizon``. Default zeros = no behavior change.
        self.lwf_weight: float = 0.0
        self.lwf_pi_weight: float = 0.0
        # Feature-level distillation weights (new: distill encoder features
        # rather than trajectory outputs — decouples "domain understanding"
        # from "target-specific output" so target adaptation is less punished).
        self.lwf_feature_agent_weight: float = 0.0   # constrain ego deep feature
        self.lwf_feature_lane_weight: float = 0.0    # constrain lane tokens
        self.long_horizon_gamma: float = 0.0  # 0 = uniform weights
        self.long_horizon_floor: float = 1.0  # minimum per-step weight


    def forward(self, data):
        out_forecast = self.net.forward_forecast(data)
        out_mae = self.net.forward_mae(data, out_forecast)

        out_forecast.update(out_mae)
        return out_forecast

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, out, data):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        if others_reg_mask.any():
            others_reg_loss = F.smooth_l1_loss(
                y_hat_others[others_reg_mask], y_others[others_reg_mask]
            )
        else:
            others_reg_loss = torch.zeros(1, device=y_hat_others.device).squeeze()

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        if lane_reg_mask.any():
            lane_pred_loss = F.mse_loss(
                lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
            )
        else:
            lane_pred_loss = torch.zeros(1, device=lane_pred.device).squeeze()

        # hist pred loss
        x_hat = out["x_mae_hat"]
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        if x_reg_mask.any():
            hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])
        else:
            hist_loss = torch.zeros(1, device=x_hat.device).squeeze()

        # future pred loss
        y_hat = out["y_mae_hat"]
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"].view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps)
        if reg_mask.any():
            future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])
        else:
            future_loss = torch.zeros(1, device=y_hat.device).squeeze()

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }

    def cal_loss_fre(self, out, data):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss 
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"].view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"].view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    def cal_loss_fre_obs(self, out, data, obs_fut_mask):
        # Forecast Los
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        # Horizon-weighted variant of the observed-future regression. With
        # gamma=0 (default) the weights are uniform → equivalent to the
        # original smooth_l1_loss reduction.
        T_y = y_hat_best.shape[-2]
        hw = self._horizon_weights(T_y, y_hat_best.device)  # [T_y]
        # obs_fut_mask: [..., T_y] aligned along the last axis; broadcast hw onto it.
        hw_b = hw.view(*([1] * (obs_fut_mask.ndim - 1)), T_y)
        sl1 = F.smooth_l1_loss(y_hat_best[..., :2], y, reduction="none")  # [..., T_y, 2]
        sl1_per_step = sl1.mean(dim=-1)  # [..., T_y]
        m = obs_fut_mask.float()
        agent_reg_loss = (sl1_per_step * m * hw_b).sum() / (m * hw_b).sum().clamp_min(1e-6)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        others_reg_mask = others_reg_mask * obs_fut_mask.unsqueeze(1)
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        forecast_loss = (
                self.forecast_loss_weight[0] * agent_reg_loss 
                + self.forecast_loss_weight[1] * agent_cls_loss
                + self.forecast_loss_weight[2] * others_reg_loss
        )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions"][:,:,:self.historical_steps,:] - data["x_centers"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_mask"][:, :, :self.historical_steps]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"]#.view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y"]#.view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_mask"][:, :, self.historical_steps:]
        reg_mask[~future_pred_mask] = False
        # reg_mask = reg_mask.view(-1, self.future_steps)
        reg_mask = reg_mask * obs_fut_mask.unsqueeze(1)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        loss = (
            self.loss_weight[0] * forecast_loss
            + self.loss_weight[1] * mae_loss
            )

        return {
            "loss" : loss,
            "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            "cls_loss": agent_cls_loss,
            "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    # def cal_loss_fre_obs_effi(self, out, data, obs_fut_mask):
        # Forecast Los
        y_hat = out["y_hat"]
        y = data["y_agent"][:, 0]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2][obs_fut_mask], y[obs_fut_mask])
        # agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # others_reg_mask = ~data["x_padding_mask"][:, 1:, self.historical_steps:]
        # others_reg_mask = others_reg_mask * obs_fut_mask.unsqueeze(1)
        # others_reg_loss = F.smooth_l1_loss(
        #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
        # )

        # forecast_loss = (
        #         self.forecast_loss_weight[0] * agent_reg_loss 
        #         + self.forecast_loss_weight[1] * agent_cls_loss
        #         + self.forecast_loss_weight[2] * others_reg_loss
        # )

        ## MAE loss

        # lane pred loss
        lane_pred = out["lane_mae_pred"]
        lane_normalized = out["lane_normalized"]
        lane_pred_mask = out["lane_pred_mask"]

        lane_padding_mask = data["lane_padding_mask"]

        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = out["x_mae_hat"].view(-1, self.historical_steps, 2)
        hist_pred_mask = out["hist_pred_mask"]

        x_gt = (data["x_positions_4_mae"][:,:,:self.historical_steps,:] - data["x_centers_4_mae"].unsqueeze(-2)).view(-1, self.historical_steps, 2)
        x_reg_mask = ~data["x_padding_hist_4_mae"]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.historical_steps)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x_gt[x_reg_mask])

        # future pred loss
        y_hat = out["y_mae_hat"]#.view(-1, self.future_steps, 2)
        future_pred_mask = out["future_pred_mask"]

        y_gt = data["y_4_mae"]#.view(-1, self.future_steps, 2)
        reg_mask = ~data["x_padding_fut_4_mae"]
        reg_mask[~future_pred_mask] = False
        # reg_mask = reg_mask.view(-1, self.future_steps)
        reg_mask = reg_mask * obs_fut_mask.unsqueeze(1)
        future_loss = F.l1_loss(y_hat[reg_mask], y_gt[reg_mask])

        mae_loss = (
                    self.mae_loss_weight[0] * future_loss
                    + self.mae_loss_weight[1] * hist_loss
                    + self.mae_loss_weight[2] * lane_pred_loss
        )

        # loss = (
        #     self.loss_weight[0] * forecast_loss
        #     + self.loss_weight[1] * mae_loss
        #     )

        return {
            # "loss" : loss,
            # "forecast_loss": forecast_loss,
            "reg_loss": agent_reg_loss,
            # "cls_loss": agent_cls_loss,
            # "others_reg_loss": others_reg_loss,
            "mae_loss": mae_loss,
            "future_loss" : future_loss,
            "hist_loss" : hist_loss,
            "lane_pred_loss" : lane_pred_loss,
        }
    
    # ===================================================================
    # Anti-forgetting (LwF) + long-horizon TTT helpers
    # ===================================================================
    def setup_lwf_teacher(
        self,
        lwf_weight: float = 0.0,
        lwf_pi_weight: float = 0.0,
        lwf_feature_agent_weight: float = 0.0,
        lwf_feature_lane_weight: float = 0.0,
    ):
        """Clone the *currently-loaded* net as a frozen teacher.

        Call this once in test.py AFTER load_from_checkpoint and BEFORE the
        TTT loop. The teacher represents the source-pretrained predictor and
        is never updated. All-zero weights keep behavior identical to vanilla T4P.

        - ``lwf_weight``: distill on trajectory outputs (output-level LwF).
        - ``lwf_pi_weight``: distill on mode-prob logits.
        - ``lwf_feature_agent_weight``: distill on ego deep feature (x_agent).
          Constrains the "domain understanding" without constraining the
          specific trajectory — allows target adaptation to still work.
        - ``lwf_feature_lane_weight``: distill on lane token features.
          Constrains the encoded scene representation.
        """
        self.lwf_weight = float(lwf_weight)
        self.lwf_pi_weight = float(lwf_pi_weight)
        self.lwf_feature_agent_weight = float(lwf_feature_agent_weight)
        self.lwf_feature_lane_weight = float(lwf_feature_lane_weight)
        any_on = (self.lwf_weight > 0.0 or self.lwf_pi_weight > 0.0
                  or self.lwf_feature_agent_weight > 0.0
                  or self.lwf_feature_lane_weight > 0.0)
        if not any_on:
            self._lwf_teacher = None
            return
        teacher = deepcopy(self.net)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self._lwf_teacher = teacher

    def configure_long_horizon(self, gamma: float = 0.0, floor: float = 1.0):
        """Configure per-timestep loss weighting for long-horizon supervision.

        Weight at step t (1-indexed) is ``floor + gamma * (t / T)``. With
        gamma=0 (default) the schedule is uniform → no behavior change.
        gamma=2.0 roughly triples the loss at the final step vs the first.
        """
        self.long_horizon_gamma = float(gamma)
        self.long_horizon_floor = float(floor)

    def _horizon_weights(self, T: int, device) -> torch.Tensor:
        """Return shape [T] per-step weights, normalized to mean 1."""
        if self.long_horizon_gamma == 0.0:
            return torch.ones(T, device=device)
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
        w = self.long_horizon_floor + self.long_horizon_gamma * t
        w = w * (T / w.sum())  # normalize so mean weight = 1 (preserves scale)
        return w

    @torch.no_grad()
    def _teacher_forward(self, data):
        """Run the frozen teacher with the same per-agent path the student uses."""
        if self._lwf_teacher is None:
            return None
        was_training = self._lwf_teacher.training
        self._lwf_teacher.eval()
        out_f = self._lwf_teacher.forward_forecast_peragent_fre(data)
        out_m = self._lwf_teacher.forward_mae_fre(data, out_f)
        out_m.update(out_f)
        if was_training:
            self._lwf_teacher.train()
        return out_m

    def compute_lwf_loss(self, student_out, data, obs_fut_mask=None) -> torch.Tensor:
        """LwF distillation: student tracks the frozen source teacher on the
        *unobserved* portion of the future. Where GT is available
        (obs_fut_mask=True), we let the original ``cal_loss_fre_obs`` regression
        loss do the work; LwF complements it by anchoring the part of the
        horizon that has no supervision yet — exactly the part where T4P
        drifts and forgets.
        """
        self._lwf_call_count = getattr(self, '_lwf_call_count', 0) + 1
        any_active = (self.lwf_weight > 0.0 or self.lwf_pi_weight > 0.0
                      or self.lwf_feature_agent_weight > 0.0
                      or self.lwf_feature_lane_weight > 0.0)
        if self._lwf_teacher is None or not any_active:
            self._lwf_skipped_reason = (
                'teacher_is_None' if self._lwf_teacher is None else 'all_weights_zero'
            )
            return torch.zeros((), device=next(self.parameters()).device)

        # Sync teacher's actor_embeds with the student's (test.py recreates
        # them per scene, so the deepcopy at __init__ time is stale).
        try:
            self._lwf_teacher.actor_embeds = self.net.actor_embeds
        except Exception as _e:
            self._lwf_error_count = getattr(self, '_lwf_error_count', 0) + 1
            self._lwf_skipped_reason = f'actor_embeds_sync_failed:{_e}'
            return torch.zeros((), device=next(self.parameters()).device)

        # Guard: teacher forward needs consistent shapes. If ``data`` has been
        # pad_sequenced across time but ``actor_names`` was skipped in the pad
        # loop (see test.py), the actor lookup produces a [N0,D] tensor while
        # actor_feat is [T_obs, N_max, D] → dim-1 mismatch. Detect and skip.
        names = data.get("actor_names")
        x_key_padding_mask = data.get("x_key_padding_mask")
        if (names is not None and x_key_padding_mask is not None
                and hasattr(x_key_padding_mask, "shape")):
            try:
                n_from_names = len(names[0])
                n_from_padding = x_key_padding_mask.shape[-1]
                if n_from_names != n_from_padding:
                    self._lwf_error_count = getattr(self, '_lwf_error_count', 0) + 1
                    self._lwf_skipped_reason = (
                        f'shape_mismatch:names={n_from_names}!=padding={n_from_padding}. '
                        f'Pass current-step batch, not accumulated test_batch_.'
                    )
                    if self._lwf_error_count <= 3:
                        print(f"[LwF] {self._lwf_skipped_reason}")
                    return torch.zeros((), device=next(self.parameters()).device)
            except Exception:
                pass  # non-fatal — proceed and let teacher_forward raise if broken

        try:
            with torch.no_grad():
                teacher_out = self._teacher_forward(data)
        except Exception as _e:
            self._lwf_error_count = getattr(self, '_lwf_error_count', 0) + 1
            self._lwf_skipped_reason = f'teacher_forward_failed:{type(_e).__name__}:{_e}'
            if self._lwf_error_count <= 3:
                print(f"[LwF] teacher_forward raised: {self._lwf_skipped_reason}")
            return torch.zeros((), device=next(self.parameters()).device)

        # 1) trajectory regression distillation (best-mode minADE-style)
        y_s, y_t = student_out["y_hat"], teacher_out["y_hat"].detach()
        # match the student's selected mode to the teacher to make distillation
        # invariant to mode permutation
        l2 = torch.norm(y_s[..., :2] - y_t[..., :2].mean(dim=1, keepdim=True), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2, dim=-1)
        y_s_best = y_s[torch.arange(y_s.shape[0]), best_mode]
        y_t_best = y_t[torch.arange(y_t.shape[0]), best_mode]

        # apply horizon weights so distillation also emphasizes the long horizon
        T = y_s_best.shape[-2]
        w = self._horizon_weights(T, y_s_best.device).view(1, T, 1)
        # focus distillation on the *unobserved* horizon (complement of obs_fut_mask)
        if obs_fut_mask is not None and obs_fut_mask.ndim >= 1:
            # obs_fut_mask: [T_obs, T] from test.py — True where GT is observed
            # We want a per-step weight that is HIGH where GT is NOT yet observed
            unobs = (~obs_fut_mask.bool()).float()  # [T_obs, T]
            unobs_step = unobs.mean(dim=0).view(1, T, 1)
            w = w * (0.5 + unobs_step)  # multiplicative: stronger on unobserved
        reg_lwf = ((y_s_best[..., :2] - y_t_best[..., :2]).pow(2) * w).mean()

        # 2) optional logits distillation on mode probabilities pi
        pi_lwf = torch.zeros((), device=reg_lwf.device)
        if self.lwf_pi_weight > 0.0 and "pi" in student_out and "pi" in teacher_out:
            log_s = F.log_softmax(student_out["pi"], dim=-1)
            log_t = F.log_softmax(teacher_out["pi"].detach(), dim=-1)
            pi_lwf = F.kl_div(log_s, log_t, log_target=True, reduction="batchmean")

        # 3) NEW — feature-level distillation on encoder outputs.
        # Motivation: output-level LwF (reg_lwf) constrains "which trajectory",
        # which fights target adaptation directly. Feature LwF constrains
        # "how the scene is understood" instead, letting the decoder freely
        # produce a target-appropriate trajectory from source-preserved
        # scene features.
        feat_agent_lwf = torch.zeros((), device=reg_lwf.device)
        if (self.lwf_feature_agent_weight > 0.0
                and "x_agent" in student_out and "x_agent" in teacher_out):
            f_s = student_out["x_agent"]
            f_t = teacher_out["x_agent"].detach()
            feat_agent_lwf = F.mse_loss(f_s, f_t)

        feat_lane_lwf = torch.zeros((), device=reg_lwf.device)
        if (self.lwf_feature_lane_weight > 0.0
                and "x_encoder_deep" in student_out and "x_encoder_deep" in teacher_out):
            f_s = student_out["x_encoder_deep"]
            f_t = teacher_out["x_encoder_deep"].detach()
            # Distill only over LANE tokens (indices N: onwards) — actor tokens
            # already covered by x_agent; overlapping constraints would double-count.
            # We approximate "lane region" by taking the last M tokens where M is
            # inferred from lane_key_padding_mask if available.
            lane_mask = data.get("lane_key_padding_mask")
            if lane_mask is not None and hasattr(lane_mask, "shape"):
                M = lane_mask.shape[-1]
                if f_s.shape[1] >= M and f_t.shape[1] >= M:
                    f_s = f_s[:, -M:]
                    f_t = f_t[:, -M:]
                    # Skip padded lane tokens (True = padded)
                    valid = (~lane_mask.bool()).unsqueeze(-1).float()
                    denom = valid.sum().clamp_min(1.0)
                    feat_lane_lwf = ((f_s - f_t).pow(2) * valid).sum() / (denom * f_s.shape[-1])

        total = (self.lwf_weight * reg_lwf
                 + self.lwf_pi_weight * pi_lwf
                 + self.lwf_feature_agent_weight * feat_agent_lwf
                 + self.lwf_feature_lane_weight * feat_lane_lwf)
        self._lwf_last_loss = float(total.detach().item())
        self._lwf_skipped_reason = None  # success
        if self._lwf_call_count <= 3:
            print(f"[LwF] call#{self._lwf_call_count}: "
                  f"reg={float(reg_lwf):.4f} pi={float(pi_lwf):.4f} "
                  f"feat_agent={float(feat_agent_lwf):.4f} "
                  f"feat_lane={float(feat_lane_lwf):.4f} "
                  f"total={self._lwf_last_loss:.4f}")
        return total

    def _preprocess_data(self, data):
        for key in ["x", "x_velocity_diff", "y", "x_attr", "x_centers", "x_angles",
                    "lane_positions", "lane_centers", "lane_angles", "x_positions",
                    "x_positions_4_mae", "x_centers_4_mae"]:
            if key in data and isinstance(data[key], torch.Tensor):
                data[key] = torch.nan_to_num(data[key], nan=0.0, posinf=0.0, neginf=0.0)
        for key in ["x_key_padding_mask", "lane_key_padding_mask", "x_padding_mask", "lane_padding_mask"]:
            if key in data and isinstance(data[key], torch.Tensor) and data[key].dtype != torch.bool:
                data[key] = data[key].bool()
        # Ensure ego agent (index 0) is never masked — prevents all-masked attention rows → NaN
        if "x_key_padding_mask" in data and data["x_key_padding_mask"].shape[1] > 0:
            data["x_key_padding_mask"][:, 0] = False

    def training_step(self, data, batch_idx):
        self._preprocess_data(data)
        out = self(data)
        losses = self.cal_loss(out, data)

        if not torch.isfinite(losses["loss"]):
            return None

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        self._preprocess_data(data)
        out = self(data)
        losses = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"][:, 0])

        for k, v in losses.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    # def on_test_start(self) -> None:
    #     save_dir = Path("./submission")
    #     save_dir.mkdir(exist_ok=True)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    #     self.submission_handler = SubmissionAv2(
    #         save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
    #     )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
    
    def configure_ttt_optimizers(self, conf):
        blacklist = set(conf.blacklist)
        whitelist = set(conf.whitelist)
        assert len(blacklist & whitelist) == 0

        update = set()
        freeze = set()

        blacklist_weight_modules, whitelist_weight_modules = set(), set()
        for module in blacklist:
            blacklist_weight_modules.add(getattr(nn, module))
        for module in whitelist:
            whitelist_weight_modules.add(getattr(nn, module))
        blacklist_weight_modules, whitelist_weight_modules = tuple(blacklist_weight_modules), tuple(whitelist_weight_modules)

        assert len(blacklist) == len(blacklist_weight_modules)
        assert len(whitelist) == len(whitelist_weight_modules)
        
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if isinstance(module, whitelist_weight_modules):
                    update.add(full_param_name)
                elif isinstance(module, blacklist_weight_modules):
                    freeze.add(full_param_name)
                elif 'actor_type_embed' in param_name:
                    freeze.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    if conf.update_param:
                        update.add(full_param_name)
                    else:
                        freeze.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = update & freeze
        union_params = update | freeze
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(update))
                ],
                "weight_decay": self.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]

    def freeze_layers(self, conf):
        if conf.fr_embedding:
            for param in self.net.pos_embed.parameters():
                param.requires_grad = False
            for param in self.net.decoder_pos_embed.parameters():
                param.requires_grad = False
            self.net.actor_type_embed.requires_grad = False
            self.net.lane_type_embed.requires_grad = False
            self.net.history_mask_token.requires_grad = False
            self.net.future_mask_token.requires_grad = False
            self.net.lane_mask_token.requires_grad = False

        if conf.fr_first_layer:
            for param in self.net.hist_embed.parameters():
                param.requires_grad = False
            for param in self.net.future_embed.parameters():
                param.requires_grad = False
            for param in self.net.lane_embed.parameters():
                param.requires_grad = False
            
        if conf.fr_enc_layer:
            for param in self.net.blocks.parameters():
                param.requires_grad = False
            for param in self.net.norm.parameters():
                param.requires_grad = False

        if conf.fr_dec_layer:
            for param in self.net.decoder_embed.parameters():
                param.requires_grad = False
            for param in self.net.decoder_blocks.parameters():
                param.requires_grad = False
            for param in self.net.decoder_norm.parameters():
                param.requires_grad = False

        if conf.fr_last_fore:
            for param in self.net.decoder.parameters():
                param.requires_grad = False
            for param in self.net.dense_predictor.parameters():
                param.requires_grad = False

        if conf.fr_last_mae:
            for param in self.net.lane_pred.parameters():
                param.requires_grad = False
            for param in self.net.history_pred.parameters():
                param.requires_grad = False
            for param in self.net.future_pred.parameters():
                param.requires_grad = False