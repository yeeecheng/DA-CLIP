from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy


from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class MultiTypeDegradationPredictor(nn.Module):
    def __init__(self, num_bins=7, temperature=0.07, image_encoder=None):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature

        # Defer encoder and banks until forward()
        self.image_encoder = image_encoder
        self.degraded_prompt_feature = None
        self.bin_center_bank = None

        # Placeholder regressor, will be re-initialized per forward()
        in_dim = 7
        self.regressor = nn.Sequential(
            nn.Linear(in_dim,  8 * in_dim),
            nn.ReLU(),
            nn.Linear(8 * in_dim, self.num_bins * 4),  # one branch per type
            nn.Tanh()
        )

    def forward(self, image_degra_features, deg_type, bin_center_features, degraded_prompt_features):
        # image_degra_features: (B, D)
        # deg_type: (B,) of int IDs: 0=blur, 1=resize, 2=jpeg, 3=noisy
        # bin_center_bank: dict of {type_str: Tensor(K,)}
        # text_feat_bank: dict of {type_str: Tensor(K, D)}

        B, D = image_degra_features.shape
        K = self.num_bins

        # Cosine similarity and softmax: (B, K)
        sim = torch.sum(image_degra_features.unsqueeze(1) * degraded_prompt_features, dim=-1)  # (B, K)
        probs = F.softmax(sim / self.temperature, dim=-1)
        # Compute deltas: (B, 4, K) → then pick per-sample type: (B, K)
        delta_all = self.regressor(sim).view(B, 4, K)
        delta = delta_all[torch.arange(B), deg_type.view(-1)]  # (B, K)

        # Adjusted bin center: (B, K)
        adjusted = bin_center_features / (1.0 + delta)

        # Weighted sum prediction: (B,)
        preds = torch.sum(probs * adjusted, dim=-1)
        return preds

class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

        self.predictor = MultiTypeDegradationPredictor(num_bins=7)

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)

    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            image_features = self.clip.visual(image, control=hiddens)

            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            gt_images: Optional[torch.Tensor] = None,
            # deg_label: Optional[torch.Tensor] = None,
            deg_type: Optional[torch.Tensor] = None,
            gt_val: Optional[torch.Tensor] = None,
            bin_center_bank: Optional[torch.Tensor] = None,
            degraded_prompt_bank: Optional[torch.Tensor] = None,
            neg_texts: Optional[torch.Tensor] = None,
            deg_neg_texts: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        gt_image_features = self.encode_image(gt_images, control=False, normalize=True) if gt_images is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        neg_texts_features = [self.encode_text(net_text, normalize=True) for net_text in neg_texts.chunk(40, dim=-1)]
        deg_neg_text_features = [self.encode_text(deg_neg_text, normalize=True) for deg_neg_text in deg_neg_texts.chunk(120, dim=-1)]
        neg_texts_features = torch.stack(neg_texts_features, dim=0).transpose(0, 1)
        deg_neg_text_features = torch.stack(deg_neg_text_features, dim=0).transpose(0, 1)

        bin_center_features = bin_center_bank
        
        if degraded_prompt_bank is not None:
            degraded_prompt_features = [self.encode_text(degraded_prompt_bank[:, i, :], normalize=True) for i in range(7)]
            degraded_prompt_features = torch.stack(degraded_prompt_features, dim=1)  # [512, 7, D]

        pred = self.predictor(
            image_degra_features=image_degra_features,
            deg_type=deg_type,
            bin_center_features=bin_center_features,
            degraded_prompt_features=degraded_prompt_features
        )

        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "gt_image_features": gt_image_features,
            # "deg_label": deg_label,
            "deg_type": deg_type,
            "gt_val": gt_val,
            "pred": pred,
            "text_degra_features": text_degra_features,
            "bin_center_features": bin_center_features,
            "degraded_prompt_features": degraded_prompt_features,
            "logit_scale": self.logit_scale.exp(),
            "deg_neg_text_features": deg_neg_text_features,
            "neg_texts_features": neg_texts_features,
        }