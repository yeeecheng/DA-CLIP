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


class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

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
            pos_text: Optional[torch.Tensor] = None,
            neg_texts: Optional[torch.Tensor] = None,
            deg_neg_texts: Optional[torch.Tensor] = None,
            resiual_images: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        gt_image_features = self.encode_image(gt_images, control=False, normalize=True) if gt_images is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        resiual_image_features = self.encode_image(resiual_images, control=False, normalize=True) if resiual_images is not None else None

        # degradation_features = self.encode_text(degradation, normalize=True) if text is not None else None
        pos_text_features = self.encode_text(pos_text, normalize=True) if pos_text is not None else None
        neg_texts_features = [self.encode_text(net_text, normalize=True) for net_text in neg_texts.chunk(40, dim=-1)]
        deg_neg_text_features = [self.encode_text(deg_neg_text, normalize=True) for deg_neg_text in deg_neg_texts.chunk(40, dim=-1)]
        neg_texts_features = torch.stack(neg_texts_features, dim=0).transpose(0, 1)
        deg_neg_text_features = torch.stack(deg_neg_text_features, dim=0).transpose(0, 1)
        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "gt_image_features": gt_image_features,
            "resiual_image_features": resiual_image_features,
            # "degradation_features": degradation_features,
            "pos_text_features": pos_text_features,
            "neg_texts_features": neg_texts_features,
            "deg_neg_text_features": deg_neg_text_features,
            "logit_scale": self.logit_scale.exp()
        }


