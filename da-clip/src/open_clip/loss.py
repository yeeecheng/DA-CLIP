import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text


    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss

class DaClipLoss(ClipLoss):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            intra_type_contrastive_loss_weight=1.0,
            inter_type_contrastive_loss_weight=1.0,
            temperature=0.07
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.l1_loss_fn = nn.L1Loss(reduction="mean")
        self.l1_loss_weight = 0.1
        self.reg_l1_loss_fn = nn.L1Loss(reduction="mean")
        self.intra_type_contrastive_loss_weight = intra_type_contrastive_loss_weight
        self.inter_type_contrastive_loss_weight = inter_type_contrastive_loss_weight
        self.temperature = temperature

        self.type_ranges = {
            '0': (0.5, 4.0),
            '1': (5.0, 40.0),
            '2': (0.5, 4.0),
            '3': (10.0, 80.0),
        }



    def compute_fcrc_loss(self, image_degra_features, all_d_type_tokens_features, gt_val, bin_center_features, deg_type):
        """
        - image_degra_features: (B, D)
        - all_d_type_tokens_features: (B, 28, D)
        - bin_center_features: (B, 4, 7)
        - deg_val: (B, 4)
        """
        device = image_degra_features.device
        B, D = image_degra_features.shape
        num_types, num_bins = 4, 7

        # Step 1: Cosine similarity (B, 28)
        sim = F.cosine_similarity(image_degra_features.unsqueeze(1), all_d_type_tokens_features, dim=-1)  # (B, 28)
        # sim_exp = torch.exp(sim / self.temperature)  # (B, 28)
        sim_exp = torch.softmax(sim / self.temperature, dim=-1)

        # Step 2: Positive index for each sample
        bin_centers_selected = bin_center_features[torch.arange(B), deg_type]  # (B, 7)
        deg_val_selected = gt_val[torch.arange(B), deg_type]  # (B,)
        abs_diffs = torch.abs(deg_val_selected.unsqueeze(1) - bin_centers_selected)  # (B, 7)
        bin_idx = torch.argmin(abs_diffs, dim=-1)  # (B,)

        pos_idx = deg_type * num_bins + bin_idx
        pos = sim_exp[torch.arange(B), pos_idx]

         # Step 3: Normalize deg_val (B, 4) per type
        norm_scale = 1.0
        norm_deg_val = torch.zeros_like(gt_val)  # (B, 4)
        type_ranges_list = [(0.5, 4.0), (5.0, 40.0), (0.5, 4.0), (10.0, 80.0)]
        for t in range(4):
            low, high = type_ranges_list[t]
            norm_deg_val[:, t] = (gt_val[:, t] - low) / (high - low  + 1e-8) * norm_scale  # (B, 4)
        # get deg_type by deg_val（normalize）
        norm_deg_val_main = norm_deg_val[torch.arange(B), deg_type]  # (B,)

        # Pairwise lambda
        dist_same_type = torch.abs(norm_deg_val_main.view(B, 1) - norm_deg_val_main.view(1, B))   # (B, B)
        # Inter-type fixed 1.5
        same_type_mask = (deg_type.view(B, 1) == deg_type.view(1, B)).float()
        dist_diff = torch.ones_like(dist_same_type) * 4.0
        dist = same_type_mask * dist_same_type + (1.0 - same_type_mask) * dist_diff
        dist = dist / (dist.sum(dim=1, keepdim=True) + 1e-8)  # (B, B)

        # Step 4: Compute Neg using all 28 tokens
        # sim_exp: (B, 28), lambda_weight: (B, B)
        neg = (dist @ sim_exp).sum(dim=1) - dist.diag() * pos  # (B,)
        neg = neg.view(-1)  # Ensure neg is a 1D tensor

        # Step 5: Final loss
        loss = -torch.log(pos / (pos + neg + 1e-6)).mean()
        return loss

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            output_dict=False,
            image_degra_features=None,
            gt_image_features=None,
            text_degra_features=None,
            # deg_label=None,
            deg_type=None,
            gt_val= None,
            all_d_type_tokens_features=None,
            bin_center_features=None,
            pred=None,
    ):
        # CLIP-style contrastive loss
        clip_loss = super().forward(image_features, text_features, logit_scale)

        # GT ↔ image feature alignment loss
        gt_l1_loss = 0.0
        if gt_image_features is not None:
            gt_l1_loss = self.l1_loss_fn(image_features, gt_image_features)
            gt_l1_loss = self.l1_loss_weight * gt_l1_loss

        # regression loss 全改用 MSE
        reg_ls_loss = 0.0
        if gt_val is not None:
            # 針對存在 type -> MSE
            mask_exist = (gt_val > 0).float()
            # print(mask_exist)
            loss_exist = F.mse_loss(pred * mask_exist, gt_val * mask_exist, reduction='sum') / (mask_exist.sum() + 1e-8)

            # 針對不存在 type -> MSE
            mask_non_exist = (gt_val == 0).float()
            # print(mask_non_exist)
            loss_non_exist = F.mse_loss(pred * mask_non_exist, torch.zeros_like(pred) * mask_non_exist, reduction='sum') / (mask_non_exist.sum() + 1e-8)

            reg_ls_loss = loss_exist + 1.0 * loss_non_exist  # 權重可以保留 1.0 或加強


        fcrc_loss = 0.0
        fcrc_loss = self.compute_fcrc_loss(image_degra_features, all_d_type_tokens_features, gt_val, bin_center_features, deg_type)


        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "gt_l1_loss": gt_l1_loss,
                "reg_ls_loss": reg_ls_loss,
                "fcrc_loss": fcrc_loss
            }
        # return clip_loss, gt_l1_loss, intra_type_contrastive_loss
        return clip_loss, gt_l1_loss, reg_ls_loss, fcrc_loss

class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
