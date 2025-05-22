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
        Multi-type FCRC loss:
        - image_degra_features: (B, D)
        - all_d_type_tokens_features: (B, 28, D)
        - bin_center_features: (B, 4, 7)
        - gt_val: (B, 4)
        - deg_type: (B, 4), binary mask
        """
        device = image_degra_features.device
        B, D = image_degra_features.shape
        num_types, num_bins = 4, 7

        # === Step 1: Collect valid (sample_idx, type_idx) pairs ===
        exist_mask = deg_type.bool()  # (B, 4)
        sample_idx, type_idx = torch.nonzero(exist_mask, as_tuple=True)  # (N,), (N,)
        N = sample_idx.shape[0]

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # === Step 2: Prepare per-sample features ===
        img_feat = image_degra_features[sample_idx]  # (N, D)
        sim_all = F.cosine_similarity(img_feat.unsqueeze(1), all_d_type_tokens_features[sample_idx], dim=-1)  # (N, 28)
        sim_exp = torch.softmax(sim_all / self.temperature, dim=-1)  # (N, 28)

        # === Step 3: Select ground truth bin ===
        bin_centers = bin_center_features[sample_idx, type_idx]  # (N, 7)
        gt_vals = gt_val[sample_idx, type_idx]  # (N,)
        abs_diffs = torch.abs(gt_vals.unsqueeze(1) - bin_centers)  # (N, 7)
        bin_idx = torch.argmin(abs_diffs, dim=-1)  # (N,)

        pos_token_idx = type_idx * num_bins + bin_idx  # (N,)
        pos = sim_exp[torch.arange(N), pos_token_idx]  # (N,)

        # === Step 4: Normalize gt_vals ===
        type_ranges = [(0.5, 4.0), (5.0, 40.0), (0.5, 4.0), (10.0, 80.0)]
        lows = torch.tensor([type_ranges[t][0] for t in type_idx], device=device)
        highs = torch.tensor([type_ranges[t][1] for t in type_idx], device=device)
        gt_vals_norm = (gt_vals - lows) / (highs - lows + 1e-8)  # (N,)

        # === Step 5: Pairwise distance & weighting ===
        diff_matrix = torch.abs(gt_vals_norm[:, None] - gt_vals_norm[None, :])  # (N, N)
        same_type_mask = (type_idx[:, None] == type_idx[None, :]).float()  # (N, N)
        lambda_weight = same_type_mask * diff_matrix + (1.0 - same_type_mask) * 4.0
        lambda_weight = lambda_weight / (lambda_weight.sum(dim=1, keepdim=True) + 1e-8)  # (N, N)

        # === Step 6: Compute negative score ===
        neg = (lambda_weight @ sim_exp).sum(dim=1) - lambda_weight.diag() * pos  # (N,)

        # === Step 7: Final loss ===
        loss = -torch.log(pos / (pos + neg + 1e-6)).mean()
        return loss


    # def compute_fcrc_loss(
    #     self,
    #     image_degra_features,            # (B, D)
    #     all_d_type_tokens_features,      # (B, 28, D)
    #     gt_val,                          # (B, 4)
    #     bin_center_features,             # (B, 4, 7)
    #     deg_type                         # (B, 4)
    # ):
    #     device = image_degra_features.device
    #     B, D = image_degra_features.shape
    #     num_types, num_bins = 4, 7
    #     eps = 1e-6

    #     # === Step 1: FCRC Loss ===
    #     exist_mask = deg_type.bool()  # (B, 4)
    #     batch_idx, type_idx = torch.nonzero(exist_mask, as_tuple=True)
    #     N = batch_idx.shape[0]
    #     if N == 0:
    #         return torch.tensor(0.0, device=device, requires_grad=True)

    #     img_feat = image_degra_features[batch_idx]  # (N, D)
    #     gt_val_selected = gt_val[batch_idx, type_idx]  # (N,)
    #     bin_centers = bin_center_features[batch_idx, type_idx]  # (N, 7)
    #     all_tokens = all_d_type_tokens_features[batch_idx]  # (N, 28, D)

    #     sim_all = F.cosine_similarity(img_feat.unsqueeze(1), all_tokens, dim=-1)  # (N, 28)
    #     sim_exp = torch.softmax(sim_all / self.temperature, dim=-1)

    #     abs_diffs = torch.abs(gt_val_selected.unsqueeze(1) - bin_centers)  # (N, 7)
    #     bin_idx = torch.argmin(abs_diffs, dim=-1)  # (N,)
    #     pos_idx = type_idx * num_bins + bin_idx  # (N,)
    #     pos = sim_exp[torch.arange(N), pos_idx]
    #     neg = (sim_exp.sum(dim=1) - pos)

    #     fcrc_loss = -torch.log(pos / (pos + neg + eps)).mean()

    #     # === Step 2: Inter-type Repulsion ===
    #     repulsion_loss = 0.0
    #     margin_sq = 0.5 ** 2
    #     ref_token = all_d_type_tokens_features[0]  # (28, D)
    #     type_ids = torch.arange(num_types, device=device).repeat_interleave(num_bins)  # (28,)
    #     for i in range(num_types):
    #         for j in range(i + 1, num_types):
    #             bins_i = ref_token[type_ids == i]  # (7, D)
    #             bins_j = ref_token[type_ids == j]  # (7, D)
    #             dists = torch.cdist(bins_i, bins_j, p=2)  # (7, 7)
    #             margin_penalty = F.relu(margin_sq - dists ** 2).mean()
    #             repulsion_loss += margin_penalty
    #     repulsion_loss = repulsion_loss / (num_types * (num_types - 1) / 2)

    #     # === Step 3: Ordinal Constraint ===
    #     order_loss = 0.0
    #     for t in range(num_types):
    #         bin_feats = ref_token[t * num_bins:(t + 1) * num_bins]  # (7, D)
    #         for i in range(num_bins - 1):
    #             sim_i = F.cosine_similarity(bin_feats[i].unsqueeze(0), bin_feats[i + 1:], dim=-1)  # (6,)
    #             order_loss += F.relu(sim_i.mean() - F.cosine_similarity(bin_feats[i], bin_feats[i], dim=0))
    #     order_loss = order_loss / (num_types * (num_bins - 1))

    #     # === Final Loss ===
    #     loss = (
    #         fcrc_loss +
    #         0.1 * repulsion_loss +
    #         0.05 * order_loss
    #     )

    #     return loss



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

        reg_ls_loss = 0.0
        if gt_val is not None:
            # deg_type: (B, 4), gt_val: (B, 4)
            mask_exist = deg_type.float()
            mask_non_exist = 1.0 - mask_exist

            # loss for types that exist
            loss_exist = F.mse_loss(pred * mask_exist, gt_val * mask_exist, reduction='sum') / (mask_exist.sum() + 1e-8)

            # loss for types that should not exist (predict near 0)
            loss_non_exist = F.mse_loss(pred * mask_non_exist, torch.zeros_like(pred), reduction='sum') / (mask_non_exist.sum() + 1e-8)

            reg_ls_loss = loss_exist + 1.0 * loss_non_exist


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
