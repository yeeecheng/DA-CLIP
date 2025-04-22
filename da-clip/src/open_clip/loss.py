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
            l1_loss_weight=0.1,
            num_loss_weight=0.5,
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
        self.l1_loss_fn = nn.L1Loss()
        # self.l2_loss_fn = nn.MSELoss()

        self.l1_loss_weight = l1_loss_weight
        self.num_loss_weight = num_loss_weight
        self.intra_degra_weight = 0.5
        self.temperature = temperature

    def numerical_contrastive_loss(self, image_degra_features, pos_text_features, neg_texts_features, deg_neg_text_features):
        B, D = image_degra_features.shape
        N_neg = neg_texts_features.size(1)

        text_features_all = torch.cat([
            pos_text_features.unsqueeze(1), neg_texts_features, deg_neg_text_features
        ], dim=1)

        logits_img2text = torch.bmm(text_features_all, image_degra_features.unsqueeze(2)).squeeze(2) / self.temperature
        labels_img2text = torch.zeros(B, dtype=torch.long, device=image_degra_features.device)
        loss_img2text = F.cross_entropy(logits_img2text, labels_img2text)

        logits_text2img = pos_text_features @ image_degra_features.T / self.temperature
        labels_text2img = torch.arange(B, dtype=torch.long, device=image_degra_features.device)
        loss_text2img = F.cross_entropy(logits_text2img, labels_text2img)

        loss = (loss_img2text + loss_text2img) / 2.0
        return loss

    def degradation_feature_contrastive_loss(self, image_degra_features, labels):
        B = image_degra_features.size(0)

        # Normalize features
        features = F.normalize(image_degra_features, dim=1)

        # Cosine similarity (B x B)
        similarity_matrix = features @ features.T

        # Build label mask (positive pairs only)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove self-similarity
        logits = similarity_matrix / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(B, device=features.device)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

    def forward(
            self,
            image_features,
            text_features,
            image_degra_features,
            text_degra_features,
            logit_scale,
            output_dict=False
    ):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        degra_loss = super().forward(image_degra_features, text_degra_features, logit_scale)

        if output_dict:
            return {"contrastive_loss": clip_loss, "degra_loss": degra_loss}

        return clip_loss, degra_loss

    # def forward(
    #         self,
    #         image_features,
    #         text_features,
    #         image_degra_features,
    #         logit_scale,
    #         output_dict=False,
    #         gt_image_features=None,
    #         pos_text_features=None,
    #         neg_texts_features=None,
    #         deg_neg_text_features=None,
    #         deg_label=None,
    # ):
    #     # CLIP-style contrastive loss
    #     clip_loss = super().forward(image_features, text_features, logit_scale)

    #     # Text ↔ degradation feature contrastive loss
    #     num_contrastive_loss = self.numerical_contrastive_loss(
    #         image_degra_features, pos_text_features, neg_texts_features, deg_neg_text_features
    #     )

    #     # GT ↔ image feature alignment loss
    #     gt_l1_loss = 0.0
    #     if gt_image_features is not None:
    #         gt_l1_loss = self.l1_loss_fn(image_features, gt_image_features)
    #         gt_l1_loss = self.l1_loss_weight * gt_l1_loss

    #     # Degraded feature ↔ degraded feature contrastive loss
    #     feat_feat_loss = 0.0
    #     if deg_label is not None:
    #         feat_feat_loss = self.degradation_feature_contrastive_loss(image_degra_features, deg_label)
    #         feat_feat_loss = self.intra_degra_weight * feat_feat_loss

    #     if output_dict:
    #         return {
    #             "contrastive_loss": clip_loss,
    #             "degra_loss": num_contrastive_loss,
    #             "gt_l1_loss": gt_l1_loss,
    #             "feat_feat_loss": feat_feat_loss
    #         }

    #     return clip_loss, num_contrastive_loss, gt_l1_loss, feat_feat_loss


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
