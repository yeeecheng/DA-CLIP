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

        

    # def compute_intra_type_contrastive_loss(self, image_degra_features, text_degra_features, deg_label, deg_type):

    #     loss = 0.0
    #     B, D = image_degra_features.shape
    #     N = text_degra_features.shape[0]

    #     # Cosine similarity (B x B)
    #     sim = (image_degra_features @ text_degra_features.T) / self.temperature  # f(z_i, w_j)
    #     sim_exp = torch.exp(sim)  # shape (B, B)

    #     # label distance matrix (|y_i - y_j|): (B, B)
    #     levels_per_type = 8
    #     local_label = 2 * (deg_label - deg_type * levels_per_type)  # (B,)
    #     label_dist = torch.abs(local_label.unsqueeze(1) - local_label.unsqueeze(0)).float()

    #     # type mask (only compare samples of same degradation type)
    #     type_mask = (deg_type.unsqueeze(1) == deg_type.unsqueeze(0))  # shape (B, B), bool

    #     # Compute λ_{i,j} = softmax(β * d_{i,j}) only within same type
    #     raw_weights = torch.exp(self.beta * label_dist) * type_mask.float()  # (B, B)
    #     weights = raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-8)  # normalized across j

    #     eye = torch.eye(B, device=sim.device)
    #     mask = 1.0 - eye  # remove diagonal self-pairs from denominator

    #     # Diagonal: matched sim
    #     pos_sim = sim_exp.diag()
    #     denom = (weights * sim_exp * mask).sum(dim=1) + pos_sim

    #     loss = -torch.log(pos_sim / (denom + 1e-8))

    #     return loss.mean()

    # def compute_inter_type_contrastive_loss(self, image_degra_features, deg_type, margin=0.2):
    #     unique_types = deg_type.unique()
    #     centers = []

    #     # 1. calculate each type center
    #     for t in unique_types:
    #         mask = (deg_type == t)
    #         if mask.sum() > 0:
    #             center = image_degra_features[mask].mean(dim=0, keepdim=True)
    #             centers.append(center)

    #     centers = torch.cat(centers, dim=0)  # (T, D)
    #     centers = F.normalize(centers, dim=-1)  # cosine similarity

    #     # 2. calculate pairwise similarity
    #     sim_matrix = centers @ centers.T  # (T, T)
    #     T = sim_matrix.size(0)
    #     eye = torch.eye(T, device=sim_matrix.device)
    #     mask = 1.0 - eye  # (T, T), zero diagonal

    #     # 3. apply margin-based repulsion
    #     penalty = F.relu(sim_matrix - margin)  # only penalize when too similar
    #     loss = (penalty * mask).sum() / (mask.sum() + 1e-8)

    #     return loss

    def numerical_contrastive_loss(self, image_degra_features, text_degra_features, neg_texts_features, deg_neg_text_features):
        B, D = image_degra_features.shape
        N_neg = neg_texts_features.size(1)

        text_features_all = torch.cat([
            text_degra_features.unsqueeze(1), neg_texts_features, deg_neg_text_features
        ], dim=1)

        logits_img2text = torch.bmm(text_features_all, image_degra_features.unsqueeze(2)).squeeze(2) / self.temperature
        labels_img2text = torch.zeros(B, dtype=torch.long, device=image_degra_features.device)
        loss_img2text = F.cross_entropy(logits_img2text, labels_img2text)

        logits_text2img = text_degra_features @ image_degra_features.T / self.temperature
        labels_text2img = torch.arange(B, dtype=torch.long, device=image_degra_features.device)
        loss_text2img = F.cross_entropy(logits_text2img, labels_text2img)

        loss = (loss_img2text + loss_text2img) / 2.0
        return loss

    def compute_fcrc_loss(self, image_degra_features, degraded_prompt_features, deg_val, bin_center_features, deg_type, beta=1.0):

        device = image_degra_features.device
        B, D = image_degra_features.shape
        K = bin_center_features.shape[0]

        # Step 1: Find closest bin center for each deg_val
        # (B, K) = |deg_val - bin_center|
        # abs_diffs = torch.abs(deg_val.view(B, 1) - bin_center_features.view(1, K))  # (B, K)
        abs_diffs = torch.abs(deg_val.unsqueeze(1) - bin_center_features)
        bin_idx = torch.argmin(abs_diffs, dim=1)  # (B,)
        selected_text_feat = degraded_prompt_features[torch.arange(B), bin_idx]   # (B, D)

        # Step 2: Cosine similarity matrix (B, B)
        sim_matrix = F.cosine_similarity(
            image_degra_features.unsqueeze(1),  # (B, 1, D)
            selected_text_feat.unsqueeze(0),    # (1, B, D)
            dim=-1
        )  # (B, B)

        # Step 3: Soft contrastive scaling
        sim_exp = torch.exp(sim_matrix / self.temperature)  # (B, B)

        # Step 4: Relative label distance (lambda weighting)
        # dist = torch.abs(deg_val.view(B, 1) - deg_val.view(1, B))  # (B, B)
        # normalize deg_val to [0, 1] within each type
        norm_deg_val = torch.zeros_like(deg_val)
        for t, (low, high) in self.type_ranges.items():
            mask = (str(deg_type) == t)
            norm_deg_val[mask] = (deg_val[mask] - low) / (high - low + 1e-8)
        # compute same type mask
        same_type_mask = (deg_type.view(B, 1) == deg_type.view(1, B)).float()

        # compute pairwise distance
        dist_same = torch.abs(norm_deg_val.view(B, 1) - norm_deg_val.view(1, B))  # [0, 1]
        dist_diff = torch.ones_like(dist_same) * 1.5  # constant distance for different type (larger than max same-type dist)

        # combine both
        dist = same_type_mask * dist_same + (1.0 - same_type_mask) * dist_diff
        
        lambda_weight = dist * beta
        lambda_weight = lambda_weight / (lambda_weight.sum(dim=1, keepdim=True) + 1e-8)  # (B, B)

        # Step 5: Compute FCRC loss
        pos = sim_exp.diag()  # (B,)
        neg = (lambda_weight * sim_exp).sum(dim=1) - lambda_weight.diag() * pos
        loss = -torch.log(pos / (pos + neg + 1e-6)).mean()
        return loss




        # loss = 0.0

        # B = image_degra_features.size(0)
        # device = image_degra_features.device

        # type_keys = list(degraded_prompt_feature.keys())
        # text_feat_list = []

        # for i in range(B):
        #     type_idx = deg_type[i].item()
        #     type_str = type_keys[type_idx]
        #     bin_centers = bin_center_bank[type_str].to(device)  # (K,)
        #     abs_diffs = torch.abs(bin_centers - deg_val[i])     # (K,)
        #     bin_idx = torch.argmin(abs_diffs).item()            # int
        #     text_feat_list.append(degraded_prompt_feature[type_str][bin_idx])  # (D,)

        # text_feat = torch.stack(text_feat_list).to(device)  # (B, D)

        # # Step 1: cosine similarity
        # sim_matrix = F.cosine_similarity(image_degra_features.unsqueeze(1), text_feat.unsqueeze(0), dim=-1)  # (B, B)
        # sim_exp = torch.exp(sim_matrix / self.temperature)  # (B, B)

        # # Step 2: label distance
        # dist = (deg_val.unsqueeze(1) - deg_val.unsqueeze(0)).abs()  # (B, B)
        # lambda_weight = dist * beta
        # lambda_weight = lambda_weight / (lambda_weight.sum(dim=1, keepdim=True) + 1e-8)

        # # Step 3: compute loss
        # pos = sim_exp.diag()  # (B,)
        # neg = (lambda_weight * sim_exp).sum(dim=1) - lambda_weight.diag() * pos
        # loss = -torch.log(pos / (pos + neg + 1e-6)).mean()
        # return loss


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
            degraded_prompt_features=None,
            bin_center_features=None,
            pred=None,
            deg_neg_text_features=None,
            neg_texts_features=None,

    ):
        # CLIP-style contrastive loss
        clip_loss = super().forward(image_features, text_features, logit_scale)

        # GT ↔ image feature alignment loss
        gt_l1_loss = 0.0
        if gt_image_features is not None:
            gt_l1_loss = self.l1_loss_fn(image_features, gt_image_features)
            gt_l1_loss = self.l1_loss_weight * gt_l1_loss

        # regression l1 loss
        reg_ls_loss = 0.0
        if gt_val is not None:
            reg_ls_loss = self.reg_l1_loss_fn(pred, gt_val)

        fcrc_loss = 0.0
        fcrc_loss = self.compute_fcrc_loss(image_degra_features, degraded_prompt_features, gt_val, bin_center_features, deg_type)


        # Text ↔ degradation feature contrastive loss
        num_contrastive_loss = self.numerical_contrastive_loss(
            image_degra_features, text_degra_features, neg_texts_features, deg_neg_text_features
        )

        # intra_type_contrastive_loss = 0.0
        # intra_type_contrastive_loss = self.intra_type_contrastive_loss_weight * self.compute_intra_type_contrastive_loss(image_degra_features, text_degra_features, deg_label, deg_type)

        # inter_type_contrastive_loss = 0.0
        # inter_type_contrastive_loss = self.inter_type_contrastive_loss_weight * self.compute_inter_type_contrastive_loss(image_degra_features, deg_type)

        if output_dict:
            return {
                "contrastive_loss": clip_loss,
                "gt_l1_loss": gt_l1_loss,
                "reg_ls_loss": reg_ls_loss,
                "fcrc_loss": fcrc_loss,
                "num_contrastive_loss": num_contrastive_loss
            }
        # return clip_loss, gt_l1_loss, intra_type_contrastive_loss
        return clip_loss, gt_l1_loss, reg_ls_loss, fcrc_loss, num_contrastive_loss

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
