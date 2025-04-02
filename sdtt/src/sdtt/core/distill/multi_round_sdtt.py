import torch
from transformers import AutoModelForMaskedLM
from loguru import logger
from sdtt.models.loading_utils import get_backbone
import torch.nn.functional as F
import time
from pathlib import Path
import os
import sys
import copy

from sdtt.core.sampling.ancestral import sample_categorical
from sdtt.core.diffusion.absorbing import DiffusionCore
from sdtt.core.sampling import AncestralSampler, AnalyticSampler
from tqdm import trange
from sdtt.data.utils import params2key
from pathlib import Path
import os


from transformers import AutoTokenizer
from ...data import dataloader
from lightning.fabric import Fabric
import lightning as L

from tqdm import trange
import numpy as np
from einops import rearrange
import pandas as pd
from huggingface_hub import create_branch, create_repo, PyTorchModelHubMixin, hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file
from huggingface_hub import login

from collections import deque
import random
import time, statistics

@torch.jit.script
def tv_dist(log_p: torch.Tensor, log_q: torch.Tensor):
    p = log_p.exp()
    q = log_q.exp()
    diff = (p - q).abs()
    loss = diff.sum(-1).mean()
    loss = loss / 2
    return loss


def load_scaling_student(size="sm", round=1):
    if not round in list(range(1, 8)):
        raise ValueError(f"Round value is too large: should be 1 <= round <= 7. Actual value: `{round}`")
    
    if size not in ("sm", "md", "large"):
        raise ValueError(f"Valid model sizes: sm, md, large. Actual value: `{size}`")
    
    revision = f"scaling_400k_{size}_step_{round * 10_000}"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model


def load_small_student(loss="kld", round=1, config=None):
    if not round in list(range(1, 8)):
        raise ValueError(f"Round value is too large: should be 1 <= round <= 7. Actual value: `{round}`")
    
    if loss not in ("kld", "mse", "tvd"):
        raise ValueError(f"Valid losses sizes: kld, mse, tvd. Actual value: `{loss}`")

    revision = f"baselines_{loss}_step_{round * 10_000}"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision, config=config)
    return model


def load_mdlm_small(config=None):
    revision = "teacher_1M_sm"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model

    


def load_scaling_teacher(size="sm"):
    if size not in ("sm", "md", "large"):
        raise ValueError(f"Valid model sizes: sm, md, large. Actual value: `{size}`")
    
    revision = f"teacher_400k_{size}_step_400000"
    model = MultiRoundSDTT.from_pretrained("jdeschena/sdtt", revision)
    return model




class MultiRoundSDTT(DiffusionCore, PyTorchModelHubMixin, AncestralSampler, AnalyticSampler):
    def __init__(self, config, tokenizer, verbose=True):
        DiffusionCore.__init__(self, config, tokenizer)
        AncestralSampler.__init__(self, config)
        AnalyticSampler.__init__(self, config)

        self.neg_infinity = -1000000.0
        self.verbose = verbose

        self.teacher = None
        self.num_distill_steps = self.config.parameterization.num_distill_steps
        self.tot_num_sampl_steps = self.config.parameterization.orig_num_sampling_steps
        self.min_num_sampl_steps = self.config.parameterization.min_num_sampling_steps
        self.distill_mode = self.config.parameterization.distill_mode
        self.start_from_hf = self.config.parameterization.start_from_hf
        self.reset_optimizer_on_growth = (
            config.parameterization.reset_optimizer_on_growth
        )
                
        self.use_ema_on_growth = config.parameterization.use_ema_on_growth

        self.sampling_eps_tensor = torch.tensor(self.sampling_eps)
        self.sampling_mode = self.config.parameterization.sampling_mode
        assert self.sampling_mode in ("ancestral", "analytic")

        self.grow_dt_every = config.parameterization.grow_dt_every

        self.dt = (1 - self.sampling_eps) / self.tot_num_sampl_steps
        self.loss_precision = self.config.parameterization.loss_precision

        mode = self.distill_mode
        self._loss_fn = None  # fn to compare preds & targets
        if mode == "mse":
            self._loss_fn = self._mse
        elif mode == "tvd":
            self._loss_fn = self._tvd
        elif mode == "kl-fwd":
            self._loss_fn = self._fwd_kl
        elif mode == "kl-bwd":
            self._loss_fn = self._bwd_kl
        else:
            raise ValueError(mode)
        if verbose:
            logger.info(f"Distillation loss: {mode}")
        
        if config.mode == "train" and config.is_di4c:
            self.list_cnt = 0
            self.distil_list = deque()
            self.data_list = deque()
            self.data_cor_list = deque()
            self.consis_list = deque()
            self.consis_cor_list = deque()

        self.prepare_teacher_and_student()

    @classmethod
    def from_pretrained(cls, repo, revision, config=None, student_as_teacher=True):
        ckpt_path = hf_hub_download(repo_id=repo, revision=revision, filename="model.safetensors")
        ckpt = load_file(ckpt_path)
        if config is None:
            config_path = hf_hub_download(repo_id=repo, revision=revision, filename="config.json")
            config = OmegaConf.load(config_path)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
        model = cls(config, tokenizer, verbose=False)
        model.load_state_dict(ckpt, strict=False)

        if student_as_teacher:
            if model.use_ema_on_growth:
                # Use EMA as teacher and student, and reset EMA for next round
                model.store_ema()
                student_ckpt = copy.deepcopy(model.backbone.state_dict())
                model.restore_ema()
                model.backbone.load_state_dict(student_ckpt, strict=False)
                model.init_ema()
            else:
                student_ckpt = model.backbone.state_dict()
            model.teacher[0].load_state_dict(student_ckpt)
        
        if "is_teacher_di4c" in config:
            model.teacher[0].is_di4c = config.is_teacher_di4c
        else:
            model.teacher[0].is_di4c = False
        
        return model

    def push_to_hub(self, repo, revision="main", private=True):
        login(token=os.environ["HF_WRITE_KEY"])
        repo = create_repo(repo_id=repo, private=private, exist_ok=True).repo_id
        create_branch(repo, branch=revision, exist_ok=True)

        dict_config = OmegaConf.to_container(self.config)
        PyTorchModelHubMixin.push_to_hub(self, repo_id=repo, branch=revision, private=private, config=dict_config)

    def prepare_teacher_and_student(self):
        """
        If start from hf checkpoint:
            - Load the hf arch in student + teacher
        Else:
            - Init teacher as a copy of student

        if start checkpoint is not kuleshov-group/mdlm-owt -> load from disk

        """
        if self.verbose:
            logger.info("Loading teacher checkpoint...")
        ckpt_path = self.config.parameterization.checkpoint_path

        if self.start_from_hf and ckpt_path == "kuleshov-group/mdlm-owt":
            assert self.config.data_preprocess.legacy_start_end_bos
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                "kuleshov-group/mdlm-owt", trust_remote_code=True
            )

            # Hack so that teacher doesn't get registered as child
            self.teacher = [
                AutoModelForMaskedLM.from_pretrained(
                    "kuleshov-group/mdlm-owt", trust_remote_code=True
                ).eval()
            ]

            if self.config.compile:
                self.teacher[0] = torch.compile(self.teacher[0])

        else:
            self.teacher = [get_backbone(self.config, self.vocab_size)]

        if ckpt_path != "kuleshov-group/mdlm-owt":
            if self.verbose:
                logger.info(f"Loading checkpoint in teacher from `{ckpt_path}`.")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]

            if not self.config.compile:
                ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

            # Note: if loading a distilled version of the original mdlm model (loaded from hf), the
            #   checkpoint has keys starting with backbone.backbone. However, we need to have only one
            #   `backbone.` prefix. In case we are using our models, there should be no `backbone.` prefix
            #   at all, but this is left for later
            ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}

            if self.start_from_hf:
                ckpt = {"backbone." + k: v for k, v in ckpt.items()}

            self.teacher[0].load_state_dict(ckpt, strict=False)
            self.backbone.load_state_dict(ckpt, strict=False)

        self.teacher[0].eval()
        self.teacher[0].requires_grad_(False)
        # Reset EMA to use weights from checkpoint
        self.init_ema()

        if "is_teacher_di4c" in self.config:
            self.teacher[0].is_di4c = self.config.is_teacher_di4c
        else:
            self.teacher[0].is_di4c = False

        if self.verbose:
            logger.info("Teacher checkpoint loaded.")

    def forward(self, xt, cond):
        if not self.time_conditioning:
            cond = torch.zeros_like(cond)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.backbone(xt, cond)
        logits = self._subs_parameterization(logits, xt)
        return logits

    def forward_teacher(self, xt, cond):
        if not self.time_conditioning:
            cond = torch.zeros_like(cond)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            logits = self.teacher[0](xt, cond)

        logits = self._subs_parameterization(logits, xt)
        return logits

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def to(self, device):
        DiffusionCore.to(self, device=device)
        self.teacher[0].to(device=device)

    @torch.no_grad
    def _teacher_logprobs_on_mask(self, xt, t_start):
        """
        Collect teacher predictions for ALL mask tokens
        """
        dt = self.dt

        space = torch.linspace(
            1, 0, self.num_distill_steps, device=t_start.device
        ).double()[:, None]
        t_start = t_start[None, :].double()
        t_end = t_start - dt * self.num_distill_steps
        # Evenly-spaced interpolation between t_start and t_end
        ts = t_start * space + (1 - t_start) * t_end
        # Ensure we don't feed the model values smaller than sampling_eps
        ts = torch.maximum(ts, self.sampling_eps_tensor)

        teacher_predictions = torch.zeros(
            (*xt.shape, self.vocab_size), device=xt.device
        )
        unmasked_tokens = torch.zeros(xt.shape, device=xt.device)
        curr_x = xt

        for idx in range(len(ts)):
            t = ts[idx].float()
            # TODO: add analytic sampler
            if self.sampling_mode == "ancestral":
                log_p_x0, q_xs = self._compute_ddpm_update(
                    curr_x, t, dt, forward=self.forward_teacher
                )
                update = sample_categorical(q_xs)
                new_batch = self._ddpm_sample_update(curr_x, update)

            elif self.sampling_mode == "analytic":
                log_p_x0, new_batch = self._analytic_update(
                    curr_x,
                    t,
                    dt,
                    forward=self.forward_teacher,
                )
            else:
                raise ValueError(self.sampling_mode)

            updated = curr_x != new_batch
            # Extract predictions for denoised tokens
            teacher_predictions[updated] = log_p_x0[updated]
            unmasked_tokens += updated
            curr_x = new_batch

        # Put predictions from model on last step for remaining MASK tokens
        last_preds_update_mask = (curr_x == self.mask_index) * torch.logical_not(
            unmasked_tokens
        )
        last_preds_update_mask = last_preds_update_mask[..., None].to(bool)
        teacher_predictions = torch.where(
            last_preds_update_mask, log_p_x0, teacher_predictions
        )
        return teacher_predictions

    def loss(self, x, t=None, attention_mask=None):
        if attention_mask is not None:
            assert (
                (attention_mask.to(int) == 1).all().item()
            ), "attention mask not supported"

        x0 = x
        if t is None:
            t = self._sample_t(x0.shape[0], x0.device)

        sigma, move_chance, dsigma = self._t_to_sigma(t)
        xt = self.q_xt(x0, move_chance)
        sigma = sigma.squeeze(-1)  # Original shape [bs, 1]
        # Loss on all masked tokens
        teacher_preds = self._teacher_logprobs_on_mask(xt, t)
        student_preds = self.forward(xt, sigma)
        is_mask = xt == self.mask_index

        target = teacher_preds[is_mask]
        preds = student_preds[is_mask]

        if self.loss_precision == "64":
            target = target.to(torch.float64)
            preds = preds.to(torch.float64)
        elif self.loss_precision == "32":
            target = target.to(torch.float32)
            preds = preds.to(torch.float32)

        loss = self._loss_fn(preds, target)
        return loss

    def _mse(self, preds, target):
        return F.mse_loss(preds, target)

    def _tvd(self, preds, target):
        return (preds - target).abs().sum(-1).mean()

    def _fwd_kl(self, preds, target):
        return F.kl_div(preds, target, log_target=True, reduction="batchmean")

    def _bwd_kl(self, preds, target):
        return F.kl_div(target, preds, log_target=True, reduction="batchmean")
    
    def _sample_t_like(self, x0, log_unif=False, distil_prob=0.3):
        t = torch.rand(x0.shape[0]).to(x0.device)
        if log_unif:
            eps_t = 1e-5
            log_t = t * (np.log(1-eps_t) - np.log(eps_t))
            t = eps_t * torch.exp(log_t)
        else:
            delta = self.config.distil_delta
            if np.random.rand() < distil_prob:
                t *= delta
            else:
                t = delta + (1-delta) * t
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)
        return t

    def di4c_loss(self, x0, t=None, attention_mask=None):
        if t is None:
            # t = self._sample_t(x0.shape[0], x0.device)
            t = self._sample_t_like(x0[:1], log_unif=self.config.log_unif)*torch.ones(x0.size(0)).to(self.device)

        sigma, move_chance, dsigma = self._t_to_sigma(t)
        xt = self.q_xt(x0, move_chance)
        unet_t = sigma.squeeze(-1)  # Original shape [bs, 1]

        batch_size = self.config.latent_bsize

        t_rep = t.repeat_interleave(batch_size, dim=0)
        xt_rep = xt.repeat_interleave(batch_size, dim=0)
        self.backbone.eval()
        self.noise.eval()
        dt_rep = torch.minimum(
            torch.ones_like(t_rep)/self.T, t_rep/2)
        _, xs = self._ddpm_update(
            xt_rep, t_rep, dt_rep, forward=self.forward_teacher)
        sigma_s, _, _ = self._t_to_sigma(t_rep - dt_rep)
        unet_s = sigma_s.squeeze(-1)

        use_cv = True # modify later
        device = self.device
        B, D = x0.size(0), x0.size(1)

        with torch.no_grad():
            if self.teacher[0].is_di4c:
                unet_t_rep = unet_t.repeat_interleave(batch_size, dim=0)
                logits_teacher = self.forward_teacher(xt_rep, unet_t_rep).detach()
                p0t_teacher_lam = F.softmax(logits_teacher, dim=2)
                log_p0t_teacher_lam = F.log_softmax(logits_teacher, dim=2) # (B*batch_size, D, S)
                p0t_teacher = p0t_teacher_lam.view(B, batch_size, D, -1).mean(dim=1).detach()
                log_p0t_teacher = torch.logsumexp(log_p0t_teacher_lam.view(B, batch_size, D, -1), dim=1) - np.log(batch_size)
            else:
                logits_teacher = self.forward_teacher(xt, unet_t).detach()
                p0t_teacher = F.softmax(logits_teacher, dim=2) # (B, D, S)
                log_p0t_teacher = F.log_softmax(logits_teacher, dim=2)
            logits_student_teacher = self.forward(xs, unet_s).detach()
        
        self.backbone.train()
        self.noise.train()
        logits_student = self.forward(
            xt.repeat_interleave(batch_size, dim=0),
            unet_t.repeat_interleave(batch_size, dim=0))

        S = logits_student.size(2)
        # batch_size = latent_bsize

        log_p0t_student_lam = F.log_softmax(logits_student, dim=2) # (B*batch_size, D, S)
        # --------------- Datapoint loss --------------------------
        b_idx = torch.arange(B, device=device).repeat_interleave(D)
        d_idx = torch.arange(D, device=device).repeat(B)
        s_idx = x0.view(-1).long()
        dll_ = log_p0t_student_lam.view(B, batch_size, D, S)[
            b_idx,
            :,
            d_idx,
            s_idx].view(B, D, batch_size) # (B, D, batch_size)
        dll = dll_.sum(dim=1) # (B, batch_size)
        data_loss_pointwise = - (torch.logsumexp(dll, dim=1) - np.log(batch_size))
        data_loss_nll = dll_.logsumexp(dim=2) - np.log(batch_size)
        if use_cv:
            data_loss_cor = data_loss_pointwise + data_loss_nll.sum(dim=1)
            log_p0t_student = torch.logsumexp(log_p0t_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
            data_loss_indep = (p0t_teacher * (log_p0t_teacher - log_p0t_student)).view(B, -1).sum(dim=1)
        else:
            data_loss_cor = data_loss_pointwise
            data_loss_indep = 0

        r = 1 - t
        alpha_t = self.config.alpha_t
        if alpha_t == "sigmoid":
            time_coeff = torch.sigmoid(20*(0.5-r))
        elif alpha_t == "linear":
            time_coeff = 1 - r
        else:
            time_coeff = 1.0
        time_coeff *= self.config.alpha_const
        
        data_loss = time_coeff*data_loss_cor + data_loss_indep

        if t < self.config.distil_delta:
            distil_loss = (p0t_teacher.repeat_interleave(batch_size, dim=0) * (
                log_p0t_teacher.repeat_interleave(batch_size, dim=0) - log_p0t_student_lam)).view(B, -1).sum(dim=1) / batch_size # (B,)
            consis_loss = consis_loss_indep = consis_loss_cor = torch.zeros(x0.size(0)).to(self.device)
        else:
            # --------------- Consistency loss --------------------
            with torch.no_grad():
                p0u_student_lam = F.softmax(logits_student_teacher, dim=2)
                log_p0u_student_lam = F.log_softmax(logits_student_teacher, dim=2) # (B*batch_size, D, S)
                p0u_student = p0u_student_lam.view(B, batch_size, D, S).mean(dim=1).detach()
                log_p0u_student = torch.logsumexp(log_p0u_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
                x_0_cat = torch.distributions.categorical.Categorical(p0u_student_lam) # (B*batch_size, D)
                x_0 = x_0_cat.sample().view(B, batch_size, D)
                del x_0_cat
            if use_cv:
                log_p0t_student = torch.logsumexp(log_p0t_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
                consis_loss_indep = (p0u_student * (log_p0u_student.detach() - log_p0t_student)).view(B, -1).sum(dim=1) # (B,)
            else:
                consis_loss_indep = 0
            consis_loss_cor = 0
            for i in range(x_0.shape[1]):
                b_idx = torch.arange(B, device=device).repeat_interleave(D)
                d_idx = torch.arange(D, device=device).repeat(B)
                s_idx = x_0[:,i].reshape(-1).long()
                cll_ = log_p0t_student_lam.view(B, batch_size, D, S)[
                    b_idx,
                    :,
                    d_idx,
                    s_idx].view(B, D, batch_size) # (B, D, batch_size)
                cll = cll_.sum(dim=1) # (B, batch_size)
                if use_cv:
                    cll = torch.logsumexp(cll, dim=1) - np.log(batch_size) - (cll_.logsumexp(dim=2) - np.log(batch_size)).sum(dim=1)
                else:
                    cll = torch.logsumexp(cll, dim=1) - np.log(batch_size)
                consis_loss_cor -= cll / x_0.shape[1] # (B,)
            consis_loss = consis_loss_cor + consis_loss_indep # (B,)

            distil_loss = torch.zeros(x0.size(0)).to(self.device)

        loss = torch.mean(distil_loss + data_loss + consis_loss)

        self.list_cnt += 1
        self.distil_list.append(torch.mean(distil_loss).item())
        self.data_list.append(torch.mean(data_loss).item())
        self.data_cor_list.append(torch.mean(data_loss_cor).item())
        self.consis_list.append(torch.mean(consis_loss).item())
        self.consis_cor_list.append(torch.mean(consis_loss_cor).item())
        if len(self.distil_list) > 100:
            self.distil_list.popleft()
            self.data_list.popleft()
            self.data_cor_list.popleft()
            self.consis_list.popleft()
            self.consis_cor_list.popleft()
        if self.list_cnt % 50 == 0 and self.trainer.global_rank == 0:
            print(np.mean(self.distil_list), np.mean(self.data_list), np.mean(self.data_cor_list), np.mean(self.consis_list), np.mean(self.consis_cor_list))

    
        return loss

    def training_step(self, batch):
        if self.ema is not None:
            assert (
                not self._using_ema_weights
            ), "SHOULD NOT USE EMA WEIGHTS DURING TRAINING!!!"
        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        if self.config.is_di4c:
            loss = self.di4c_loss(x, attention_mask=attention_mask)

        else:
            step = self.trainer.global_step
            if step > 0 and step % self.grow_dt_every == 0:
                curr_round = step // self.grow_dt_every
                self.dt = (
                    (1 - self.sampling_eps) / self.tot_num_sampl_steps * (self.num_distill_steps**curr_round)
                )
                effective_num_steps = round(1 / self.dt)
                if effective_num_steps < self.min_num_sampl_steps:
                    logger.info(
                        f"Reached below the minimal effective number of sampling steps, stopping..."
                    )
                    sys.exit()
                else:
                    logger.info(
                        f"Step {step}: Doubling `dt`! New effective number of steps: {effective_num_steps}."
                    )
                self._student_to_teacher()
                if self.reset_optimizer_on_growth:
                    logger.info("Resetting optimizers...")
                    self.trainer.strategy.setup_optimizers(self.trainer)

            loss = self.loss(x, attention_mask=attention_mask)
        
        self.log(
            name="train/loss",
            value=loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def _student_to_teacher(
        self,
    ):
        start = time.perf_counter()
        if self.use_ema_on_growth:
            # Use EMA as teacher and student, and reset EMA for next round
            self.store_ema()
            student_ckpt = copy.deepcopy(self.backbone.state_dict())
            self.restore_ema()
            self.backbone.load_state_dict(student_ckpt)
            self.init_ema()
        else:
            student_ckpt = self.backbone.state_dict()

        self.teacher[0].load_state_dict(student_ckpt)
        end = time.perf_counter()

        logger.info(f"Swapped student into teacher in {end - start:.2f} seconds.")
        save_path = (
            Path(os.getcwd())
            / "student_checkpoints"
            / f"{self.trainer.global_step}.ckpt"
        )
        self.trainer.save_checkpoint(save_path)

    @torch.no_grad()
    def sample(
        self,
        n_samples=8,
        num_steps=256,
        seq_len=1024,
        sampler="ancestral",
        cache_preds=False,
        verbose=False,
        add_bos=False,
        add_eos=False,
        project_fn=lambda x: x,
    ):
        assert not cache_preds, "Not implemented"
        if cache_preds:
            assert (
                not self.config.time_conditioning
            ), "Cannot use caching with time-conditional network"

        assert sampler in ("ancestral", "analytic")
        if seq_len is None:
            seq_len = self.config.model.length

        batch = self._sample_prior(n_samples, seq_len)
        batch = project_fn(batch)

        if add_bos:
            batch[:, 0] = self.tokenizer.bos_token_id

        if add_eos:
            batch[:, -1] = self.tokenizer.eos_token_id

        # +1 because we use the last value for denoising
        ts = torch.linspace(1.0, self.sampling_eps, steps=num_steps + 1)
        dt = (1 - self.sampling_eps) / num_steps

        for i in trange(num_steps, desc="sampling...", disable=not verbose):
            t = ts[i] * torch.ones(n_samples, 1, device=self.device)
            if sampler == "ancestral":
                _, new_batch = self._ddpm_update(batch, t, dt)
            elif sampler == "analytic":
                _, new_batch = self._analytic_update(batch, t, dt)
            new_batch = project_fn(new_batch)
            # If no caching or an update was made, remove cache
            # if not cache_preds or not torch.allclose(new_batch, batch):
            #    cache = None
            batch = new_batch

        # Denoise
        if (batch == self.mask_index).any():
            t = ts[-1] * torch.ones(n_samples, 1, device=self.device)
            _, batch = self._ddpm_update(
                batch, t, dt, denoise=True, mask_idx=self.mask_index
            )
            batch = project_fn(batch)

        return batch


def sample_uncond(module):
    logger.info("Starting unconditional sampling.")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    uncond_cfg = sampling_cfg.uncond

    ckpt_name = config.checkpointing.resume_ckpt_path.split("/")[-1]
    metadata = dict(
        num_samples=uncond_cfg.num_samples,
        from_ema=uncond_cfg.from_ema,
        num_steps=uncond_cfg.num_steps,
        seq_len=uncond_cfg.seq_len,
        sampler=uncond_cfg.sampler,
        add_bos=uncond_cfg.add_bos,
        add_eos=uncond_cfg.add_eos,
        checkpoint_name=ckpt_name,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "uncond" / save_fname
    assert not save_path.exists(), save_fname

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(100 + fabric.global_rank)
    # Note: the next line creates a bug when calling functions from the module
    # pl_module = fabric.setup(module)
    pl_module = module
    fabric.to_device(pl_module)

    bs = uncond_cfg.batch_size
    num_steps = uncond_cfg.num_steps
    seq_len = uncond_cfg.seq_len
    target_num_samples = uncond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    n_sampling_rounds = target_num_samples // (tot_num_device * bs)

    if uncond_cfg.from_ema:
        pl_module.store_ema()

    latencies = []
    all_samples = []
    for _ in trange(
        n_sampling_rounds,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        with fabric.autocast():
            # start_time = time.perf_counter()
            out = pl_module.sample(
                n_samples=bs,
                num_steps=num_steps,
                seq_len=seq_len,
                sampler=uncond_cfg.sampler,
                add_bos=uncond_cfg.add_bos,
                add_eos=uncond_cfg.add_eos,
                cache_preds=uncond_cfg.cache_preds,
            )
            # end_time = time.perf_counter()
        out = fabric.all_gather(data=out)
        if fabric.global_rank == 0:
            if out.ndim == 3:  # ndim == 2 when running on one device
                out = rearrange(out, "dev bs l -> (dev bs) l")
            all_samples.append(out.cpu())
        del out
        # latencies.append(end_time - start_time)
    
    # avg_latency = statistics.mean(latencies[1:])
    # std_latency = statistics.stdev(latencies[1:])
    # print(f"Average Latency: {avg_latency:.4f} ± {std_latency:.4f} seconds")

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez(save_path, samples=all_samples, metadata=metadata)
        logger.info(f"Saved {len(all_samples)} samples in {save_path}")
    
    # print(decode_tokens_one_by_one(all_samples[0], pl_module.tokenizer))

    # Restore orig model weights
    if uncond_cfg.from_ema:
        pl_module.restore_ema()

def decode_tokens_one_by_one(tokens, tokenizer):
    decoded_text = ""
    for token in tokens:
        if isinstance(token, (list, np.ndarray)):
            current_token = token[0] if len(token) > 0 else token
        else:
            current_token = token
        decoded = tokenizer.decode([current_token])
        decoded_text += decoded
    
    return decoded_text


def sample_cond_prefix(module):
    logger.info("Starting conditional sampling (cond on prefix).")
    config = module.config
    sampling_cfg = config.parameterization.sampling
    cond_cfg = sampling_cfg.cond_prefix

    ckpt_name = config.checkpointing.resume_ckpt_path.split("/")[-1]
    metadata = dict(
        checkpoint_name=ckpt_name,
        num_samples=cond_cfg.num_samples,
        from_ema=cond_cfg.from_ema,
        dataset=cond_cfg.dataset,
        seq_len=cond_cfg.seq_len,
        prefix_len=cond_cfg.prefix_len,
        num_cont_per_prefix=cond_cfg.num_cont_per_prefix,
        min_seq_len=cond_cfg.min_seq_len,
        num_steps=cond_cfg.num_steps,
        sampler=cond_cfg.sampler,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    save_fname = params2key(**metadata) + ".npz"
    save_path = Path(os.getcwd()) / "samples" / "cond" / save_fname
    assert not save_path.exists(), save_fname
    # Extract args from cfg
    bs = cond_cfg.batch_size
    prefix_len = cond_cfg.prefix_len
    num_steps = cond_cfg.num_steps
    seq_len = cond_cfg.seq_len
    target_num_samples = cond_cfg.num_samples
    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert target_num_samples % (tot_num_device * bs) == 0
    # Load prefix dataset
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

    fabric = Fabric(
        accelerator=config.trainer.accelerator,
        precision=config.trainer.precision,
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
    )
    fabric.launch()
    L.seed_everything(200 + fabric.global_rank)

    if fabric.global_rank > 0:
        fabric.barrier()  # Make sure that only the first device does the preprocessing

    dataset = dataloader.get_dataset(
        cond_cfg.dataset,
        tokenizer,
        mode="valid",
        cache_dir=config.data_preprocess.data_cache,
        num_proc=config.trainer.devices * config.loader.num_workers,
        min_seq_len=cond_cfg.min_seq_len,
        seq_len=seq_len,
        group_text=False,
        remove_text=True,
        add_bos=cond_cfg.add_bos,
        add_eos=cond_cfg.add_eos,
    )

    if fabric.global_rank == 0:
        fabric.barrier()  # Make sure the data was preprocessed on one device before starting

    assert len(dataset) >= target_num_samples
    dataset = dataset.select(range(cond_cfg.num_samples))

    pl_module = module
    fabric.to_device(pl_module)

    if cond_cfg.from_ema:
        pl_module.store_ema()

    all_samples = []
    start = fabric.global_rank * bs
    stop = target_num_samples
    end = fabric.world_size * bs
    for idx in trange(
        start,
        stop,
        end,
        desc=f"Sampling with n_steps={num_steps}, seq_len={seq_len}",
        disable=fabric.global_rank > 0,
    ):
        docs = dataset[idx : idx + bs]["input_ids"]
        print_abstract = False
        if print_abstract:
            input_sentence = pl_module.tokenizer.encode("Diffusion models have demonstrated exceptional performances in various fields of generative modeling. While they often outperform competitors including VAEs and GANs in sample quality and diversity, they suffer from slow sampling speed due to their iterative nature. Recently, distillation techniques and consistency models are mitigating this issue in continuous domains, but discrete diffusion models have some specific challenges towards faster generation. Most notably, in the current literature, correlations between different dimensions (pixels, locations) are ignored, both by its modeling and loss functions, due to computational limitations.")
            input_sentence = torch.tensor(input_sentence, dtype=torch.long)
            docs[0] = input_sentence[:len(docs[0])]

        prefixes = docs[:, :prefix_len]

        def project_fn(batch):
            batch[:, :prefix_len] = prefixes
            return batch

        # Generate potentially multiple continuations per prefix (typically 5)
        for _ in range(cond_cfg.num_cont_per_prefix):
            with fabric.autocast():
                out = pl_module.sample(
                    n_samples=bs,
                    num_steps=num_steps,
                    seq_len=seq_len,
                    sampler=cond_cfg.sampler,
                    add_bos=cond_cfg.add_bos,
                    add_eos=cond_cfg.add_eos,
                    cache_preds=cond_cfg.cache_preds,
                    project_fn=project_fn,
                )
            out = fabric.all_gather(data=out)
            if fabric.global_rank == 0:
                # unstack after all_gather
                if out.ndim == 3:
                    out = rearrange(out, "dev bs l -> (dev bs) l")
                all_samples.append(out.cpu())
            del out
        
    # print(pl_module.tokenizer.batch_decode(all_samples[0][:1]))
    # print(pl_module.tokenizer.batch_decode(all_samples[1][:1]))
    # print(pl_module.tokenizer.batch_decode(all_samples[2][:1]))

    if cond_cfg.calc_sbleu:
        sbleu = calc_cond_self_bleu(all_samples, pl_module.tokenizer, num_cont=cond_cfg.num_cont_per_prefix)
        logger.info(f"Conditional self-bleu: {sbleu}")

    # Join and save to disk
    if fabric.global_rank == 0:
        all_samples = torch.cat(all_samples, dim=0).numpy()
        all_samples = all_samples[:target_num_samples * cond_cfg.num_cont_per_prefix]

        save_path.parent.mkdir(exist_ok=True, parents=True)
        references = dataset[:target_num_samples]["input_ids"].numpy()
        np.savez(
            save_path, samples=all_samples, references=references, metadata=metadata
        )
        logger.info(f"Saved samples in {save_path}")

    if cond_cfg.from_ema:
        pl_module.restore_ema()


def _eval_suffix_nll_generators(module: MultiRoundSDTT, config, prefix: torch.Tensor, suffix):
    N = len(suffix)
    device = module.device
    batch_size = config.eval.lambada_openai.batch_size
    num_samples = config.eval.lambada_openai.num_samples
    add_eos = config.eval.lambada_openai.add_eos
    assert num_samples % batch_size == 0

    all_t = module._sample_t(num_samples, device=module.device)
    full_sentence = torch.cat([prefix, suffix], dim=-1, ).repeat(batch_size, 1).to(module.device)

    for idx in range(0, num_samples, batch_size):
        curr_t = all_t[idx: idx + batch_size]
        sigma, move_chance, dsigma = module._t_to_sigma(curr_t)
        sigma = sigma.squeeze(-1)

        xt = module.q_xt(full_sentence, move_chance)
        xt[:, :len(prefix)] = full_sentence[:, :len(prefix)]
        if add_eos:
            xt[:, -1] = full_sentence[:, -1]

        y = full_sentence
        scale = (dsigma / torch.expm1(sigma))[:, None]

        yield xt.to(device), y.to(device), scale.to(device), sigma.to(device), curr_t.to(device)


@torch.no_grad
def eval_suffix_nll(config, module: MultiRoundSDTT, prefix, suffix, sigma):
    """
    1. Generate all ways to mask the suffix.
    2. Evaluate the loss over all possible maskings
    3. Average over all possible masking
    """

    all_losses = []
    for xt, y, scale, sigma, t in _eval_suffix_nll_generators(module, config, prefix, suffix):
        preds = module(xt, sigma).log_softmax(-1)

        loss = - torch.gather(preds, dim=-1, index=y[..., None])[..., 0]
        is_masked = xt == module.mask_index
        loss = torch.where(is_masked.to(bool), loss, 0.0) * scale

        loss = loss.sum(-1)
        loss = loss.mean()
        all_losses.append(float(loss))

    return float(np.mean(all_losses))


@torch.no_grad
def eval_lambada(module: MultiRoundSDTT):
    logger.info("Starting eval acc/ppl on openai lambada")
    config = module.config
    lambada_cfg = config.eval.lambada_openai

    if config.eval.lambada_openai.from_ema:
        module.store_ema()

    tokenizer = module.tokenizer

    dataset = dataloader.get_dataset(
        "EleutherAI/lambada_openai",
        tokenizer,
        mode="test",
        cache_dir=config.data_preprocess.data_cache,
        num_proc=config.trainer.devices * config.loader.num_workers,
        group_text=False,
        remove_text=False,
        add_bos=lambada_cfg.add_bos,
        add_eos=lambada_cfg.add_eos,
    )

    tot_num_device = config.trainer.num_nodes * config.trainer.devices
    assert tot_num_device == 1, "Code only works with one device"

    pl_module = module
    pl_module = pl_module.cuda()
    t = torch.tensor([pl_module.sampling_eps], device="cuda")
    sigma = pl_module._t_to_sigma(t)[0][0]

    all_losses = []
    all_last_correct = []
    add_eos = lambada_cfg.add_eos

    for idx in trange(
        len(dataset),
        desc="Evaluating lambada..."
    ):
        prefix = dataset[idx]["prefix_ids"]
        suffix = dataset[idx]["suffix_ids"]
        suffix_mask = suffix.clone()
        if add_eos:
            suffix_mask[:-1] = pl_module.mask_index
        else:
            suffix_mask[:] = pl_module.mask_index

        input_ids = torch.cat([prefix, suffix_mask]).cuda().reshape(1, -1)
        preds = pl_module(input_ids, sigma)

        assert pl_module.mask_index == preds.shape[-1] - 1
        greedy_tokens = preds[0, :, :-1].argmax(-1)
        suff_len = len(suffix)

        if add_eos:
            correct = greedy_tokens[-suff_len:-1].cpu() == suffix[:-1]
            correct = correct.all().item()

            loss = eval_suffix_nll(config, pl_module, prefix, suffix, sigma)

            all_losses.append(loss)
            all_last_correct.append(correct)

        else:
            raise NotImplementedError

    acc = np.mean(all_last_correct)
    avg_loss = np.mean(all_losses)

    from run_eval import CURR_DATETIME_STR
    csv_save_path = Path(os.getcwd()) / "csv" / CURR_DATETIME_STR / "lambada.csv"
    header = [
        "num_samples",
        "from_ema",
        "add_bos",
        "add_eos",
        "checkpoint_path",
        "acc",
        "ppl",
    ]

    row = [
        lambada_cfg.num_samples,
        lambada_cfg.from_ema,
        lambada_cfg.add_bos,
        lambada_cfg.add_eos,
        config.checkpointing.resume_ckpt_path,
        float(acc),
        float(np.exp(avg_loss)),
    ]

    df = pd.DataFrame([row], columns=header)
    csv_save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_save_path)
    logger.info(f"Lambada results: \n{df}\n{'=' * 50}")

    if config.eval.lambada_openai.from_ema:
        module.restore_ema()

import os
from multiprocessing import Pool
import nltk
from nltk.translate.bleu_score import SmoothingFunction

def calc_cond_self_bleu(samples, tokenizer, num_cont):
    variations = []
    cond_sbleu = []
    for j in range(num_cont):
        variations.append(torch.cat(samples[j::num_cont], dim=0))
    for i in trange(variations[0].size(0), desc='Calculating conditional self-bleu...'):
        sentences = []
        for j in range(num_cont):
            sentences.append(tokenizer.batch_decode(variations[j][i]))
        sbleu = SelfBleu()
        sbleu.reference = sentences
        cond_sbleu.append(sbleu.get_score())
    return np.mean(cond_sbleu)

class SelfBleu:
    def __init__(self, test_text='', gram=4, scaling_up=True, is_tqdm=False):
        # super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True
        self.scale = 100 if scaling_up else 1
        self.is_tqdm = is_tqdm

    def get_name(self):
        return self.name

    def get_score(self, is_fast=False, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.scale * self.get_bleu_fast()
        return self.scale * self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        # pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        if self.is_tqdm:
            tlist = trange(sentence_num, desc='Calculating self-bleu...')
        else:
            tlist = range(sentence_num)
        for index in tlist:
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(self.calc_bleu(other, hypothesis, weight))

        return np.mean(result)

