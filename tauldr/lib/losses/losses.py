import torch
import torch.nn as nn
import lib.losses.losses_utils as losses_utils
import math
import numpy as np
import torch.autograd.profiler as profiler
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lib.sampling.sampling import get_initial_samples, one_step_sampling

@losses_utils.register_loss
class DistilChain():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.cross_ent = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.distil_qmc = False
        self.time_schedule = None
        self.min_distil_time = 0.001
        self.max_distil_time = 0.01

        self.use_cv = True
    

    def calc_loss(self, minibatch, state, writer=None, eval=False):
        # dum = True
        # if dum:
        #     loss = self.calc_nll(minibatch, state['model'], state['distil_batch_size'])
        #     print(loss.item())
        #     return loss

        teacher = state['teacher']
        model = state['model'] # model with lams
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = teacher.device
        batch_size = state['distil_batch_size']
        if B * batch_size > 128:
            new_B = 128 // batch_size
            new_idx = torch.randperm(B,device=device)[:new_B]
            minibatch = minibatch[new_idx]
            B = new_B

        delta_t = torch.exp(torch.rand((B,), device=device) * np.log(self.max_distil_time / self.min_distil_time)) * self.min_distil_time
        # ts = torch.exp(torch.rand((B,), device=device) * (-torch.log(self.min_time + delta_t))) * (self.min_time + delta_t)
        ts = torch.rand((B,), device=device) * (1.0 - self.min_time - delta_t) + self.min_time + delta_t
        us = ts - delta_t

        qt0 = teacher.transition(ts) # (B, S, S)
        # --------------- Sampling x_t, x_u --------------------
        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.flatten().long(),
            :
        ] # (B*D, S)
        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        x_logits_teacher = teacher(x_t, ts) # (B, D, S)
        p0t_teacher = F.softmax(x_logits_teacher, dim=2).detach() # (B, D, S)
        log_p0t_teacher = F.log_softmax(x_logits_teacher, dim=2).detach()

        lams = torch.rand((B*batch_size, model.lam_raw_dim), device=device)
        # more generally, sample according to the distribution of lams
        x_t_rep = x_t.repeat_interleave(batch_size, dim=0)
        ts_rep = ts.repeat_interleave(batch_size)
        x_logits_student_lam = model(x_t_rep, ts_rep, lams) # (B*batch_size, D, S)
        log_p0t_student_lam = F.log_softmax(x_logits_student_lam, dim=2) # (B*batch_size, D, S)
        p0t_student_lam = F.softmax(x_logits_student_lam, dim=2) # (B*batch_size, D, S)
        us_rep = us.repeat_interleave(batch_size)

        # --------------- Datapoint loss --------------------------

        b_idx = torch.arange(B, device=device).repeat_interleave(D)
        d_idx = torch.arange(D, device=device).repeat(B)
        s_idx = minibatch.view(-1).long()
        dll_ = log_p0t_student_lam.view(B, batch_size, D, S)[
                        b_idx,
                        :,
                        d_idx,
                        s_idx
                    ].view(B, D, batch_size) # (B, D, batch_size)
        dll = dll_.sum(dim=1) # (B, batch_size)
        if self.use_cv:
            dll = torch.logsumexp(dll, dim=1) - np.log(batch_size) - (dll_.logsumexp(dim=2) - np.log(batch_size)).sum(dim=1)
            data_loss_cor = - dll # (B,)
            log_p0t_student = torch.logsumexp(log_p0t_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
            data_loss_indep = (p0t_teacher * (log_p0t_teacher - log_p0t_student)).view(B, -1).sum(dim=1)
        else:
            data_loss_cor = - (torch.logsumexp(dll, dim=1) - np.log(batch_size))
            data_loss_indep = 0

        # --------------- Distillation loss --------------------

        # sampling x_{min_t}
        mints = self.min_time + torch.randint(2, (B*batch_size,), device=device) * torch.rand((B*batch_size,), device=device) * self.max_distil_time

        x_0_for_mint = minibatch.repeat_interleave(batch_size, dim=0) # (B*b, D)
        qmint0 = teacher.transition(mints) # (B*b, S, S)
        qmint0_rows_reg = qmint0[torch.arange(B*batch_size, device=device).repeat_interleave(D),
            x_0_for_mint.flatten().long(), :] # (B*b*D, S)
        x_mint_cat = torch.distributions.categorical.Categorical(qmint0_rows_reg)
        x_mint = x_mint_cat.sample().view(B*batch_size, D)

        p0mint_teacher = F.softmax(teacher(x_mint, mints), dim=2).detach()
        log_p0mint_teacher = F.log_softmax(teacher(x_mint, mints), dim=2).detach()
        
        # add p_{0|t_min} distillation for each lambda
        mlams = torch.rand((B*batch_size, model.lam_raw_dim), device=device)
        log_p0mint_student_lam = F.log_softmax(model(x_mint, mints, mlams), dim=2)

        distil_loss = (p0mint_teacher * (log_p0mint_teacher - log_p0mint_student_lam)).view(B, -1).sum(dim=1) / batch_size # (B,)

        # --------------- Consistency loss --------------------
        
        # sample time u
        # us = torch.rand((B,), device=device) * (ts - self.min_time) + self.min_time
        
        # sample again
        # chain_with_teacher = True
        # if chain_with_teacher:
        x_u, _ = self.calc_pst(
            teacher, x_t_rep,
            p0t_teacher.repeat_interleave(batch_size, dim=0),
            ts_rep, us_rep,
            sample=True
        )
        # else:
        #     lams_ = torch.rand((B*batch_size, model.lam_raw_dim), device=device)
        #     x_logits_student_ = model(x_t_rep, ts_rep, lams_) # (B*batch_size, D, S)
        #     p0t_student_lam_ = F.softmax(x_logits_student_, dim=2) # (B*batch_size, D, S)
        #     # then sample x_u | x_t from the student model
        #     x_u, _ = self.calc_pst(
        #         teacher, x_t_rep,
        #         p0t_student_lam_,
        #         ts_rep, us_rep,
        #         sample=True
        #     ) # (B*batch_size, D)
        
        lams_u = torch.rand((B*batch_size, model.lam_raw_dim), device=device)
        x_u_logits_student = model(x_u, us_rep, lams_u) # (B*batch_size, D, S)
        
        # idx_teacher = us_rep < self.min_time + self.max_distil_time
        # x_u_logits_teacher = teacher(x_u[idx_teacher], us_rep[idx_teacher])
        # x_u_logits_student[idx_teacher] = x_u_logits_teacher

        p0u_student_lam = F.softmax(x_u_logits_student, dim=2)
        log_p0u_student_lam = F.log_softmax(x_u_logits_student, dim=2) # (B*batch_size, D, S)
        p0u_student = p0u_student_lam.view(B, batch_size, D, S).mean(dim=1).detach()
        log_p0u_student = torch.logsumexp(log_p0u_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)

        # with torch.no_grad():
        #     x_u_logits_teacher = teacher(x_u, us_rep)
        #     p0u_teacher_lam = F.softmax(x_u_logits_teacher, dim=2)

        # marginal_consis_loss = self.kl_loss(log_p0u_student, p0t_teacher)
        # log_p0t_student_mean = torch.mean(log_p0t_student_lam.view(B, batch_size, D, S), dim=1)
        # log_p0u_student_mean = torch.mean(log_p0u_student_lam.view(B, batch_size, D, S), dim=1)
        # consis_loss_indep = self.kl_loss(log_p0t_student_mean, p0u_student.detach())
        # - torch.sum(p0u_student.detach() * (log_p0t_student_mean - (p0u_student.detach()+self.ratio_eps).log())) / B

        # consis_loss_indep = self.kl_loss(log_p0t_student, p0u_student.detach())
        if self.use_cv:
            consis_loss_indep = (p0u_student * (log_p0u_student.detach() - log_p0t_student)).view(B, -1).sum(dim=1) # (B,)
        else:
            consis_loss_indep = 0
        
        x_0_cat = torch.distributions.categorical.Categorical(p0u_student_lam.detach()) # (B*batch_size, D)
        x_0 = x_0_cat.sample().view(B, batch_size, D)

        consis_loss_cor = 0
        
        for i in range(x_0.shape[1]):
            b_idx = torch.arange(B, device=device).repeat_interleave(D)
            d_idx = torch.arange(D, device=device).repeat(B)
            s_idx = x_0[:,i].reshape(-1).long()
            cll_ = log_p0t_student_lam.view(B, batch_size, D, S)[
                            b_idx,
                            :,
                            d_idx,
                            s_idx
                        ].view(B, D, batch_size) # (B, D, batch_size)
            cll = cll_.sum(dim=1) # (B, batch_size)
            if self.use_cv:
                cll = torch.logsumexp(cll, dim=1) - np.log(batch_size) - (cll_.logsumexp(dim=2) - np.log(batch_size)).sum(dim=1)
            else:
                cll = torch.logsumexp(cll, dim=1) - np.log(batch_size)
            consis_loss_cor -= cll / x_0.shape[1] # (B,)

        # consis_loss_cor = consis_loss_cor.mean()
        consis_loss = consis_loss_cor + consis_loss_indep # (B,)

        time_coeff = torch.ones_like(ts)
        if self.time_schedule is not None:
            if self.time_schedule == 'zero':
                time_coeff = 0.
            elif self.time_schedule == 'sigmoid':
                time_coeff = F.sigmoid(20*(ts-0.5))
            else:
                time_coeff = ts
        data_loss = time_coeff * data_loss_cor + data_loss_indep # (B,)

        if eval:
            return data_loss.mean(), distil_loss.mean(), consis_loss.mean(), consis_loss_cor.mean()

        return torch.mean(distil_loss + data_loss + consis_loss)
        # return torch.mean(10*torch.exp(-20*us)*distil_loss + data_loss + consis_loss)

    def calc_pst(self, model, x_t, p0t, ts, ss, sample=False, x_0=None, log_prob=False, sample_zero=False):
        # model <- teacher network
        # x_t: (B, D)
        # p0t: (B, D, S)
        # ts, ss: (B,)
        B, D, S = p0t.shape
        device = model.device

        qt0 = model.transition(ts) + self.ratio_eps # (B, S, S)
        qs0 = model.transition(ss) + self.ratio_eps # (B, S, S)
        qts = model.transition(ts, ss) + self.ratio_eps # (B, S, S)
        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_t.flatten().long()
        ] # (B*D, S)

        qts_rows_reg = qts[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_t.flatten().long()
        ] # (B*D, S)

        if sample:
            if x_0 is None:
                x_0_cat = torch.distributions.categorical.Categorical(p0t.detach()) # (B, D)
                x_0 = x_0_cat.sample()
            if sample_zero:
                return x_0
            qt0_denom = qt0_rows_reg[torch.arange(B*D,
                device=device), x_0.flatten().long()].view(B*D, 1)
            qt0_denom = qt0_denom.expand(-1, S)
            qs0_rows_reg = qs0[
                torch.arange(B, device=device).repeat_interleave(D),
                x_0.flatten().long(),
                :
            ] # (B*D, S)
            ps_0t = qts_rows_reg / qt0_denom * qs0_rows_reg # (B*D, S)
            ps_0t = ps_0t + self.ratio_eps
            ps_0t = ps_0t / ps_0t.sum(dim=-1).view(-1, 1).expand(-1, S)
            x_s_cat = torch.distributions.categorical.Categorical(ps_0t)
            x_s = x_s_cat.sample().view(B, D)
            return x_s, ps_0t
        else:
            if log_prob:
                log_p0t = p0t
                qt0_denom = qt0_rows_reg.view(B, D, S) + self.ratio_eps
                log_ret0 = log_p0t - qt0_denom.log()
                log_rets = torch.zeros((B, D, S), device=device)
                log_rets = torch.log(log_ret0.exp() @ qs0 + self.ratio_eps)
                qts_numer = qts_rows_reg.view(B, D, S) + self.ratio_eps
                return log_rets + qts_numer.log()
            else:
                qt0_denom = qt0_rows_reg.view(B, D, S)
                ret0 = p0t / qt0_denom
                rets = torch.matmul(ret0, qs0)
                qts_numer = qts_rows_reg.view(B, D, S)
                ps_0t = rets * qts_numer
                ps_0t = ps_0t / ps_0t.sum(dim=-1)[:,:,None].expand(-1, -1, S)
                return ps_0t
        
    def calc_nll(self, minibatch, model, distil_batch_size):
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        batch_size = distil_batch_size
        device = model.device
        if B * distil_batch_size > 128:
            new_B = 128 // distil_batch_size
            new_idx = torch.randperm(B,device=device)[:new_B]
            minibatch = minibatch[new_idx]
            B = new_B

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time
        qt0 = model.transition(ts) # (B, S, S)
        # --------------- Sampling x_t --------------------
        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.flatten().long(),
            :
        ] # (B*D, S)
        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        lams = torch.rand((B*batch_size, model.lam_raw_dim), device=device)
        # more generally, sample according to the distribution of lams
        x_t_rep = x_t.repeat_interleave(batch_size, dim=0)
        ts_rep = ts.repeat_interleave(batch_size)
        x_logits_student = model(x_t_rep, ts_rep, lams) # (B*batch_size, D, S)
        # x_logits_student = model(x_t_rep, ts_rep)
        log_p0t_student_lam = F.log_softmax(x_logits_student, dim=2) # (B*batch_size, D, S)

        b_idx = torch.arange(B, device=device).repeat_interleave(D)
        d_idx = torch.arange(D, device=device).repeat(B)
        s_idx = minibatch.view(-1).long()
        ll = log_p0t_student_lam.view(B, batch_size, D, S)[
                        b_idx,
                        :,
                        d_idx,
                        s_idx
                    ].view(B, D, batch_size) # (B, D, batch_size)
        ll = ll.sum(dim=1) # (B, batch_size)
        ll = torch.logsumexp(ll, dim=1) - np.log(batch_size)

        return - ll.mean()
    

@losses_utils.register_loss
class GenericAux():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.cross_ent = nn.CrossEntropyLoss()


    def calc_loss(self, minibatch, state, writer):
        model = state['model']
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = model.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(ts) # (B, S, S)

        rate = model.rate(ts) # (B, S, S)


        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.flatten().long(),
            :
        ] # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            x_t.long().flatten(),
            :
        ] # (B*D, S)
        rate_vals_square[
            torch.arange(B*D, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, D)


        # ---------- First term of ELBO (regularization) ---------------


        if self.one_forward_pass:
            x_logits = model(x_tilde, ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_tilde
        else:
            x_logits = model(x_t, ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_t

        # For (B, D, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B,D,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, D, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )



        # ----- second term of continuous ELBO (signal term) ------------

        
        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            p0t_sig = F.softmax(model(x_tilde, ts), dim=2) # (B, D, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, D)



        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_tilde.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, D, S)


        x_tilde_mask = torch.ones((B,D,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,D,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,D,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, D, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, D) + self.ratio_eps



        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)

        reg_mean = torch.mean(reg_term)


        writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
        writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])


        neg_elbo = sig_mean + reg_mean



        perm_x_logits = torch.permute(x_logits, (0,2,1))

        nll = self.cross_ent(perm_x_logits, minibatch.long())

        return neg_elbo + self.nll_weight * nll

@losses_utils.register_loss
class CChain(GenericAux):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.chain_length = 0.0005
        self.chain_batch_size = 128
        self.preservation_coeff = 1
        self.consistency_coeff = 10
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def calc_loss(self, minibatch, state, writer):
        loss = self.preservation_coeff * super().calc_loss(minibatch, state, writer)

        if state['n_iter'] % state['cchain_freq'] == 0:
            chain_loss = self.consistency_coeff * self.chain_loss(minibatch, state,
                                          pointwise=state['cchain_pointwise'])
            loss = loss + chain_loss

        return loss
    
    def chain_loss(self, minibatch, state, pointwise=False):
        chain_length = self.min_time + self.chain_length * (np.random.rand() - self.min_time)
        # if state['n_iter'] % 391 == 0:
        #     print(chain_length)
        model = state['model']
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape

        if pointwise:
            B = 1
               
        initial_dist = self.cfg.sampler.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        x_T = get_initial_samples(B, D, device, S, initial_dist,
                initial_dist_std)
        x_logits = model(x_T, torch.ones((B,), device=device)) # (B, D, S)
        p0T = F.softmax(x_logits, dim=2).detach()
        # minibatch = torch.distributions.categorical.Categorical(logits=x_logits).sample() # x_0

        ts = 1.0 - torch.rand((B,), device=device) * (1.0 - self.min_time - chain_length)
        # qt0 = model.transition(ts) # (B, S, S)
        # rate = model.rate(ts) # (B, S, S)

        # qt0_rows_reg = qt0[
        #     torch.arange(B, device=device).repeat_interleave(D),
        #     minibatch.flatten().long(),
        #     :
        # ] # (B*D, S)

        # x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        # x_t = x_t_cat.sample().view(B, D).detach()
        x_t = self.calc_pst(model, x_T, p0T, torch.ones((B,),device=device), ts, sample=True)

        pred_t = F.softmax(model(x_t, ts), dim=2)

        if pointwise:
            # pred_t = pred_t.view(D, S)
            u = ts[0] - chain_length
            # x_u = one_step_sampling(model, 1, D, S, self.ratio_eps, x_t, ts,
            #                   delta=chain_length*torch.ones((self.chain_batch_size,), device=device
            #                                                 ), p0t=pred_t, batch_size=self.chain_batch_size)
            
            x_u = self.calc_pst(model, x_t.expand(self.chain_batch_size, D),
                                pred_t.expand(self.chain_batch_size, D, S),
                                ts[0]*torch.ones((self.chain_batch_size,),device=device),
                                u*torch.ones((self.chain_batch_size,), device=device),
                                sample=True)
            
            pred_t = pred_t.view(D, S)
            pred_u = F.softmax(model(x_u,u*torch.ones((self.chain_batch_size,),device=device)),
                               dim=2).mean(dim=0)

        else:
            us = ts - chain_length
            # x_u = one_step_sampling(model, B, D, S, self.ratio_eps, x_t, ts,
            #                   delta=chain_length*torch.ones((B, 1), device=device), p0t=pred_t, batch_size=1)
            x_u = self.calc_pst(model, x_t, pred_t, ts, us, sample=True)
            pred_t = pred_t.mean(dim=0)
            pred_u = F.softmax(model(x_u, us), dim=2).mean(dim=0)

        pred_t_permute = torch.permute(pred_t, (0,1)) + self.ratio_eps
        pred_u_permute = torch.permute(pred_u, (0,1)) + self.ratio_eps
        
        # chain_loss = self.kl_loss(pred_t_permute.log(), pred_u_permute).sum()
        chain_loss = self.kl_loss(pred_t_permute.log(), pred_u_permute).sum()

        return chain_loss
    
    def chain_loss_combined(self, minibatch, state, writer, pointwise=False):
        model = state['model']
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = model.device

        ts = 1.0 - torch.rand((B,), device=device) * (1.0 - self.min_time - self.chain_length)

        qt0 = model.transition(ts) # (B, S, S)

        rate = model.rate(ts) # (B, S, S)


        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.flatten().long(),
            :
        ] # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            x_t.long().flatten(),
            :
        ] # (B*D, S)
        rate_vals_square[
            torch.arange(B*D, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, D)


        # ---------- First term of ELBO (regularization) ---------------


        if self.one_forward_pass:
            x_logits = model(x_tilde, ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_tilde
        else:
            x_logits = model(x_t, ts) # (B, D, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
            reg_x = x_t

        # For (B, D, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B,D,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            reg_x.long().flatten()
        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, D, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )



        # ----- second term of continuous ELBO (signal term) ------------

        
        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            p0t_sig = F.softmax(model(x_tilde, ts), dim=2) # (B, D, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, D)



        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_tilde.long().flatten()
        ].view(B, D, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, D, S)


        x_tilde_mask = torch.ones((B,D,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,D,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,D,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, D, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D*S),
            torch.arange(S, device=device).repeat(B*D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D*S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*D)
        ].view(B, D, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            minibatch.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, D) + self.ratio_eps



        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)

        reg_mean = torch.mean(reg_term)


        writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
        writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])


        neg_elbo = sig_mean + reg_mean



        perm_x_logits = torch.permute(x_logits, (0,2,1))

        nll = self.cross_ent(perm_x_logits, minibatch.long())

        # ----- consistency chain loss uing x_t ------------

        # Given:
        # x_logits = model(x_tilde, ts) # (B, D, S)
        # p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
        # reg_x = x_tilde

        if pointwise:
            chain_length = torch.rand(ts[0:1].size(), device=device) * torch.minimum(
                ts[0:1] - self.min_time, self.chain_length * torch.ones_like(ts[0:1], device=device))
            x_t = reg_x[0:1]
            u = ts[0] - chain_length

            # x_u = one_step_sampling(model, 1, D, S, self.ratio_eps, x_t, ts[0:1],
            #                   delta=chain_length, p0t=pred_t, batch_size=self.chain_batch_size)
            x_u, _ = self.calc_pst(model, x_t.expand(self.chain_batch_size, D),
                                p0t_reg[0:1].expand(self.chain_batch_size, D, S),
                                ts[0]*torch.ones((self.chain_batch_size,),device=device),
                                u*torch.ones((self.chain_batch_size,), device=device),
                                sample=True)
            pred_t = p0t_reg[0].view(D, S)

            pred_u = F.softmax(model(x_u,u*torch.ones((self.chain_batch_size,),device=device)),
                               dim=2).mean(dim=0)

        else:
            x_t = reg_x
            pred_t = p0t_reg
            chain_length = torch.rand(ts.size(), device=device) * torch.minimum(
                ts - self.min_time, self.chain_length * torch.ones_like(ts, device=device))
            us = ts - chain_length
            # x_u = one_step_sampling(model, B, D, S, self.ratio_eps, x_t, ts,
            #                   delta=chain_length, p0t=pred_t, batch_size=1)
            x_u = self.calc_pst(model, x_t, pred_t, ts, us, sample=True)
            pred_t = pred_t.mean(dim=0)
            pred_u = F.softmax(model(x_u, us), dim=2).mean(dim=0)

        pred_t_permute = torch.permute(pred_t, (0,1)) + self.ratio_eps
        pred_u_permute = torch.permute(pred_u, (0,1)) + self.ratio_eps
        
        # chain_loss = self.kl_loss(pred_t_permute.log(), pred_u_permute).sum()
        chain_loss = self.kl_loss(pred_t_permute.log(), pred_u_permute).sum()
        # chain_loss = self.kl_loss(pred_u_permute.log(), pred_t_permute.detach()).sum()
        # chain_loss = torch.mean((pred_t_permute.log() - pred_u_permute.log())**2)
        # chain_loss = torch.mean((pred_t_permute - pred_u_permute)**2)

        return neg_elbo + self.nll_weight * nll, chain_loss
    
    def calc_pst(self, model, x_t, p0t, ts, ss, sample=False):
        # model <- teacher network
        # x_t: (B, D)
        # p0t: (B, D, S)
        # ts, ss: (B,)
        B, D, S = p0t.shape
        device = model.device

        qt0 = model.transition(ts) + self.ratio_eps # (B, S, S)
        qs0 = model.transition(ss) # (B, S, S)
        qts = model.transition(ts, ss) # (B, S, S)
        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_t.flatten().long()
        ] # (B*D, S)

        qts_rows_reg = qts[
            torch.arange(B, device=device).repeat_interleave(D),
            :,
            x_t.flatten().long()
        ] # (B*D, S)

        if sample:
            x_0_cat = torch.distributions.categorical.Categorical(p0t.detach()) # (B, D)
            x_0 = x_0_cat.sample()
            qt0_denom = qt0_rows_reg[torch.arange(B*D,
                device=device), x_0.flatten().long()].view(B*D, 1)
            qt0_denom = qt0_denom.expand(-1, S)
            qs0_rows_reg = qs0[
                torch.arange(B, device=device).repeat_interleave(D),
                x_0.flatten().long(),
                :
            ] # (B*D, S)
            ps_0t = qts_rows_reg / qt0_denom * qs0_rows_reg # (B*D, S)
            ps_0t = ps_0t + self.ratio_eps
            ps_0t = ps_0t / ps_0t.sum(dim=-1).view(-1, 1).expand(-1, S)
            x_s_cat = torch.distributions.categorical.Categorical(ps_0t)
            x_s = x_s_cat.sample().view(B, D)
            return x_s
        else:
            qt0_denom = qt0_rows_reg.view(B, D, S)
            ret0 = p0t / qt0_denom
            rets = torch.matmul(ret0, qs0)
            qts_numer = qts_rows_reg.view(B, D, S)
            return rets * qts_numer

@losses_utils.register_loss
class CChainOnly(CChain):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def calc_loss(self, minibatch, state, writer):
        return self.consistency_coeff * self.chain_loss(minibatch, state, pointwise=state['cchain_pointwise'])

@losses_utils.register_loss
class CChainCombined(CChain):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def calc_loss(self, minibatch, state, writer):
        chain_loss = self.chain_loss(minibatch, state, pointwise=state['cchain_pointwise'])
        loss, sample_chain_loss = self.chain_loss_combined(minibatch, state, writer, pointwise=state['cchain_pointwise'])
        return (chain_loss + sample_chain_loss) * self.consistency_coeff / 2 + loss * self.preservation_coeff





@losses_utils.register_loss
class ConditionalAux():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
        self.condition_dim = cfg.loss.condition_dim
        self.cross_ent = nn.CrossEntropyLoss()



    def calc_loss(self, minibatch, state, writer):
        model = state['model']
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C*H*W)
        B, D = minibatch.shape
        device = model.device

        ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time

        qt0 = model.transition(ts) # (B, S, S)

        rate = model.rate(ts) # (B, S, S)

        conditioner = minibatch[:, 0:self.condition_dim]
        data = minibatch[:, self.condition_dim:]
        d = data.shape[1]


        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.flatten().long(),
            :
        ] # (B*d, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, d)

        rate_vals_square = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            x_t.long().flatten(),
            :
        ] # (B*d, S)
        rate_vals_square[
            torch.arange(B*d, device=device),
            x_t.long().flatten()
        ] = 0.0 # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, d, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, d)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample() # (B,) taking values in [0, d)
        rate_new_val_probs = rate_vals_square[
            torch.arange(B, device=device),
            square_dims,
            :
        ] # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample() # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, d)


        # ---------- First term of ELBO (regularization) ---------------


        if self.one_forward_pass:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts) # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim:, :] # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, d, S)
            reg_x = x_tilde
        else:
            model_input = torch.concat((conditioner, x_t), dim=1)
            x_logits_full = model(model_input, ts) # (B, D, S)
            x_logits = x_logits_full[:, self.condition_dim:, :] # (B, d, S)
            p0t_reg = F.softmax(x_logits, dim=2) # (B, d, S)
            reg_x = x_t

        # For (B, d, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B,d,S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)
        
        qt0_denom_reg = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten()
        ].view(B, d, S) + self.ratio_eps

        rate_vals_reg = rate[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            reg_x.long().flatten()
        ].view(B, d, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1,2) # (B, d, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1,2)
        )



        # ----- second term of continuous ELBO (signal term) ------------

        
        if self.one_forward_pass:
            p0t_sig = p0t_reg
        else:
            model_input = torch.concat((conditioner, x_tilde), dim=1)
            x_logits_full = model(model_input, ts) # (B, d, S)
            x_logits = x_logits_full[:, self.condition_dim:, :]
            p0t_sig = F.softmax(x_logits, dim=2) # (B, d, S)

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d*S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*d)
        ].view(B, d, S)

        outer_qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.long().flatten(),
            x_tilde.long().flatten()
        ] + self.ratio_eps # (B, d)



        qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


        qt0_denom_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            :,
            x_tilde.long().flatten()
        ].view(B, d, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        ) # (B, d, S)


        x_tilde_mask = torch.ones((B,d,S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(d),
            torch.arange(d, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(d*S),
            torch.arange(S, device=device).repeat(B*d),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B,d,S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B,d,1)) * inner_log_sig,
            dim=(1,2)
        )

        # now getting the 2nd term normalization

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(d),
            x_tilde.long().flatten()
        ].view(B, d)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp # (B,d)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
            Z_subtraction.view(B, d, 1) + \
            Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(d*S),
            torch.arange(S, device=device).repeat(B*d),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, d, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(d*S),
            data.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B*d)
        ].view(B, d, S)

        qt0_sig_norm_denom = qt0[
            torch.arange(B, device=device).repeat_interleave(d),
            data.long().flatten(),
            x_tilde.long().flatten()
        ].view(B, d) + self.ratio_eps



        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,d,1)),
            dim=(1,2)
        )

        sig_mean = torch.mean(- outer_sum_sig/sig_norm)
        reg_mean = torch.mean(reg_term)

        writer.add_scalar('sig', sig_mean.detach(), state['n_iter'])
        writer.add_scalar('reg', reg_mean.detach(), state['n_iter'])

        neg_elbo = sig_mean + reg_mean


        perm_x_logits = torch.permute(x_logits, (0,2,1))

        nll = self.cross_ent(perm_x_logits, data.long())

        return neg_elbo + self.nll_weight * nll
