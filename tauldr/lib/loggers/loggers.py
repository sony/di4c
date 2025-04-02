import ml_collections
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.nn.functional as F
from pathlib import Path
import os
import time
from tqdm import tqdm

import lib.loggers.logger_utils as logger_utils
import lib.sampling.sampling_utils as sampling_utils
import lib.sampling.sampling as sampling
import lib.losses.losses as losses
import lib.utils.bookkeeping as bookkeeping


@logger_utils.register_logger
def denoisingImages(*args, **kwargs):
    state = kwargs['state']
    cfg = kwargs['cfg']
    writer = kwargs['writer']
    minibatch = kwargs['minibatch']
    dataset = kwargs['dataset']
    model = state['model']

    ts = [0.01, 0.3, 0.5, 0.6,0.7,0.8, 1.0]
    C,H,W = cfg.data.shape
    B = 1
    S = cfg.data.S

    def imgtrans(x):
        # C,H,W -> H,W,C
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        return x

    fig, ax = plt.subplots(6, len(ts))
    for img_idx in range(3):
        for t_idx in range(len(ts)):
            qt0 = model.transition(torch.tensor([ts[t_idx]], device=model.device)) # (B, S, S)
            qt0_rows = qt0[
                0, minibatch[img_idx, ...].flatten().long(), :
            ]
            x_t_cat = torch.distributions.categorical.Categorical(
                qt0_rows
            )
            x_t = x_t_cat.sample().view(1, C*H*W)

            x_0_logits = model(x_t, torch.tensor([ts[t_idx]], device=model.device)).view(B,C,H,W,S)
            x_0_max_logits = torch.max(x_0_logits, dim=4)[1]

            ax[2*img_idx, t_idx].imshow(imgtrans(x_t.view(B,C,H,W)[0, ...].detach().cpu()))
            ax[2*img_idx, t_idx].axis('off')
            ax[2*img_idx+1, t_idx].imshow(imgtrans(x_0_max_logits[0, ...].detach().cpu()))
            ax[2*img_idx+1, t_idx].axis('off')

    writer.add_figure('denoisingImages', fig, state['n_iter'])


@logger_utils.register_logger
def ConditionalDenoisingNoteSeq(*args, **kwargs):
    state = kwargs['state']
    cfg = kwargs['cfg']
    writer = kwargs['writer']
    dataset = kwargs['dataset']
    model = state['model']
    minibatch = kwargs['minibatch']

    ts = [0.01, 0.1, 0.3, 0.7, 1.0]
    total_L = cfg.data.shape[0]
    data_L = cfg.data.shape[0] - cfg.loss.condition_dim
    S = cfg.data.S


    with torch.no_grad():
        fig, ax = plt.subplots(2, len(ts))
        for data_idx in range(1):
            for t_idx in range(len(ts)):
                qt0 = model.transition(torch.tensor([ts[t_idx]], device=model.device)) # (B, S, S)
                conditioner = minibatch[data_idx, 0:cfg.loss.condition_dim].view(1, cfg.loss.condition_dim)
                data = minibatch[data_idx, cfg.loss.condition_dim:].view(1, data_L)
                qt0_rows = qt0[
                    0, data.flatten().long(), :
                ]
                x_t_cat = torch.distributions.categorical.Categorical(
                    qt0_rows
                )
                x_t = x_t_cat.sample().view(1, data_L)

                model_input = torch.concat((conditioner, x_t), dim=1)

                full_x_0_logits = model(model_input, torch.tensor([ts[t_idx]], device=model.device)).view(1,total_L,S)
                x_0_logits = full_x_0_logits[:, cfg.loss.condition_dim:, :]

                x_0_max_logits = torch.max(x_0_logits, dim=2)[1]

                x_0_np = x_0_max_logits.cpu().detach().numpy()
                x_t_np = x_t.cpu().detach().numpy()
                conditioner_np = conditioner[data_idx, :].cpu().detach().numpy()

                ax[2*data_idx, t_idx].scatter(np.arange(total_L),
                    np.concatenate((conditioner_np, x_t_np[0, :]), axis=0), s=0.1)
                ax[2*data_idx, t_idx].axis('off')
                ax[2*data_idx, t_idx].set_ylim(0, S)
                ax[2*data_idx+1, t_idx].scatter(np.arange(total_L),
                    np.concatenate((conditioner_np, x_0_np[0, :]), axis=0), s=0.1)
                ax[2*data_idx+1, t_idx].axis('off')
                ax[2*data_idx+1, t_idx].set_ylim(0, S)

    fig.set_size_inches(len(ts)*2, 2*2)

    writer.add_figure('ConditionaldenoisingNoteSeq', fig, state['n_iter'])

@logger_utils.register_logger
def ELBO(*args, **kwargs):
    state = kwargs['state']
    cfg = kwargs['cfg']
    writer = kwargs['writer']
    dataset = kwargs['dataset']
    model = state['model']

    C,H,W = cfg.data.shape
    D = C*H*W
    S = cfg.data.S
    total_B = cfg.logging.total_B
    B = cfg.logging.B
    total_N = cfg.logging.total_N
    min_time = cfg.logging.min_t
    eps = cfg.logging.eps
    device = cfg.device


    assert total_B/B == round(total_B/B)

    print(f"Calculating likelihoods on {total_B} out of {dataset.data.shape[0]} images")
    data = dataset.data[0:total_B, ...]


    if cfg.logging.initial_dist == 'gaussian':
        initial_dist_std = model.Q_sigma
        target = np.exp(
            - ((np.arange(1, S+1) - S//2)**2) / (2 * initial_dist_std**2)
        )
        target = target / np.sum(target)

        initial_dist = torch.distributions.categorical.Categorical(
            torch.from_numpy(target).to(device)
        )
    elif cfg.logging.initial_dist == 'uniform':
        initial_dist = torch.distributions.categorical.Categorical(
            torch.ones((S,), device=device)
        )
    else:
        raise NotImplementedError

    with torch.no_grad():
        elbos = np.zeros((total_B//B, total_N))
        # chain_losses = np.zeros(total_B//B)

        for b_repeat in tqdm(range(total_B//B)):
            for n_repeat in range(total_N):
                x_0_CHW = data[b_repeat*B:(b_repeat+1)*B, ...] # (B, C, H, W)
                x_0_D = x_0_CHW.view(B, D)


                qT0 = model.transition(torch.tensor([1.0], device=device))[0, :, :]

                qT0dist = torch.distributions.categorical.Categorical(
                    qT0[x_0_D.flatten().long(), :].view(B,D,S)
                )
                x_T = qT0dist.sample() # (B, D)
                logpref = initial_dist.log_prob(x_T) # (B, D)

                prefterm = - torch.mean(torch.sum(logpref, dim=1))



                S = cfg.data.S
                minibatch = x_0_D

                ts = torch.rand((B,), device=device) * (1.0 - min_time) + min_time

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


                x_logits = model(x_t, ts) # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2) # (B, D, S)
                reg_x = x_t

                # For (B, D, S, S) first S is x_0 second S is x'

                qt0_numer_reg = qt0.view(B, S, S)
                
                qt0_denom_reg = qt0[
                    torch.arange(B, device=device).repeat_interleave(D),
                    :,
                    reg_x.long().flatten()
                ].view(B, D, S) + eps

                rate_vals_reg = rate[
                    torch.arange(B, device=device).repeat_interleave(D),
                    :,
                    reg_x.long().flatten()
                ].view(B, D, S)

                reg_tmp = rate_vals_reg @ qt0_numer_reg.transpose(1,2) # (B, D, S)

                reg_term = torch.sum(
                    (p0t_reg / qt0_denom_reg) * reg_tmp,
                    dim=(1,2)
                )



                # ----- second term of continuous ELBO (signal term) ------------

                
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
                ] + eps # (B, D)



                qt0_numer_sig = qt0.view(B, S, S) # first S is x_0, second S is x


                qt0_denom_sig = qt0[
                    torch.arange(B, device=device).repeat_interleave(D),
                    :,
                    x_tilde.long().flatten()
                ].view(B, D, S) + eps

                inner_log_sig = torch.log(
                    (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + eps
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
                ].view(B, D) + eps



                sig_norm = torch.sum(
                    (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B,D,1)),
                    dim=(1,2)
                )

                sig_mean = torch.mean(- outer_sum_sig/sig_norm)
                reg_mean = torch.mean(reg_term)

                neg_elbo = sig_mean + reg_mean

                elbos[b_repeat, n_repeat] = (prefterm + neg_elbo).detach().cpu().numpy()

                # # compute consistency loss
                # chain_length = 0.01
                # x_t = reg_x
                # pred_t = p0t_reg # (D, S)
                # us = ts - chain_length
                # chain_length = torch.rand(ts.size(), device=device) * torch.minimum(
                #     ts - min_time, chain_length * torch.ones_like(ts, device=device))
                # x_u = sampling.one_step_sampling(model, B, D, S, eps, x_t, ts,
                #                 delta=chain_length, p0t=pred_t, batch_size=1)
                # pred_t = pred_t.mean(dim=0)
                # pred_u = F.softmax(model(x_u, us), dim=2).mean(dim=0)

                # pred_t_permute = torch.permute(pred_t, (0,1)) + eps
                # pred_u_permute = torch.permute(pred_u, (0,1)) + eps
                
                # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
                # chain_losses[b_repeat] = kl_loss(pred_t_permute.log(), pred_u_permute).mean()

                


    elbos = elbos / (3*32*32)
    writer.add_scalar('neg_elbo', np.mean(elbos), 0)
    writer.add_numpy_data('full_elbos', elbos, 0)

    # writer.add_scalar('chain_loss', np.mean(chain_losses), 0)
