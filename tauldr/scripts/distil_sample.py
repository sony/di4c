import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections
import sys
import os
sys.path.append(os.getcwd())
import config.eval.cifar10 as cifar10_conf
# from config.eval.cifar10 import get_config as get_eval_config
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.sampling.sampling as sampling
import lib.sampling.sampling_utils as sampling_utils
from PIL import Image

from lib.losses.losses import DistilChain
from tqdm import tqdm
from config.train.cifar10 import get_config as get_teacher_config
t_cfg = get_teacher_config()

import argparse
parser = argparse.ArgumentParser()

import time, statistics

parser.add_argument('-d', '--device', default='cuda:0')
parser.add_argument('-n', '--num_steps', type=int, default=10)
parser.add_argument('-l', '--last_teacher', type=int, default=0)
sampling_opt = parser.parse_args()


distil_model = True
max_sample = True
last_lams = 1
fix_lams = False
fix_lams_value = 0.5

last_teacher = sampling_opt.last_teacher
num_steps = sampling_opt.num_steps
how_many = 50000
batch = 50
cfg = 0

qmc_size = 8
min_time = 0.01
# num_steps = 10
# how_many = 10
# batch = 1
device = sampling_opt.device

# fix seed
np.random.seed(483656)
torch.manual_seed(169785)

if distil_model:
    from config.train.distil_cifar10 import get_config as get_train_config

    checkpoint_path = 'path/to/your/checkpoint'

    save_samples_path = ''
    for folder in checkpoint_path.split('/')[:3]:
        save_samples_path = save_samples_path + folder + '/'
    if fix_lams:
        # how_many = 50
        save_samples_path = save_samples_path + 'fix_lams/'
    if last_teacher == 0:
        save_samples_path = save_samples_path + str(num_steps) + 'samples'
    else:
        save_samples_path = save_samples_path + str(num_steps - last_teacher) + '_' + str(last_teacher) + 'samples'
else:
    from config.train.cifar10 import get_config as get_train_config
    checkpoint_path = 'cifar10/checkpoints/ckpt_0001999999.pt'
    if max_sample:
        save_samples_path = 'results/' + str(num_steps) + 'step_baseline'
    else:
        save_samples_path = 'results/nomax/' + str(num_steps) + 'step_baseline'

os.makedirs(save_samples_path, exist_ok=True)


train_cfg = get_train_config()

# for item in eval_cfg.train_config_overrides:
#     utils.set_in_nested_dict(train_cfg, item[0], item[1])

S = train_cfg.data.S

model = model_utils.create_model(train_cfg, device)

loaded_state = torch.load(checkpoint_path,
    map_location=device)

modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
model.load_state_dict(modified_model_state)

model.eval()

if last_teacher > 0 or cfg:
    teacher = model_utils.create_model(t_cfg, device)
    pretrained_checkpoint = 'cifar10/checkpoints/ckpt_0001999999.pt'
    loaded_state = torch.load(Path(pretrained_checkpoint),
        map_location=device)
    modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
    teacher.load_state_dict(modified_model_state)
    teacher.eval()

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x



total_samples = 0
losser = DistilChain(train_cfg)

def sampler(model, batch):
    initial_dist_std = model.Q_sigma
    D = 3*32*32
    x_t = sampling.get_initial_samples(batch, D, device, S, 'gaussian', initial_dist_std)
    timepoints = torch.linspace(1.0, min_time, num_steps, device=device)
    for i in range(num_steps - 1):
        ts = timepoints[i] * torch.ones((batch,), device=device)
        ss = timepoints[i+1] * torch.ones((batch,), device=device)
        if distil_model:
            lams = torch.rand((batch, model.lam_raw_dim), device=device)
            if fix_lams:
                lams = torch.ones((batch, model.lam_raw_dim), device=device) * fix_lams_value
            logits = model(x_t, ts, lams) if i < (num_steps - last_teacher) else teacher(x_t, ts) # (batch, D, S)
            p0t = F.softmax(logits, dim=2) # (batch, D, S)
            if cfg > 0.1:
                t_logits = teacher(x_t, ts)
                p0t = F.softmax(t_logits + cfg * (logits - t_logits), dim=2)
        else:
            logits = model(x_t, ts)
            p0t = F.softmax(logits, dim=2) # (batch, D, S)
        x_t, _ = losser.calc_pst(model, x_t, p0t, ts, ss, sample=True)
        # ps_0t = losser.calc_pst(model, x_t, p0t, ts, ss, sample=False)
        # x_s_cat = torch.distributions.categorical.Categorical(ps_0t.detach()) # (B, D)
        # x_t = x_s_cat.sample()
    if distil_model:
        lams = torch.rand((batch*last_lams, model.lam_raw_dim), device=device)
        if fix_lams:
            lams =  torch.ones((batch*last_lams, model.lam_raw_dim), device=device) * fix_lams_value
        if last_teacher == 0:
            # logits = model(x_t, ss, lams) # (batch, D, S)
            x_t_rep = x_t.repeat_interleave(last_lams,dim=0)
            ss_rep = ss.repeat_interleave(last_lams,dim=0)
            logits = model(x_t_rep, ss_rep, lams) # (batch*last_lams, D, S)
            p0t = F.softmax(logits, dim=2)
            
            if max_sample and (last_lams > 1):
                max_vals, max_idx = torch.max(p0t, dim=2)
                max_vals = max_vals.view(batch, last_lams, D) + 1e-9
                max_idx = max_idx.view(batch, last_lams, D)
                idx_select = torch.max(max_vals.log().sum(dim=2), dim=1)[1]
                x_0max = max_idx[torch.arange(batch), idx_select]
                return x_0max.detach().cpu().numpy().astype(int)
            else:
                p0t = p0t.view(batch, last_lams, D, S).mean(dim=1)
                if cfg > 0.1:
                    t_logits = teacher(x_t, ss)
                    p0t = F.softmax(t_logits + cfg * (logits - t_logits), dim=2)
        else:
            logits = teacher(x_t, ss)
            p0t = F.softmax(logits, dim=2) # (batch, D, S)
    else:
        logits = model(x_t, ss)
        p0t = F.softmax(logits, dim=2) # (batch, D, S)
    if max_sample:
        x_0max = torch.max(p0t, dim=2)[1]
    else:
        x_0max = losser.calc_pst(model, x_t, p0t, ss, 0*ss, sample=True, sample_zero=True)
    return x_0max.detach().cpu().numpy().astype(int)
# deal with the min_time?

# latencies = []

for j in tqdm(range(how_many//batch)):
    # start_time = time.perf_counter()
    samples = sampler(model, batch)
    # end_time = time.perf_counter()
    # latencies.append(end_time - start_time)
    samples = samples.reshape(batch, 3, 32, 32)
    samples_uint8 = samples.astype(np.uint8)
    for i in range(samples.shape[0]):
        path_to_save = save_samples_path + f'/{total_samples + i}.png'
        img = Image.fromarray(imgtrans(samples_uint8[i]))
        img.save(path_to_save)
    total_samples += batch

# avg_latency = statistics.mean(latencies[1:])
# std_latency = statistics.stdev(latencies[1:])
# print(f"Average Latency: {avg_latency:.4f} ± {std_latency:.4f} seconds")