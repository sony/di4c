import torch
import torch.nn as nn
import ml_collections
import yaml
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm
import sys
import signal
import argparse

import lib.utils.utils as utils
import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import lib.loggers.loggers as loggers
import lib.loggers.logger_utils as logger_utils

import numpy as np
import os

global_seed = 431354
seed_idx = 0
np.random.seed(global_seed)
np_seeds = np.random.randint(1000000000, size=2000000)
torch_seeds = np.random.randint(1000000000, size=2000000)
np.random.seed(np_seeds[seed_idx])
torch.manual_seed(torch_seeds[seed_idx])

def main(cfg, c_cfg, t_cfg, custom_name=None):
    # filename = c_cfg['loss'] + str(c_cfg['lr']) + c_cfg['device'] # + str(global_seed)
    # filepath = c_cfg['log_folder'] + filename + '.txt'
    foldername = c_cfg['log_folder'] + c_cfg['device']
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    cfg.save_location = foldername
    cfg.data.batch_size = 128 // c_cfg['distil_batch_size']
    cfg.optimizer.lr = c_cfg['lr']

    # with open(filepath, mode='w') as f:
    #     f.write('Global seed: {}\n\n'.format(global_seed))
    #     for key in c_cfg.keys():
    #         f.write(key +': ' + str(c_cfg[key]) + '\n')
    
    cfg.device = c_cfg['device']
    cfg.loss.name = c_cfg['loss']
    cfg.sampler = sampler = ml_collections.ConfigDict()
    sampler.initial_dist = c_cfg['sample_dist']

    print("Training with config", cfg.experiment_name)

    preempted_path = Path("null")
    if cfg.saving.enable_preemption_recovery:

        preempted_path = bookkeeping.check_for_preempted_run(cfg.save_location,
            cfg.saving.preemption_start_day_YYYYhyphenMMhyphenDD,
            cfg,
            cfg.saving.prepare_to_resume_after_timeout
        )

    if preempted_path.as_posix() == "null":
        save_dir, checkpoint_dir, config_dir = \
            bookkeeping.create_experiment_folder(
                cfg.save_location,
                cfg.experiment_name if custom_name is None else custom_name,
                custom_name is None
        )
        bookkeeping.save_config_as_yaml(cfg, config_dir)

        # bookkeeping.save_git_hash(save_dir)

    else:
        print("Resuming from preempted run: ", preempted_path)
        save_dir = preempted_path
        checkpoint_dir, config_dir = bookkeeping.create_inner_experiment_folders(save_dir)

    writer = bookkeeping.setup_tensorboard(save_dir, 0)

    device = torch.device(cfg.device)

    model = model_utils.create_model(cfg, device)
    print("number of parameters: ", sum([p.numel() for p in model.parameters()]))

    dataset = dataset_utils.get_dataset(cfg, device)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle)
    
    t_cfg.data.train = False
    t_cfg.data.subset = True
    t_cfg.data.indices = np.random.choice(10000, size=10)
    # for validation

    loss = losses_utils.get_loss(cfg)
    if not c_cfg["use_cv"]:
        loss.use_cv = False
    if c_cfg["use_qmc"]:
        loss.distil_qmc = True
    if c_cfg['finetune']:
        cfg.training.warmup = 0
    if c_cfg['time_schedule'] is not None:
        loss.time_schedule = c_cfg['time_schedule']

    training_step = training_utils.get_train_step(cfg)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    teacher = model_utils.create_model(t_cfg, device)
    pretrained_checkpoint = 'cifar10/checkpoints/ckpt_0001999999.pt'
    loaded_state = torch.load(Path(pretrained_checkpoint),
        map_location=device)
    modified_model_state = utils.remove_module_from_keys(loaded_state['model'])
    teacher.load_state_dict(modified_model_state)
    teacher.eval()

    if c_cfg['finetune']:
        model.load_state_dict(modified_model_state)
        model.init_ema()

    state = {
        'model': model,
        'teacher': teacher,
        'optimizer': optimizer,
        'n_iter': 0,
        'distil_batch_size': c_cfg['distil_batch_size'],
        'loss_freq': 50,
    }

    bookkeeping.setup_preemption(save_dir, checkpoint_dir, state,
        12, # cfg.saving.num_checkpoints_to_keep,
        cfg.saving.prepare_to_resume_after_timeout)


    def print_nll():
        np.random.seed(378237)
        torch.manual_seed(216848)
        t_dataset = dataset_utils.get_dataset(t_cfg, device)
        distil_batch_size = state['distil_batch_size']
        state['distil_batch_size'] = 128
        state['model'].eval()
        t_dataloader = torch.utils.data.DataLoader(t_dataset,
        batch_size = 128 // state['distil_batch_size'], #cfg.data.batch_size,
        shuffle=t_cfg.data.shuffle)
        data_loss = 0
        distil_loss = 0
        # marginal_consis_loss = 0
        consis_loss = 0
        corr = 0
        n_batch = 0
        with torch.no_grad():
            for minibatch in tqdm(t_dataloader):
                n_batch += 1
                a, b, d, cor = loss.calc_loss(minibatch, state, eval=True)
                data_loss += a
                distil_loss += b
                # marginal_consis_loss += c
                consis_loss += d
                corr += cor
        print("data: {}, distil: {}, consis: {}, corr: {}".format(data_loss/n_batch, distil_loss/n_batch,
                                                       consis_loss/n_batch, corr/n_batch))
        global seed_idx
        seed_idx = (seed_idx + 1) % len(np_seeds)
        np.random.seed(np_seeds[seed_idx])
        torch.manual_seed(torch_seeds[seed_idx])
        state['model'].train()
        state['distil_batch_size'] = distil_batch_size

    if not preempted_path.as_posix() == 'null':
        state = bookkeeping.resume_training(preempted_path, state, cfg.device)

    low_freq_loggers = []
    for logger in cfg.saving.low_freq_loggers:
        low_freq_loggers.append(logger_utils.get_logger(logger))

    exit_flag = False
    checkpoint_num_list = [5000,10000,20000,40000,80000,160000,320000,640000,1280000]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=c_cfg['lr_decay'])

    while True:
        for minibatch in tqdm(dataloader):

            if state['n_iter'] % state['loss_freq'] == 0:
                print_nll()

            training_step.step(state, minibatch, loss, writer)


            if (state['n_iter'] + 1) in checkpoint_num_list or state['n_iter'] == cfg.training.n_iters-1:
                bookkeeping.save_checkpoint(checkpoint_dir, state, 12)
                if c_cfg['distil_batch_increase'] and state['n_iter'] > 20000:
                    if state['distil_batch_size'] < 64:
                        state['distil_batch_size'] *= 2


            # if state['n_iter'] % cfg.saving.checkpoint_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
            #     bookkeeping.save_checkpoint(checkpoint_dir, state, cfg.saving.num_checkpoints_to_keep)

            # if state['n_iter'] % cfg.saving.log_low_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
            #     for logger in low_freq_loggers:
            #         logger(state=state, cfg=cfg, writer=writer,
            #                minibatch=minibatch, dataset=dataset)


            state['n_iter'] += 1
            if state['n_iter'] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

        scheduler.step()

    writer.close()

    return save_dir



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args, unknown_args = parser.parse_known_args()
    if args.config == 'cifar10':
        from config.train.distil_cifar10 import get_config
        from config.train.cifar10 import get_config as get_teacher_config
    # elif args.config == 'piano':
    #     from config.train.piano import get_teacher_config
    else:
        raise NotImplementedError

    cfg = get_config()
    t_cfg = get_teacher_config()
    c_cfg = {
        'device': 'cuda:0',
        'loss': 'DistilChain',
        'lr': 2e-4,
        'lr_decay': 1,
        'time_schedule': 'sigmoid', # None / 'zero' /  'sigmoid' / 'linear'
        'freq': 1,
        'sample_dist': 'gaussian',
        'log_folder': 'results/distil/',
        'distil_batch_size': 16,
        'distil_batch_increase': False,
        'finetune': True,
        'use_qmc': False,
        'use_cv': True,
    }
    main(cfg, c_cfg, t_cfg)
