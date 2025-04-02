# Main file to launch training or evaluation
import os
import random

import numpy as np
import argparse

import torch
from torch.distributed import init_process_group, destroy_process_group

from Trainer.vit import MaskGIT
import shutil
import copy
import time, statistics

def main(args):
    """ Main function: Train or eval MaskGIT """
    teacher = None
    if args.lfg or (not (args.test_only or args.debug)):
        # Load the teacher model
        args_teacher = copy.deepcopy(args)
        args_teacher.resume = False
        args_teacher.is_student = False
        teacher = MaskGIT(args_teacher)
        teacher.vit.eval()  # Set teacher to evaluation mode
    
    student = MaskGIT(args)
    # student.vit.load_state_dict(torch.load(student_checkpoint, map_location=args.device)['model_state_dict'])
    
    if args.test_only:  # Evaluate the networks
        student.eval(teacher)
    elif args.debug:  # custom code for testing inference
        import torchvision.utils as vutils
        from torchvision.utils import save_image
        with torch.no_grad():
            # labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)] * 1, "r_row"
            labels, name = [980 for i in range(10)]*1, "r_row"
            labels = torch.LongTensor(labels).to(args.device)
            # sm_temp = 1.3          # Softmax Temperature
            # r_temp = 7             # Gumbel Temperature
            # w = 9                  # Classifier Free Guidance
            # randomize = "linear"   # Noise scheduler
            # step = 32              # Number of step
            # sched_mode = "arccos"  # Mode of the scheduler
            # # Generate sample
            latencies = []
            inference_latencies = []
            measure_latency = False
            if measure_latency:
                # labels, name = [random.randint(0, 999) for i in range(64)] * 1, "r_row"
                labels = torch.LongTensor(labels).to(args.device)
                n_runs = 10
                # warm-up for latency evaluation
                student.sample(nb_sample=labels.size(0), labels=labels, sm_temp=args.sm_temp, r_temp=args.r_temp, w=args.cfg_w,
                                                randomize=args.randomize, sched_mode=args.sched_mode, step=args.step, teacher=teacher)
            
                for i in range(n_runs):
                    start_time = time.perf_counter()
                    gen_sample, _, _ = student.sample(nb_sample=labels.size(0), labels=labels, sm_temp=args.sm_temp, r_temp=args.r_temp, w=args.cfg_w,
                                                    randomize=args.randomize, sched_mode=args.sched_mode, step=args.step, teacher=teacher)
                    end_time = time.perf_counter()
                    latencies.append(end_time - start_time)
            
                avg_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies)
                print(f"Average Latency: {avg_latency:.4f} ± {std_latency:.4f} seconds")
            else:
                gen_sample, _, _ = student.sample(nb_sample=labels.size(0), labels=labels, sm_temp=args.sm_temp, r_temp=args.r_temp, w=args.cfg_w,
                                                    randomize=args.randomize, sched_mode=args.sched_mode, step=args.step, teacher=teacher)
            
            gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
            # Save image
            img_name = args.vit_folder + f"sched_{args.sched_mode}_step={args.step}_temp={args.sm_temp}" + f"_w={args.cfg_w}_randomize={args.randomize}_{name}.jpg"
            save_image(gen_sample, img_name)
    else:  # Begin training
        student.fit_di4c(teacher, num_epochs=args.epoch)

def ddp_setup():
    """ Initialization of the multi_gpus training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def launch_multi_main(args):
    """ Launch multi training"""
    ddp_setup()
    args.device = int(os.environ["LOCAL_RANK"])
    args.is_master = args.device == 0
    main(args)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   default="imagenet", help="dataset on which dataset to train")
    parser.add_argument("--data-folder",  type=str,   default="imagenet",         help="folder containing the dataset")
    parser.add_argument("--vqgan-folder", type=str,   default="pretrained_maskgit/VQGAN/",         help="folder of the pretrained VQGAN")
    parser.add_argument("--vit-folder",   type=str,   default="path/to/save/models",         help="folder where to save the Transformer")
    parser.add_argument("--writer-log",   type=str,   default="",         help="folder where to store the logs")
    parser.add_argument("--sched_mode",   type=str,   default="arccos",   help="scheduler mode whent sampling")
    parser.add_argument("--grad-cum",     type=int,   default=1,          help="accumulate gradient")
    parser.add_argument('--channel',      type=int,   default=3,          help="rgb or black/white image")
    parser.add_argument("--num_workers",  type=int,   default=4,          help="number of workers")
    parser.add_argument("--step",         type=int,   default=4,          help="number of step for sampling")
    parser.add_argument('--seed',         type=int,   default=42,         help="fix seed")
    parser.add_argument("--epoch",        type=int,   default=301,        help="number of epoch")
    parser.add_argument('--img-size',     type=int,   default=256,        help="image size")
    parser.add_argument("--bsize",        type=int,   default=2,        help="batch size")
    parser.add_argument("--mask-value",   type=int,   default=1024,       help="number of epoch")
    parser.add_argument("--lr",           type=float, default=1e-5,       help="learning rate to train the transformer")
    parser.add_argument("--cfg_w",        type=float, default=3,          help="classifier free guidance wight")
    parser.add_argument("--r_temp",       type=float, default=4.5,        help="Gumbel noise temperature when sampling")
    parser.add_argument("--sm_temp",      type=float, default=1.,         help="temperature before softmax when sampling")
    parser.add_argument("--drop-label",   type=float, default=0.1,        help="drop rate for cfg")
    parser.add_argument("--test-only",    action='store_true',            help="only evaluate the model")
    parser.add_argument("--resume",       action='store_true',            help="resume training of the model")
    parser.add_argument("--debug",        action='store_true',            help="debug")
    
    # Add new arguments for Di4C
    parser.add_argument("--teacher_vit", type=str, default="pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth")
    parser.add_argument("--is_student", action='store_true')
    parser.add_argument("--randomize", type=str, default="linear")
    parser.add_argument("--latent_bsize", type=int, default=32, help="batch size for lambda in Di4C")
    parser.add_argument("--r_delta", type=float, default=0.05, help="t-s in Di4C training")
    parser.add_argument("--lfg", action='store_true')
    parser.add_argument("--teacher_steps", type=int, default=8)
    parser.add_argument("--alpha_t", type=str, default="sigmoid", help="const|sigmoid|linear") 
    parser.add_argument("--alpha_const", type=float, default=0.1)
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.iter = 0
    args.global_epoch = 0

    if args.seed > 0: # Set the seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True

    world_size = torch.cuda.device_count()

    if world_size > 1:  # launch multi training
        print(f"{world_size} GPU(s) found, launch multi-gpus training")
        args.is_multi_gpus = True
        launch_multi_main(args)
    else:  # launch single Gpu training
        print(f"{world_size} GPU found")
        args.is_master = True
        args.is_multi_gpus = False
        main(args)
