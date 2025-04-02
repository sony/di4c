# Trainer for MaskGIT
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image

from torch.nn.parallel import DistributedDataParallel as DDP

from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer

from Network.Taming.models.vqgan import VQModel

import time


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        self.ae = self.get_network("autoencoder")
        self.codebook_size = self.ae.n_embed   
        print("Acquired codebook size:", self.codebook_size)   
        self.vit = self.get_network("vit")                                      # Load Masked Bidirectional Transformer   
        self.patch_size = self.args.img_size // 2**(self.ae.encoder.num_resolutions-1)     # Load VQGAN
        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)    # Get cross entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay

        self.latent_dim = 1 # added for Di4C
        
        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data = self.get_data()

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=50_000)
        
        self.kl_loss = nn.KLDivLoss(reduction='batchmean') # added for Di4C

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = MaskTransformer(
                img_size=self.args.img_size, hidden_dim=768, codebook_size=self.codebook_size, depth=24, heads=16, mlp_dim=3072, dropout=0.1, # Small
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
                is_student=self.args.is_student
            )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
            else:
                ckpt = self.args.teacher_vit
            if self.args.is_master:
                print("load ckpt from:", ckpt)
            # Read checkpoint file
            checkpoint = torch.load(ckpt, map_location='cpu')
            if self.args.resume:
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
            self.args.global_epoch += checkpoint['global_epoch']
            # Load network
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)


            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device]) #, find_unused_parameters=True)

        elif archi == "autoencoder":
            # Load config
            config = OmegaConf.load(self.args.vqgan_folder + "model.yaml")
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)
            

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256, ret_r=False, given_r=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        if given_r is None:
            r = torch.rand(code.size(0)).to(code.device)
        else:
            r = given_r.to(code.device)
        if mode == "linear":                # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":              # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()).to(code.device) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        if ret_r: # added for Di4C
            return mask_code, mask, r
        
        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        # r = torch.linspace(1, 0, step)
        r = torch.linspace(0, 1, step+1) # modified
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # # fill the scheduler by the ratio of tokens to predict at each step
        # sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        # sche = sche.round()
        # sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        # sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
        
        # modified version:
        sche = val_to_mask * (self.patch_size * self.patch_size)
        sche = sche.round()
        for i in range(len(sche) - 1):
            sche[i] = sche[i] - sche[i+1]
        sche = sche[:-1]
        sche[sche == 0] = 1
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()

        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        for x, y in bar:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            masked_code, mask = self.get_mask_code(code, value=self.args.mask_value, codebook_size=self.codebook_size)

            with torch.cuda.amp.autocast():                             # half precision
                pred = self.vit(masked_code, y, drop_label=drop_label)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1)) / self.args.grad_cum

            # update weight if accumulation of gradient is done
            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            # logs
            if update_grad and self.args.is_master:
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

            if self.args.iter % log_iter == 0 and self.args.is_master:
                # Generate sample for visualization
                gen_sample = self.sample(nb_sample=10)[0]
                gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                # Show reconstruction
                unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                reco_sample = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=self.args.vit_folder+"current.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            self.args.iter += 1

        return cum_loss / n

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            train_loss = self.train_one_epoch()

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            if e % 10 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1

    def eval(self, teacher=None):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        m = self.sae.compute_and_log_metrics(self, teacher=teacher)
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12, teacher=None):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        use_z = self.args.is_student # is the model trained with Di4C
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode
            

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision
                    z = None
                    if w != 0:
                        # Model Prediction
                        if use_z:
                            z = torch.rand(code.size(0), self.latent_dim).to(self.args.device)
                        if teacher is not None:
                            logit_c = self.vit(code, labels, ~drop, z=z)
                            logit_u = teacher.vit(code, labels, drop)
                        else:
                            if z is not None:
                                z=torch.cat([z, z], dim=0)
                            logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                            torch.cat([labels, labels], dim=0),
                                            torch.cat([~drop, drop], dim=0),
                                            z=z)
                            logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        if use_z:
                            z = torch.rand(code.size(0), self.latent_dim).to(self.args.device)
                        logit = self.vit(code.clone(), labels, drop_label=~drop, z=z)
                    
                prob = torch.softmax(logit * sm_temp, -1)
                # # Sample the code from the softmax prediction
                # distri = torch.distributions.Categorical(probs=prob)

                distri = torch.distributions.Categorical(logits=logit*sm_temp)

                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = indice / (len(scheduler)-1) 
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.ae.decode_code(_code)

        self.vit.train()
        return x, l_codes, l_mask

    ##### added below for Di4C #####
    def train_di4c(self, teacher, log_iter=10000):

        os.makedirs(self.args.vit_folder + "images/", exist_ok=True)
        os.makedirs(self.args.vit_folder + "checkpoints/", exist_ok=True)

        self.vit.train()
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        # for param in self.vit.module.latent_projection.parameters():
        #     param.requires_grad = True

        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        latent_bsize = self.args.latent_bsize

        distil_list = deque()
        data_list = deque()
        data_cor_list = deque()
        consis_list = deque()
        consis_cor_list = deque()

        for x, y in bar:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            drop_label = torch.empty(y.size()).uniform_(0, 1).to(self.args.device) < self.args.drop_label

            # VQGAN encoding to img tokens
            emb, _, [_, _, code] = self.ae.encode(x)
            code = code.reshape(x.size(0), self.patch_size, self.patch_size)  # (bsize, 16, 16)
        
            # For consistency loss:
            # Mask the encoded tokens and get the r value
            r = torch.rand(code.size(0)).to(self.args.device)
            # r = torch.randint_like(y, self.args.teacher_steps) / self.args.teacher_steps # if discrete timesteps
            masked_code, mask = self.get_mask_code(code, mode=self.args.sched_mode, 
                                                    value=self.args.mask_value, 
                                                    codebook_size=self.codebook_size,
                                                    given_r=r)
            with torch.cuda.amp.autocast():
                # One step denoising using teacher
                code_, logits_teacher = teacher.one_step_denoise(masked_code, y, mask, r, r_delta=self.args.r_delta,
                                            latent_bsize=latent_bsize, 
                                            sm_temp=self.args.sm_temp, 
                                            randomize=self.args.randomize, 
                                            r_temp=self.args.r_temp,
                                            mode=self.args.sched_mode,
                                            drop_label=drop_label)
                
                p0t_teacher = F.softmax(logits_teacher, dim=2).detach() # (B, D, S)
                log_p0t_teacher = F.log_softmax(logits_teacher, dim=2).detach()
                # Process student with denoised code from teacher
                self.vit.eval()
                logits_student_teacher = self.process_student(code_, y.repeat_interleave(latent_bsize, dim=0), drop_label=drop_label.repeat_interleave(latent_bsize, dim=0)).detach()
                self.vit.train()
                logits_student = self.process_student(masked_code, y, latent_bsize, drop_label=drop_label)

                # shapes:
                # logits_teacher: [bsize, D, codebook+1]
                # logits_student(_teacher): [bsize*latent_bsize, D, codebook+1]
                # D = self.patch_size * self.patch_size
                
                # translate things to the tauLDR-Di4C implementation:
                min_time = 0.075
                max_distil_time = 0.075
                use_cv = True # modify later
                device = self.args.device
                B = x.size(0)
                D = self.patch_size * self.patch_size
                S = self.codebook_size + 1
                batch_size = latent_bsize

                log_p0t_student_lam = F.log_softmax(logits_student, dim=2) # (B*batch_size, D, S)

                # --------------- Datapoint loss --------------------------
                b_idx = torch.arange(B, device=device).repeat_interleave(D)
                d_idx = torch.arange(D, device=device).repeat(B)
                s_idx = code.view(-1).long()
                dll_ = log_p0t_student_lam.view(B, batch_size, D, S)[
                                b_idx,
                                :,
                                d_idx,
                                s_idx
                            ].view(B, D, batch_size) # (B, D, batch_size)
                dll = dll_.sum(dim=1) # (B, batch_size)
                if use_cv:
                    dll = torch.logsumexp(dll, dim=1) - np.log(batch_size) - (dll_.logsumexp(dim=2) - np.log(batch_size)).sum(dim=1)
                    data_loss_cor = - dll # (B,)
                    log_p0t_student = torch.logsumexp(log_p0t_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
                    data_loss_indep = (p0t_teacher * (log_p0t_teacher - log_p0t_student)).view(B, -1).sum(dim=1)
                else:
                    data_loss_cor = - (torch.logsumexp(dll, dim=1) - np.log(batch_size))
                    data_loss_indep = 0
                # data_loss = data_loss_cor + data_loss_indep
                
                # --------------- Distillation loss --------------------
                # mints = min_time + torch.randint(2, (B*batch_size,), device=device) * torch.rand((B*batch_size,), device=device) * max_distil_time
                # x_0_for_mint = code.repeat_interleave(batch_size, dim=0) # (B*b, D)
                # x_mint, _ = self.get_mask_code(x_0_for_mint, mode=self.args.sched_mode, 
                #                                     value=self.args.mask_value, 
                #                                     codebook_size=self.codebook_size,
                #                                     given_r=1-mints)
                # logits_teacher_mint = teacher.vit(x_mint, y.repeat_interleave(batch_size))
                # p0mint_teacher = F.softmax(logits_teacher_mint, dim=2).detach()
                # log_p0mint_teacher = F.log_softmax(logits_teacher_mint, dim=2).detach()
                # logits_student_mint = self.process_student(x_mint,y.repeat_interleave(batch_size))
                # log_p0mint_student_lam = F.log_softmax(logits_student_mint, dim=2)
                # To save memory, we avoid sampling again
                p0mint_teacher = p0t_teacher.repeat_interleave(batch_size, dim=0)
                log_p0mint_teacher = log_p0t_teacher.repeat_interleave(batch_size, dim=0)
                log_p0mint_student_lam = log_p0t_student_lam
                distil_loss = (p0mint_teacher * (log_p0mint_teacher - log_p0mint_student_lam)).view(B, -1).sum(dim=1) / batch_size # (B,)

                # --------------- Consistency loss --------------------
                p0u_student_lam = F.softmax(logits_student_teacher, dim=2)
                log_p0u_student_lam = F.log_softmax(logits_student_teacher, dim=2) # (B*batch_size, D, S)
                p0u_student = p0u_student_lam.view(B, batch_size, D, S).mean(dim=1).detach()
                log_p0u_student = torch.logsumexp(log_p0u_student_lam.view(B, batch_size, D, S), dim=1) - np.log(batch_size)
                if use_cv:
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
                    if use_cv:
                        cll = torch.logsumexp(cll, dim=1) - np.log(batch_size) - (cll_.logsumexp(dim=2) - np.log(batch_size)).sum(dim=1)
                    else:
                        cll = torch.logsumexp(cll, dim=1) - np.log(batch_size)
                    consis_loss_cor -= cll / x_0.shape[1] # (B,)
                consis_loss = consis_loss_cor + consis_loss_indep # (B,)

                # Combine losses
                alpha_t = self.args.alpha_t
                if alpha_t == "sigmoid":
                    time_coeff = torch.sigmoid(20*(0.5-r))
                elif alpha_t == "linear":
                    time_coeff = 1 - r
                else:
                    time_coeff = 1.0
                time_coeff *= self.args.alpha_const
                
                data_loss = time_coeff*data_loss_cor + data_loss_indep

                distil_delta = self.args.r_delta # 1 / self.args.teacher_steps
                r_idx = r >= 1 - distil_delta
                r_weight = torch.zeros_like(r)
                r_weight[r_idx] = 1
                distil_loss = (1/distil_delta) * r_weight * distil_loss
                consis_loss = (1-r_weight)*consis_loss

                loss = torch.mean(distil_loss + data_loss + consis_loss)

            # Update weight if accumulation of gradient is done
            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()

            self.scaler.scale(loss).backward()

            if update_grad:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())

            distil_list.append(torch.mean(distil_loss).item())
            data_list.append(torch.mean(data_loss).item())
            data_cor_list.append(torch.mean(data_loss_cor).item())
            consis_list.append(torch.mean(consis_loss).item())
            consis_cor_list.append(torch.mean((1-r_weight)*consis_loss_cor).item())
            if len(distil_list) > 100:
                distil_list.popleft()
                data_list.popleft()
                data_cor_list.popleft()
                consis_list.popleft()
                consis_cor_list.popleft()
            if self.args.iter % 50 == 0 and self.args.is_master:
                print(np.mean(distil_list), np.mean(data_list), np.mean(data_cor_list), np.mean(consis_list), np.mean(consis_cor_list))

            # Logs
            # if update_grad and self.args.is_master:
            #     self.log_add_scalar('Train/TotalLoss', np.array(window_loss).sum(), self.args.iter)
            #     self.log_add_scalar('Train/DataLoss', torch.mean(data_loss).item(), self.args.iter)
            #     self.log_add_scalar('Train/DistillationLoss', torch.mean(distil_loss).item(), self.args.iter)
            #     self.log_add_scalar('Train/ConsistencyLoss', torch.mean(consis_loss).item(), self.args.iter)

            # if self.args.iter % log_iter == 0 and self.args.is_master:
            #     # Generate sample for visualization
            #     gen_sample = self.sample(nb_sample=10)[0]
            #     gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
            #     self.log_add_img("Images/Sampling", gen_sample, self.args.iter)

            #     # Show reconstruction
            #     unmasked_code = torch.softmax(logits_student, -1).max(-1)[1]
            #     reco_sample = self.reco(x=x[:4], code=code[:4], 
            #                             unmasked_code=unmasked_code[:4], 
            #                             mask=mask[:4])
            #     reco_sample = vutils.make_grid(reco_sample.data, nrow=4, padding=2, normalize=True)
            #     self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

            # Save Network
            if self.args.iter % log_iter == 0 and self.args.iter > 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder+"checkpoints/{}iter.pth".format(self.args.iter),
                                iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
            if self.args.iter % log_iter == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder+"checkpoints/current.pth",
                                iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)
            
            if self.args.iter % 1000 == 0 and self.args.is_master:
                with torch.no_grad():
                    labels = [1, 7, 282, 604, 724, 179, 681, 367, 635, random.randint(0, 999)]
                    labels = torch.LongTensor(labels).to(self.args.device)
                    gen_sample, _, _ = self.sample(nb_sample=labels.size(0), labels=labels, sm_temp=self.args.sm_temp, r_temp=self.args.r_temp, w=self.args.cfg_w,
                                                    randomize=self.args.randomize, sched_mode=self.args.sched_mode, step=self.args.step, teacher=teacher)
                    gen_sample = vutils.make_grid(gen_sample, nrow=5, padding=2, normalize=True)
                    # Save image
                    img_name = self.args.vit_folder + "images/{}iter.jpg".format(self.args.iter)
                    save_image(gen_sample, img_name)
                    
            self.args.iter += 1

        return cum_loss / n

    def fit_di4c(self, teacher, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_di4c(teacher)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    def process_student(self, masked_code, y, latent_bsize=1, drop_label=None):
        """
        Process the student model with multiple latent vectors for each masked code.
        
        :param masked_code: torch.LongTensor of shape (bsize, 16, 16)
        :param y: torch.LongTensor of shape (bsize,)
        :param latent_bsize: int, number of latent vectors per masked code
        :return: torch.FloatTensor of shape (bsize*latent_bsize, 16*16, codebook_size+1)
        """
        bsize = masked_code.size(0)
        
        # Repeat masked_code and y for each latent vector
        masked_code_repeat = masked_code.repeat_interleave(latent_bsize, dim=0)
        y_repeat = y.repeat_interleave(latent_bsize, dim=0)
        drop_label_repeat = drop_label.repeat_interleave(latent_bsize, dim=0)
        
        # Generate latent vectors
        z = torch.rand(bsize * latent_bsize, self.latent_dim).to(self.args.device)
        
        # Process through the student model
        student_pred = self.vit(masked_code_repeat, y_repeat, z=z, drop_label=drop_label_repeat)
        
        # Reshape the output
        student_pred = student_pred.view(bsize*latent_bsize, self.patch_size*self.patch_size, -1)
        
        return student_pred

    def one_step_denoise(self, code, y, mask, r, r_delta=0.05, latent_bsize=10, sm_temp=1, randomize="linear", r_temp=4.5, mode="arccos", drop_label=None, cfg_w=0):
        bsize = code.size(0)
        code = code.view(bsize, self.patch_size, self.patch_size)
        mask = mask.view(bsize, self.patch_size * self.patch_size)
        
        # Calculate mask rate
        r_next = torch.clamp(r + r_delta, max=1)
        
        if mode == "linear":
            val_to_mask = 1 - r_next
        elif mode == "square":
            val_to_mask = 1 - r_next ** 2
        elif mode == "cosine":
            val_to_mask = torch.cos(r_next * math.pi * 0.5)
        elif mode == "arccos":
            val_to_mask = torch.arccos(r_next) / (math.pi * 0.5)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Calculate number of tokens to unmask
        total_tokens = self.patch_size * self.patch_size
        currently_masked = mask.sum(dim=1)
        t = ((1 - val_to_mask) * total_tokens - (total_tokens - currently_masked)).long().clamp(min=0*currently_masked, max=currently_masked)

        # logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
        #                                     torch.cat([labels, labels], dim=0),
        #                                     torch.cat([~drop, drop], dim=0),
        #                                     z=torch.cat([z, z], dim=0))
        #                     logit_c, logit_u = torch.chunk(logit, 2, dim=0)
        all_drop = torch.ones(bsize, dtype=torch.bool).to(self.args.device)
        if drop_label is None:
            drop_label = ~all_drop
        with torch.no_grad():
            if cfg_w == 0:
                logits = self.vit(code, y, drop_label=drop_label)
            else:
                logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                torch.cat([y, y], dim=0),
                                torch.cat([drop_label, all_drop], dim=0))
                logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                # Classifier Free Guidance
                _w = cfg_w * r[:, None, None] * self.args.teacher_steps / (self.args.teacher_steps - 1)
                logits = (1 + _w) * logit_c - _w * logit_u
                
        prob = torch.softmax(logits * sm_temp, -1)
    
        # Repeat prob, code, and mask for latent_bsize
        prob_repeat = prob.repeat_interleave(latent_bsize, dim=0)
        code_repeat = code.repeat_interleave(latent_bsize, dim=0)
        mask_repeat = mask.repeat_interleave(latent_bsize, dim=0)
        # t_repeat = t.repeat_interleave(latent_bsize, dim=0)
        
        # Sample the code from the softmax prediction
        distri = torch.distributions.Categorical(probs=prob_repeat)
        pred_code = distri.sample()
        
        conf = torch.gather(prob_repeat, 2, pred_code.view(bsize * latent_bsize, self.patch_size*self.patch_size, 1))
        
        if randomize == "linear":
            ratio = r.repeat_interleave(latent_bsize, dim=0) * self.args.teacher_steps / (self.args.teacher_steps - 1) # r?
            rand = r_temp * np.random.gumbel(size=(bsize * latent_bsize, self.patch_size*self.patch_size)) * (1 - ratio.unsqueeze(1).cpu().numpy())
            conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
        elif randomize == "random":
            conf = torch.rand_like(conf)
        
        # Only consider masked tokens
        conf[~mask_repeat.bool()] = -float('inf')

        conf = conf.view(bsize, latent_bsize, -1)
        mask_repeat = mask_repeat.view(bsize, latent_bsize, -1)
        code_repeat = code_repeat.view(bsize, latent_bsize, self.patch_size, self.patch_size)
        
        # Choose the predicted token with the highest confidence
        for i in range(bsize):
            if t[i] > 0:
                tresh_conf, indice_mask = torch.topk(conf[i].view(latent_bsize, -1), k=t[i], dim=-1)
                tresh_conf =  tresh_conf[:, -1]

                # Replace the chosen tokens
                conf_i = (conf[i] >= tresh_conf.unsqueeze(-1)).view(latent_bsize, self.patch_size, self.patch_size)
                f_mask = (mask_repeat[i].view(latent_bsize, self.patch_size, self.patch_size).float() * conf_i.view(latent_bsize, self.patch_size, self.patch_size).float()).bool()
                code_repeat[i][f_mask] = pred_code[i*latent_bsize:(i+1)*latent_bsize].view(latent_bsize, self.patch_size, self.patch_size)[f_mask]

        # Reshape to (bsize*latent_bsize, 16, 16)
        return code_repeat.view(bsize*latent_bsize,self.patch_size,self.patch_size), logits.view(bsize,self.patch_size*self.patch_size,-1)
        