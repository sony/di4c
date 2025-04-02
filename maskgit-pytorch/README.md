# Masked Image Modeling on VQ space

This repository is based on the implementation of https://github.com/valeoai/Maskgit-pytorch.

## Usage

1. Install requirement 

   ```bash
   conda env create -f environment.yaml
   conda activate maskgit
   ```

2. Download the pretrained model (**of MaskGIT-pytorch**) for teacher

   ```bash
   python download_models.py
   ```

3. Download the ImageNet dataset. If necessary, reorganize the filenames/structure so that `Trainer/trainer.py` can process them.

4. Run the Di4C training with two GPUs (default: **di4c-d** hyperparameters in the paper)

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 di4c_main.py --is_student --vit-folder "folder/to/save/your/model"
   ```

5. Evaluate the checkpoint
   ```bash
   torchrun --standalone --nnodes=1 --nproc_per_node=4 di4c_main.py --resume --test-only --is_student --vit-folder "path/to/checkpoint" --bsize 64 --step 4 --cfg_w 6
   ```