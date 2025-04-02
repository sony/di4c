# Discretized Gaussian Diffusion on Pixel Space

This repository is based on the implementation of https://github.com/andrew-cr/tauLDR.

## Usage

1. Install requirement

    ```bash
    conda env create --file tldr_env.yml
    ```

2. Download the pretrained model (**of tauLDR**) for teacher from https://www.dropbox.com/scl/fo/zmwsav82kgqtc0tzgpj3l/h?dl=0&rlkey=k6d2bp73k4ifavcg9ldjhgu0s into the folder `cifar10/checkpoints/`.

3. Download the CIFAR-10 dataset (raw pngs) into `cifar10/samples`.

4. Run the Di4C training with a single GPU. You can modify the setting by changing some items in the configuration list `c_cfg`. The checkpoints are saved into the folder `results/distil`.

    ```bash
    python train_distil.py cifar10
    ```

5. With `scripts/distil_sample.py`, you can run the analytical sampling of your model. You should modify `checkpoint_path` around the 65th line. 

    Sample from your student model with 10 steps: 
    ```bash
    python scripts/distil_sample.py -d cuda:0 -n 10
    ```
    You can set `-l` option to control the number of teacher steps in the **hybrid** model in the paper. The following example is 7-step student + 3-step teacher:
    ```bash
    python scripts/distil_sample.py -d cuda:0 -n 10 -l 3
    ```
    Your images will be saved into `results/distil/cuda:0/10samples` in the first case, and `results/distil/cuda:0/7_3samples` in the second case.

    You can also set `distil_model = False` at around the 39th line to sample from the teacher. In that case, your images will be saved in `results/10step_baseline`.

6. To evaluate the FID, you can run the following for a quick evaluation:

    ```bash
    python -m pytorch_fid --device cuda:0 path/to/your/samples cifar10/samples
    ```

    For our numbers in the paper, we used the code from https://github.com/w86763777/pytorch-image-generation-metrics, following the description in the original paper of tauLDR.