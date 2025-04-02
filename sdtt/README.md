# Masked Diffusion Language Models

This repostory is based on the implementation of https://github.com/jdeschena/sdtt/.

## Usage
1. Install the SDTT codebase.

    ```bash
    conda env create -f requirements.yaml
    conda activate sdtt
    pip install flash-attn
    pip install -e .
    ```

2. The pretrained models can be downloaded from HuggingFace in the first call.

3. Set the path `src/sdtt/configs/config.data_preprocess.data_cache` for the OpenWebText data cache (absolute path recommended).

4. Run the Di4C training with two GPUs, with the teacher **sdtt-7** model.

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python src/sdtt/main.py mode=train is_di4c=true round=7
    ```

    Alternatively, you can run another round of Di4C training with your trained di4c checkpoint as a teacher.

    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python src/sdtt/main.py mode=train is_di4c=true is_teacher_di4c=true parameterization.checkpoint_path=/path/to/di4c/checkpoint
    ```

5. For evaluating your checkpoint, you can just run the following.

    ```bash
    python src/sdtt/main.py mode=sample parameterization.checkpoint_path=/path/to/checkpoint is_di4c=true
    ```

    You can also evaluate the baseline performance of **sdtt-7**. Note that `round` is only valid with 6 or 7.

    ```bash
    python src/sdtt/main.py mode=sample_pretrained round=7
    ```

    In the default setting, the number of sampling steps is set to 16. Edit `num_steps` in `src/sdtt/configs/parameterization/multi-round-sdtt.yaml` to change it.