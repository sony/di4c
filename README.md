# Di4C: Distillation of Discrete Diffusion through Dimensional Correlations

This repository contains the code used in the paper "Distillation of Discrete Diffusion through Dimensional Correlations":
- Paper is available on [arXiv](https://arxiv.org/abs/2410.08709).
- It was also presented at the [NeurIPS 2024 Compression Workshop](https://openreview.net/forum?id=ibxO5X7kxc).


This repository is organized as follows (Section numbers follow the arXiv version):
- `tauldr/` contains the code for Section 5.1, which is based on [tauLDR](https://github.com/andrew-cr/tauLDR).
- `maskgit-pytorch/` contains the code for Section 5.2, which is based on [MaskGIT-pytorch](https://github.com/valeoai/Maskgit-pytorch).
- `sdtt/` contains the code for Section 5.3, which is based on [SDTT](https://github.com/jdeschena/sdtt/).

In each repository, we provide an implementation of mixture modeling on top of the teacher model and the Di4C training/inference scripts.

## Citation

```bibtex
@article{hayakawa2024distillation,
  title={Distillation of Discrete Diffusion through Dimensional Correlations},
  author={Hayakawa, Satoshi and Takida, Yuhta and Imaizumi, Masaaki and Wakaki, Hiromi and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2410.08709},
  year={2024}
}
```