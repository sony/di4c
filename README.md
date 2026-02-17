# Di4C: Distillation of Discrete Diffusion through Dimensional Correlations [ICML 2025]

This repository contains the code used in the paper "Distillation of Discrete Diffusion through Dimensional Correlations":
- Paper is available on [ICML proceedings](https://proceedings.mlr.press/v267/hayakawa25a.html) and [arXiv](https://arxiv.org/abs/2410.08709).
- It was also presented at the [NeurIPS 2024 Compression Workshop](https://openreview.net/forum?id=ibxO5X7kxc).


This repository is organized as follows (Section numbers follow the arXiv version):
- `tauldr/` contains the code for Section 5.1, which is based on [tauLDR](https://github.com/andrew-cr/tauLDR).
- `maskgit-pytorch/` contains the code for Section 5.2, which is based on [MaskGIT-pytorch](https://github.com/valeoai/Maskgit-pytorch).
- `sdtt/` contains the code for Section 5.3, which is based on [SDTT](https://github.com/jdeschena/sdtt/).

In each repository, we provide an implementation of mixture modeling on top of the teacher model and the Di4C training/inference scripts.

## Model checkpoints

The Di4C-distilled model checkpoints are available on [Zenodo](https://zenodo.org/records/15124163) as follows:
- `tldr-di4c.pt` is the `student` model in Section 5.1 (Table 1).
- `maskgit-di4c-d.pth` is the `di4c-d` model in Section 5.2 (Figure 3).
- `sdtt6-di4c2.ckpt` is the `sdtt-6 + di4c^2` model in Section 5.3 (Figure 4).
- `sdtt7-di4c2.ckpt` is the `sdtt-7 + di4c^2` model in Section 5.3 (Figure 4).

## Citation

```bibtex
@inproceedings{hayakawa2025distillation,
  title={Distillation of Discrete Diffusion through Dimensional Correlations},
  author={Hayakawa, Satoshi and Takida, Yuhta and Imaizumi, Masaaki and Wakaki, Hiromi and Mitsufuji, Yuki},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  pages={22259--22297}
  year={2025}
}
```
