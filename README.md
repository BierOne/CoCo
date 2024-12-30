# Generalization Beyond Feature Alignment: Concept Activation-Guided Contrast Learning (TIP'24)

Official PyTorch implementation of [Concept Activation-Guided Contrast Learning](https://arxiv.org/abs/2211.06843).

Note that this project is built upon [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414).

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## How to Run

`scripts/train_ours_coverage.py` script conducts leave-one-out cross-validations for the target domain.

```sh
CUDA_VISIBLE_DEVICES=0 bash scripts/test_runs.sh CoCo_SelfReg PACS
```

If you want to search for the best implementation, please use the sweep script provided by Domainbed:

```sh
CUDA_VISIBLE_DEVICES=0 bash scripts/sweep_runs.sh CoCo_SelfReg PACS
```



## Citation

Please cite the paper if you find the code helpful:

```
@article{tip-LiuTLW24,
  author       = {Yibing Liu and
                  Chris Xing Tian and
                  Haoliang Li and
                  Shiqi Wang},
  title        = {Generalization Beyond Feature Alignment: Concept Activation-Guided
                  Contrastive Learning},
  journal      = {{IEEE} Trans. Image Process.},
  volume       = {33},
  pages        = {4377--4390},
  year         = {2024}
}
```

## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.