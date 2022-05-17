# CDLNet-OJSP

Code for the papers, 
- [*CDLNet: Noise-Adaptive Convolutional Dictionary Learning
Network for Blind Denoising and Demosaicing*](https://ieeexplore.ieee.org/document/9769957/).
- [*Gabor is Enough: Interpretable Deep Denoising with a Gabor Synthesis Dictionary Prior*](https://arxiv.org/abs/2204.11146)

## Workflow
Install packages listed in `reqiurements.txt` to replicate the environment.

Edit the example `args.json` file to define your own model and training.
Architecture hyperparameter notation follows the journal article (K -
iterations, M - subbands, C - input-channels, P - filter side-length, etc.).
Paramters in `args['model']` will be passed directly to the constructor of the
network, defined in `model/net.py`. The option `args['model_type']` can be
`['CDLNet', 'GDLNet', 'JDD_CDLNet', 'DnCNN', 'FFDNet']`. Note that DnCNN and FFDNet implementations are provided but were not used to generate the numbers reported in the paper (see [KAIR](https://github.com/cszn/KAIR)).

Once your args file is defined, training may be performed via
```
$ python train.py path/to/args.json
```

Testing of the trained model is done via 
```
$ python analyze.py path/to/args.json --test path/to/dataset/ --noise_level 25
```
where the checkpoint referenced in `args['paths']['ckpt']` will be loaded.
Additional analysis of trained models can be performed with other command-line
options (see `$ python analyze.py -h`).

[website](https://nikopj.github.io/projects/dcdl), [supplementary material](https://nikopj.github.io/notes/cdlnet_supp)

## Publications

If you find this code/work useful, please cite us:
```
@article{janjusevicCDLNet2022,
author={Janjušević, Nikola and Khalilian-Gourtani, Amirhossein and Wang, Yao},
journal={IEEE Open Journal of Signal Processing}, 
title={{CDLNet}: Noise-Adaptive Convolutional Dictionary Learning Network for Blind Denoising and Demosaicing}, 
year={2022},
volume={},
number={},
pages={1-1},
doi={10.1109/OJSP.2022.3172842}
}
```
```
@misc{janjusevicGDLNet2022,
doi = {10.48550/ARXIV.2204.11146},
url = {https://arxiv.org/abs/2204.11146},
author = {Janjušević, Nikola and Khalilian-Gourtani, Amirhossein and Wang, Yao},
title = {Gabor is Enough: Interpretable Deep Denoising with a Gabor Synthesis Dictionary Prior},
publisher = {arXiv},
year = {2022},
}
```
