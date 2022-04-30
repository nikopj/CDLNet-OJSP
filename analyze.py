#!/usr/bin/env python3
import os, sys, json, copy, time
from pprint import pprint
import numpy as np
from numpy.fft import fftshift, fft2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import model
import model.nle
import utils, data, train

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="args.json")
parser.add_argument("--test", type=str, help="Run model over specified test set (provided path to image dir).", default=None)
parser.add_argument("--dictionary", action="store_true", help="Save image of final synthesis dictionary and magnitude freq-response.")
parser.add_argument("--passthrough", type=str, help="Example passthrough of model.", default=None)
parser.add_argument("--noise_level", type=int, nargs='*', help="Input noise-level(s) on [0,255] range. Single value required for --passthrough. If --test is used, multiple values can be specified.", default=[-1])
parser.add_argument("--blind", type=str, default=None, choices=["MAD", "PCA"], help="Blind noise-level estimation algorithm.")
parser.add_argument("--save", action="store_true", help="Save test, intermediate passthrough results to files.")
parser.add_argument("--thresholds", action="store_true", help="Plot network thresholds.")
parser.add_argument("--filters", action="store_true", help="Save network A,B filterbanks.")
parser.add_argument("--save_dir", type=str, help="Where to save analyze results.", default=None)
parser.add_argument("--color", action="store_true", help="Use color images.")
parser.add_argument("--demosaic", action="store_true", help="Demosaicing problem.")

ARGS = parser.parse_args()

def main(model_args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    net, _, _, epoch0 = train.init_model(model_args, device=device)
    net.eval()
    # ---------------------------------------------------
    # ---------------------------------------------------

    if ARGS.save_dir is None:
        ARGS.save_dir = model_args['paths']['save']

    if len(ARGS.noise_level) == 1:
        ARGS.noise_level = ARGS.noise_level[0]
    if ARGS.noise_level == -1:
        ARGS.noise_level = model_args['train']['fit']['noise_std']

    with torch.no_grad():
        if ARGS.test is not None:
            loader = data.get_data_loader([ARGS.test], load_color=ARGS.color, test=True)
            test(net, loader, noise_level=ARGS.noise_level, blind=ARGS.blind, device=device)

        if ARGS.dictionary:
            dictionary(net)

        if ARGS.passthrough is not None:
            passthrough(net, ARGS.passthrough, ARGS.noise_level, blind=ARGS.blind, demosaic=ARGS.demosaic, device=device, color=ARGS.color)

        if ARGS.thresholds:
            thresholds(net, noise_level=ARGS.noise_level)

        if ARGS.filters:
            filters(net, scale_each=True)

def test(net, loader, noise_level=25, blind=False, device=torch.device('cpu')):
    """ Evaluate net on test-set.
    """
    print("--------- test ---------")
    dset_name = os.path.basename(os.path.dirname(loader.dataset.root_dirs[0]))
    fn = os.path.join(ARGS.save_dir, f"test_{dset_name}_{blind}.txt")

    if not type(noise_level) in [range, list, tuple]:
        noise_level = [noise_level]

    for sigma in noise_level:
        t = tqdm(iter(loader), desc=f"TEST-{sigma}", dynamic_ncols=True)
        psnr = 0
        for itern, x in enumerate(t):
            x = x.to(device)
            mask = utils.gen_bayer_mask(x) if ARGS.demosaic else 1
            y, s = utils.awgn(x, sigma)
            y = mask*y
            if net.adaptive:
                if blind is not None and blind is not False:
                    sigma = 255 * model.nle.noise_level(y, method=blind)
                    print(f"sigma_hat = {sigma.flatten().item():.3f}")
                else:
                    print(f"using GT sigma.")
            else:
                sigma = None
            xhat, _ = net(y, s, mask=mask)
            psnr = psnr + -10*np.log10(torch.mean((x-xhat)**2).item())
        psnr = psnr / itern
        print(f"PSNR = {psnr:.3f}")

        with open(fn,'+a') as log_file:
            log_file.write(f"{sigma}, {psnr:.3f}\n")

    print(f"saved to file {fn}")
    print("done.")

def thresholds(net, noise_level=25):
    print("--------- thresholds ---------")
    c = 1 if net.adaptive else 0
    tau = torch.cat([net.t[k][0:1] + c*(noise_level/255)*net.t[k][1:2] for k in range(net.K)]).detach() # K, M, 1, 1
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(tau[:,:,0,0], cmap='hot', interpolation=None, vmin=0, vmax=tau.max()*1)
    plt.xlabel("j (subband)")
    plt.ylabel("k (iteration)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    S = 100
    cbar.set_ticks([0, np.round(S*tau.max()*0.5)/S, np.floor(S*tau.max()*1)/S])

    fn = os.path.join(ARGS.save_dir,"tau.png")
    print(f"saving {fn}...")
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close()

    print("done.")

def filters(net, scale_each=False):
    """ Saves ALL net filters
    """
    print("--------- filters ---------")
    save_dir = os.path.join(ARGS.save_dir, "filters")
    os.makedirs(save_dir, exist_ok=True)

    if type(net) == model.net.GDLNet:
        get_filter = lambda C: C.get_filter()
    elif type(net) == model.net.CDLNet:
        get_filter = lambda C: C.weight.data
    else:
        raise NotImplementedError

    D = get_filter(net.D)
    if type(net) == model.net.CDLNet and net.s == 1:
        D = D.permute(1,0,2,3)
    n = int(np.ceil(np.sqrt(D.shape[0])))

    # store filters in these lists
    AL = []; BL = []
    
    # get maximum over all filters
    mmax = 0
    for k in range(net.K):
        AL.append(get_filter(net.A[k]))
        B = get_filter(net.B[k])
        if type(net) == model.net.CDLNet and net.s == 1:
            B = B.permute(1,0,2,3)
        if k == 0:
            B = 0*B
        BL.append(B)

        amax = AL[k].abs().max() 
        bmax = BL[k].abs().max() 
        if amax > mmax:
            mmax = amax
        if bmax > mmax:
            mmax = bmax

    for k in range(net.K):
        vr = None if scale_each else (-mmax,mmax)
        Ag = make_grid(AL[k], nrow=n, padding=2, scale_each=scale_each, normalize=True, value_range=vr)

        if k==0:
            vr = (-1,1)
        else:
            vr = None if scale_each else (-mmax,mmax)
        Bg = make_grid(BL[k], nrow=n, padding=2, scale_each=scale_each, normalize=True, value_range=vr)

        fn = os.path.join(save_dir, f"AB{k:02d}_{scale_each}.png")
        print(f"Saving {fn} ...")
        save_image([Ag,Bg], fn, nrow=2, padding=5)

    fn = os.path.join(save_dir, f"D{k:02d}_{scale_each}.png")
    print(f"Saving {fn} ...")
    save_image(D, fn, nrow=n, scale_each=scale_each, normalize=True)
    print("done.")

def dictionary(net):
    """ Saves net dictionary's filters, frequency-response.
    """
    print("--------- dictionary ---------")
    if type(net) is model.net.CDLNet:
        if net.s > 1:
            D = net.D.weight.cpu()
        else:
            D = net.D.weight.cpu().permute(1,0,2,3)
    elif type(net) is model.net.GDLNet:
        D = net.D.get_filter().cpu()
    else:
        raise NotImplementedError

    n = int(np.ceil(np.sqrt(net.M)))

    fn = os.path.join(ARGS.save_dir, "D_learned.png")
    print(f"Saving learned dictionary to {fn} ...")
    save_image(D, fn, nrow=n, padding=2, scale_each=True, normalize=True)

    # plot frequency response of effective dictionary
    X = torch.tensor(fftshift(fft2(D.detach().numpy(), (64,64)), axes=(-2,-1)))

    fn = os.path.join(ARGS.save_dir, "freq.png")
    print(f"Saving dictionary magnitude response to {fn} ...")
    save_image(X.abs(), fn, nrow=n, normalize=True, scale_each=True, padding=10, pad_value=1)

def passthrough(net, img_path, noise_std, device=torch.device('cpu'), blind=False, color=False, demosaic=False):
    """ Save passthrough of single image
    """
    print("--------- passthrough ---------")
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    if ARGS.save:
        save_dir = os.path.join(ARGS.save_dir, f"passthrough_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

    print(f"using {img_path}...")
    x = utils.img_load(img_path, gray=not color).to(device)
    y, sigma = utils.awgn(x, noise_std)
    m = utils.gen_bayer_mask(y) if demosaic else 1
    y = m*y

    print(f"noise_std = {sigma}")
    if net.adaptive:
        if blind is not None and blind is not False:
            sigma = 255 * model.nle.noise_level(y, method=blind)
            print(f"sigma_hat = {sigma.flatten().item():.3f}")
        else:
            print(f"using GT sigma.")
    else:
        sigma = None

    n = round(np.sqrt(net.M))
    fg = net.forward_generator(y, sigma, mask=m)
    yp, params, m = model.utils.pre_process(y, net.s, mask=m)

    for (i, xz) in enumerate(fg):
        if not ARGS.save:
            continue

        if i < net.K:
            csc = xz.cpu().transpose(0,1).abs()
            fn = os.path.join(save_dir, f"csc{i:02d}.png")
            print(f"Saving csc{i:02d} at {fn} ...")
            save_image(csc, fn, nrow=n, padding=10, scale_each=False, normalize=True, value_range=(0, csc.max()))
    
    xhat = xz
    psnr = -10*np.log10(torch.mean((x-xhat)**2).item())
    print(f"PSNR = {psnr:.2f}")

    if ARGS.save:
        fn = os.path.join(save_dir, f"compare.png")
        print(f"Saving y, xhat, x at {fn} ...")
        save_image(torch.cat([y, xhat, x]), fn, nrow=3, scale_each=False, normalize=False)
    print("done.")

if __name__ == "__main__":
    """ Load arguments from json file and command line and pass to main.
    """
    # load provided args.json file
    model_args_file = open(ARGS.args_fn)
    model_args = json.load(model_args_file)
    pprint(model_args)
    model_args_file.close()
    main(model_args)

