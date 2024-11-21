# %%
DATA_PATH = "../data/texture/plane"
RESULT_PATH = "../results/texture/plane"

import sys
import os
import sys
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

mi.set_variant("cuda_ad_rgb")

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from optimizers import SpatioTemporalAdam, filter2d

# %%
# scene configuration

SCENE_CONFIG = {
    'name': 'Textured Plane',
    'path': os.path.join(DATA_PATH, 'scene.xml'),
    'key': 'plane.bsdf.alpha.data',
    'param_initial_value': 0.02,
}
key = SCENE_CONFIG['key']

scene = mi.load_file(SCENE_CONFIG['path'])
params = mi.traverse(scene)

param_ref = params[key].torch()
img_ref = mi.render(scene, seed=0, spp=1024)
img_ref_torch = img_ref.torch()

@dr.wrap_ad(source='torch', target='drjit')
def render(_texture, _spp, _spp_grad):
    params[key] = _texture
    params.update()
    image = mi.render(scene, params, spp=_spp, spp_grad=_spp_grad, seed=np.random.randint(2**31))
    return image

# %%
# experiments configuration
H_PARAMS = {
    "REND_SPP": 1024,
    "spp": 16,
    "spp_grad": 1,
    "N_it": 100,
    "OPT_use_adam": True,
    "OPT_beta1": 0.2,
    "OPT_lr": 0.1,
    "OPT_lr_half_iters": [],
    "OPT_sigma_d": 1e-1,
    "OPT_stride_levels": 5,
    "OPT_log_primal": True,
    "OPT_cross_term_off": False,
    "OPT_adam_eps": 1e-25,
    "save_interval": 10,
    "save_SPP": 128,
}

exps = {
    'adam': {
        'OPT_use_adam': True,
        'OPT_lr': 0.01,
        'OPT_beta1': 0.5,
        "plot_color": "#1982c4"
    },
    'low-pass': {
        'OPT_use_adam': False,
        'OPT_beta1': 0.2,
        "OPT_sigma_d": 1e3, #large sigmal = data term off
        "OPT_stride_levels": 3,
        "OPT_log_primal": False,
        "OPT_cross_term_off": True, #irrelevant
        "plot_color": "green",
    },
    'bilateral': {
        'OPT_use_adam': False,
        'OPT_beta1': 0.2,
        "OPT_sigma_d": 1e-0,
        "OPT_stride_levels": 3,
        "OPT_log_primal": True,
        "OPT_cross_term_off": True,
        "plot_color": "orange",
    },
    'cross_bilateral(ours)': {
        'OPT_use_adam': False,
        'OPT_beta1': 0.2,
        "OPT_sigma_d": 1e-1,
        "OPT_stride_levels": 5,
        "OPT_log_primal": True,
        "OPT_cross_term_off": False,
        "plot_color": "#ff595e",
    },
}

exp_names = ["adam", "low-pass", "bilateral", "cross_bilateral(ours)"]
# %%
# run the optimizations
def loss_func(img, img_ref):
    return torch.mean(torch.square(img - img_ref) / (torch.square(img_ref) + 1e-2))

def optimize(config, h_params, out_dir):
    np.random.seed(0)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'config': config,
            'h_params': h_params,
        }, f, indent=4)

    params[key] = dr.full(type(params[key]), config['param_initial_value'], params[key].shape)
    params.update();

    # add noise to the initial texture
    texture = torch.tensor(params[key], device='cuda')
    noise = torch.rand_like(texture) * 0.2
    texture += noise;
    texture.clamp_(0.005, 0.5); texture.requires_grad_(True)

    spp = h_params['spp']
    spp_grad = h_params['spp_grad']
    N_it = h_params['N_it']

    # Setup optimizer
    lr = h_params['OPT_lr']
    beta1 = h_params['OPT_beta1']
    beta2 = 1.0 - (1 - beta1)**2
    betas = (beta1, beta2)
    sigma_d = h_params['OPT_sigma_d']
    ADAM_EPS = h_params['OPT_adam_eps']
    stride_levels = h_params['OPT_stride_levels']

    opt = None
    if h_params['OPT_use_adam']:
        opt = torch.optim.Adam([texture], lr=lr, eps=ADAM_EPS, betas=betas)
    else:
        opt = SpatioTemporalAdam([texture], lr=lr, stride_levels=stride_levels, sigma_d=sigma_d, eps=ADAM_EPS, betas=betas, log_primal=h_params['OPT_log_primal'], cross_term_off=h_params['OPT_cross_term_off'])

    losses = []
    param_errs = []
    for it in range(N_it):
        # decay learning rate if applicable:
        if it in h_params["OPT_lr_half_iters"]:
            for group in opt.param_groups:
                group["lr"] /= 2.0

        opt.zero_grad()
        img = render(texture, spp, spp_grad)
        loss = loss_func(img, img_ref_torch)
        loss.backward()
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        with torch.no_grad():
            texture.clamp_(0.001, 1.0)
        losses.append(loss.item())
        param_err_curr = torch.mean(torch.abs(texture - param_ref)).item()
        param_errs.append(param_err_curr)
        if it % 10 == 0:
            print(f"Iteration {it:02d}, loss={loss:6f}, error={param_err_curr:6f}")
    torch.save(
        {'loss': losses, 'error': param_errs, 'texture': texture},
        os.path.join(out_dir, "errors.pt")
    )
    return texture.detach().clone()


out_dir_base = RESULT_PATH
for exp_name in exps:
    from copy import deepcopy
    h_params_curr = deepcopy(H_PARAMS)
    for h_params_key in exps[exp_name]:
        h_params_curr[h_params_key] = exps[exp_name][h_params_key]
    print(exp_name)
    print(json.dumps(h_params_curr, indent=4))
    out_dir = os.path.join(out_dir_base, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    optimize(SCENE_CONFIG, h_params_curr, out_dir)
# %%
# results visualization
scene = mi.load_file(SCENE_CONFIG['path'])
params = mi.traverse(scene)
for exp_name in exps:
    ckpt_ = torch.load(os.path.join(out_dir_base, exp_name, 'errors.pt'))
    texture_ = ckpt_['texture']
    img_ = render(texture_, H_PARAMS['REND_SPP'], 1).detach()
    err_ = ((img_ - img_ref_torch)/(img_ref_torch + 1e-8)).square().mean(axis=-1).cpu()
    exps[exp_name]['texture'] = texture_
    exps[exp_name]['img'] = img_
    exps[exp_name]['err'] = err_
    exps[exp_name]['losses'] = ckpt_['loss']

plt.clf();
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'font.size': 12,
    'pgf.texsystem': 'pdflatex'
})
cs = ["#ff595e", "#1982c4"]

vmax = 0.2

fig, ax = plt.subplots(3, 5, figsize=(4*2, 3*1.7), constrained_layout=True)
[a.set_axis_off() for a in ax.ravel()]
vmin = 0
vmax_err = 0.2
vmax_tex = 1
ax[0, 4].imshow(img_ref_torch.cpu()**(1/2.2))
ax[0, 4].set_title("reference")
ax[2, 4].imshow(param_ref.cpu().detach().cpu()**(1/2.2), vmin=vmin, vmax=vmax_tex, cmap="viridis")
for exp_idx, exp_name in enumerate(exp_names):
    ax[0, exp_idx].imshow(exps[exp_name]['img'].cpu()**(1/2.2))
    ax[0, exp_idx].set_title(exp_name)
    ax[1, exp_idx].imshow(exps[exp_name]['err'].cpu(), vmin=0, vmax=vmax)
    ax[2, exp_idx].imshow(exps[exp_name]['texture'].cpu().detach().cpu()**(1/2.2), vmin=vmin, vmax=vmax_tex, cmap="viridis")
plt.savefig(os.path.join(out_dir_base, 'recovery_comparison.pdf'), bbox_inches='tight', pad_inches=0.1)
# %%
# loss visualization
plt.figure(figsize=(6.5, 4))
plt.clf();

for exp_name in exp_names:
    plt.semilogy(exps[exp_name]['losses'], label=exp_name, color=exps[exp_name]['plot_color'])
plt.grid(True, which='both', linewidth=0.2, color='gray')
plt.title('Loss');
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir_base, 'loss.pdf'), bbox_inches='tight', pad_inches=0.1)
