# %%
DATA_PATH = "../data/volume/janga"
RESULT_PARENT_PATH = "../results/volume/janga"


import sys
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi
import os
import torch
import json
from copy import deepcopy

mi.set_variant('cuda_ad_rgb')

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
  sys.path.append(module_path)

from mitsuba import ScalarTransform4f as T

# %%
# experiment configuration

keys = {
    'sigmat': 'object.interior_medium.sigma_t.data',
    'albedo': 'object.interior_medium.albedo.data',
}

CONFIG_BASE = {
  "sigmat_ref": os.path.join(DATA_PATH, 'janga-smoke-264-136-136.vol'),
  "albedo_ref": os.path.join(DATA_PATH, 'albedo-noise-256-128-128.vol'),
  "OPT_lr_half_iters": [50, 100, 150],
  "iteration_count": 200,
  "spp_grad": 1,
  "spp": 16,
  "OPT_EPS": 1e-15,
  "save_interval": 10,
  "save_SPP": 128,
  "save_sensor_idx": [0, 5, 10],
  "envmap": os.path.join(DATA_PATH, 'autumn_field_puresky_1k.exr'),
  "ref_spp": 512,
  "sensor_count": 64,
  "render_res_comparison": 512,
  "render_res": 128,
  "voxel_res": 64,
  "sigmat_init": 0.002,
  "albedo_init": 0.6,
}

exps = {
  "adam": {
    "OPT_beta1": 0.2,
    "OPT_lr": 0.008,
    "OPT_upsample_iters": []
  },
  "bilateral-filter": {
    "OPT_beta1": 0.2,
    "OPT_lr": 0.02,
    "OPT_stride_levels": 3,
    "OPT_sigma_d_1d": 1e-3, #bf
    "OPT_sigma_d_3d": 2e-7, #bf
    "OPT_sigma_d_final": -1,
    "OPT_cross_term_off": True,
    "OPT_postfilter": True,
    "OPT_prefilter": False,
    "OPT_upsample_iters": []
  },
  "low-pass-filter": {
    "OPT_beta1": 0.2,
    "OPT_lr": 0.02,
    "OPT_stride_levels": 3,
    "OPT_sigma_d_1d": 1e1, #lowpass
    "OPT_sigma_d_3d": 1e1, #lowpass
    "OPT_sigma_d_final": -1,
    "OPT_cross_term_off": False,
    "OPT_postfilter": True,
    "OPT_prefilter": False,
    "OPT_upsample_iters": []
  },
  "cross-bilateral-filter": {
    "OPT_beta1": 0.2,
    "OPT_lr": 0.02,
    "OPT_stride_levels": 3,
    "OPT_sigma_d_1d": 1e-3,
    "OPT_sigma_d_3d": 2e-1,
    "OPT_sigma_d_final": -1,
    "OPT_cross_term_off": False,
    "OPT_postfilter": True,
    "OPT_prefilter": False,
    "OPT_upsample_iters": []
  }
}

os.makedirs(RESULT_PARENT_PATH, exist_ok=True)

# %%
# running optimization

for exp_name in exps:
    SP = deepcopy(CONFIG_BASE)
    for SP_key in exps[exp_name]:
        SP[SP_key] = exps[exp_name][SP_key]

    print(json.dumps(SP, indent=4))

    RESULT_PATH = os.path.join(RESULT_PARENT_PATH, exp_name)
    os.makedirs(RESULT_PATH, exist_ok=True)

    with open(os.path.join(RESULT_PATH, 'config.json'), 'w') as file:
        file.write(json.dumps(SP, indent=4))

    sensor_count = SP['sensor_count']

    batch_sensor_dict = {}
    batch_sensor_dict['type'] = 'batch'
    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensor_rotation = T.rotate([0, 1, 0], angle)
        sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
        batch_sensor_dict[f"sensor{i}"] = {
            'type': 'perspective',
            'fov': 45,
            'to_world': sensor_rotation @ sensor_to_world,
        }
    batch_sensor_dict['film'] = {
                'type': 'hdrfilm',
                'width': SP['render_res'] * sensor_count,
                'height': SP['render_res'],
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
    batch_sensor = mi.load_dict(batch_sensor_dict)

    # creating a dummy reference scene to be able to access sigma_t and albedo
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'prbvolpath'},
        'object': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': SP['sigmat_ref'],
                    'to_world': T.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                },
                'albedo': {
                    'type': 'gridvolume',
                    'filename': SP['albedo_ref'],
                    'to_world': T.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                },
                'scale': 40
            }
        },
        'emitter': {'type': 'envmap', 'filename': SP['envmap']}
    }

    scene_ref = mi.load_dict(scene_dict)

    ref_spp = SP['ref_spp']

    params_ref = mi.traverse(scene_ref)

    GSZ = SP['voxel_res']
    import torch.nn.functional as F
    sigmat_ref = torch.tensor(params_ref[keys['sigmat']], device='cuda').squeeze().unsqueeze(0).unsqueeze(0)
    sigmat_ref = F.interpolate(sigmat_ref,  (GSZ,GSZ,GSZ), mode='trilinear')
    sigmat_ref = sigmat_ref.squeeze().unsqueeze(-1)

    albedo_ref = torch.tensor(params_ref[keys['albedo']], device='cuda').squeeze().permute(3,0,1,2).unsqueeze(0)
    albedo_ref = F.interpolate(albedo_ref,  (GSZ,GSZ,GSZ), mode='trilinear')
    albedo_ref = albedo_ref.squeeze().permute(1,2,3,0)

    params_ref[keys['sigmat']] = sigmat_ref
    params_ref[keys['albedo']] = albedo_ref
    params_ref.update()

    # creating the acutal reference scene with the correct sigmat and albedo
    scene_dict_new = {
        'type': 'scene',
        'integrator': {'type': 'prbvolpath'},
        'object': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'grid': mi.VolumeGrid(params_ref[keys['sigmat']]),
                    'to_world': T.translate(-1).scale(2.0)
                },
                'albedo': {
                    'type': 'gridvolume',
                    'grid': mi.VolumeGrid(params_ref[keys['albedo']]),
                    'to_world': T.translate(-1).scale(2.0)
                },
                'scale': 40.0,
            },
        },
        'emitter': {'type': 'envmap', 'filename': SP['envmap']}
    }
    scene_ref = mi.load_dict(scene_dict_new)


    ref_image_batch = mi.render(scene_ref, sensor=batch_sensor, spp=ref_spp)
    @dr.wrap_ad(source='torch', target='drjit')
    def render(sigmat_, albedo_, it, sensor_idx, spp=512, spp_grad=1):
        params[keys['sigmat']] = sigmat_
        params[keys['albedo']] = albedo_
        params.update()
        image = mi.render(scene, params, sensor=batch_sensor, spp=spp, spp_grad=spp_grad, seed=it)
        return image

    v_res = SP['voxel_res'] // (2**len(SP["OPT_upsample_iters"]))

    # create the initialized scene for optimization
    scene_dict['object'] = {
        'type': 'cube',
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'grid': mi.VolumeGrid(dr.full(mi.TensorXf, SP['sigmat_init'], (v_res, v_res, v_res, 1))),
                'to_world': T.translate(-1).scale(2.0)
            },
            'albedo': {
                'type': 'gridvolume',
                'grid': mi.VolumeGrid(dr.full(mi.TensorXf, SP['albedo_init'], (v_res, v_res, v_res, 3))),
                'to_world': T.translate(-1).scale(2.0)
            },
            'scale': 40.0,
        },
        'bsdf': {'type': 'null'}
    }

    scene = mi.load_dict(scene_dict)

    import torch
    import torch.nn as nn
    from optimizers import SpatioTemporalAdamVol, filter3d

    params = mi.traverse(scene)
    ADAM_EPS = SP['OPT_EPS']
    beta1 = SP['OPT_beta1']
    beta2 = 1.0 - (1 - beta1)**2
    betas = (beta1, beta2)
    learning_rate = SP['OPT_lr']


    sigmat = torch.tensor(params[keys['sigmat']], device='cuda')
    sigmat.requires_grad_(True);
    albedo = torch.tensor(params[keys['albedo']], device='cuda')
    albedo.requires_grad_(True);

    opt = None
    if "adam" in exp_name:
        opt = torch.optim.Adam([sigmat, albedo], lr=learning_rate, betas=betas, eps=ADAM_EPS)
    else:
        opt_postfilter = SP['OPT_postfilter']
        opt_prefilter = SP['OPT_prefilter']
        opt_sigma_d_1d = SP['OPT_sigma_d_1d']
        opt_sigma_d_3d = SP['OPT_sigma_d_3d']
        stride_levels = SP['OPT_stride_levels']
        cross_term_off = SP['OPT_cross_term_off']
        opt = SpatioTemporalAdamVol([sigmat, albedo], lr=learning_rate, stride_levels=stride_levels, betas=betas, eps=ADAM_EPS, sigma_d_1d=opt_sigma_d_1d, sigma_d_3d=opt_sigma_d_3d, cross_term_off=cross_term_off, prefilter=opt_prefilter, postfilter=opt_postfilter)

    def save_volumes(s, a, save_dir, name):
        torch.save(
            {'s': s, 'a': a},
            os.path.join(save_dir, name)
        )


    ref_image_batch_torch = ref_image_batch.torch()
    losses = []
    param_errs_albedo = []
    param_errs_sigmat = []
    for it in range(SP['iteration_count']):
        # decay learning rate if applicable:
        if it in SP["OPT_lr_half_iters"]:
            for group in opt.param_groups:
                group["lr"] /= 2.0

        if it in SP["OPT_upsample_iters"]:
            sz = sigmat.shape[0]
            sigmat.requires_grad = False
            albedo.requires_grad = False

            sigmat = sigmat.permute(3,0,1,2).unsqueeze(0)
            sigmat = F.interpolate(sigmat,  (sz*2,sz*2,sz*2), mode='trilinear')
            sigmat = sigmat.squeeze(0).permute(1,2,3,0)
            sigmat = sigmat.contiguous()

            albedo = albedo.permute(3,0,1,2).unsqueeze(0)
            albedo = F.interpolate(albedo,  (sz*2,sz*2,sz*2), mode='trilinear')
            albedo = albedo.squeeze(0).permute(1,2,3,0)
            albedo = albedo.contiguous()

            sigmat.requires_grad = True
            albedo.requires_grad = True

            if "adam" in exp_name:
                opt = torch.optim.Adam([sigmat, albedo], lr=learning_rate, betas=betas, eps=ADAM_EPS)
            else:
                opt_postfilter = SP['OPT_postfilter']
                opt_prefilter = SP['OPT_prefilter']
                opt_sigma_d_1d = SP['OPT_sigma_d_1d']
                opt_sigma_d_3d = SP['OPT_sigma_d_3d']
                stride_levels = SP['OPT_stride_levels']
                cross_term_off = SP['OPT_cross_term_off']
                opt = SpatioTemporalAdamVol([sigmat, albedo], lr=learning_rate, stride_levels=stride_levels, betas=betas, eps=ADAM_EPS, sigma_d_1d=opt_sigma_d_1d, sigma_d_3d=opt_sigma_d_3d, cross_term_off=cross_term_off, prefilter=opt_prefilter, postfilter=opt_postfilter)

        total_loss = 0.0
        for sensor_idx in range(1):
            opt.zero_grad()
            img = render(sigmat, albedo, it, sensor_idx, spp=SP['spp'], spp_grad=SP['spp_grad'])
            loss = torch.mean(torch.square(img[...,:3] - ref_image_batch_torch[...,:3]))

            loss.backward()

            total_loss += loss.item()
        opt.step()
        with torch.no_grad():
            sigmat.clamp_(1e-6, 1.0)
            albedo.clamp_(0.0, 1.0)

        losses.append(total_loss)

        print(f"Experiment name {exp_name}, Iteration {it:02d}, loss={total_loss:6f}")


        save_volumes(sigmat, albedo, RESULT_PATH, f'{it}.pt')
    torch.save(
        {'loss': losses, 'sigma': param_errs_sigmat, 'albedo': param_errs_albedo},
        os.path.join(RESULT_PATH, "errors.pt")
    )

# %%
# loss visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.clf();
colors = {
    'adam': '#1982c4',
    'low-pass-filter': 'green',
    'bilateral-filter': 'orange',
    'cross-bilateral-filter': '#ff595e',
}

exp_names = ["adam", "low-pass-filter", "bilateral-filter", "cross-bilateral-filter"]
for exp_name in exp_names:
    RESULT_PATH = os.path.join(RESULT_PARENT_PATH, exp_name)
    ckpt_errs = torch.load(os.path.join(RESULT_PATH, f"errors.pt"))
    loss = torch.tensor(ckpt_errs['loss'])[:CONFIG_BASE['iteration_count']]
    c = None
    for ck in colors:
        if ck in exp_name:
            c = colors[ck]
    plt.semilogy(loss, label=exp_name, c=c);
plt.grid(True, which='both', linewidth=0.2, color='gray')
plt.title('Loss');
plt.ylim(1.3e-3,20e-3)
plt.legend(loc=(1,1))
plt.tight_layout()

plt.savefig(os.path.join(RESULT_PARENT_PATH, f'volume_loss.pdf'), bbox_inches='tight', pad_inches=0.1)

# %%
# results visualization
scene_dict = {
    'type': 'scene',
    'integrator': {'type': 'prbvolpath'},
    'object': {
        'type': 'cube',
        'bsdf': {'type': 'null'},
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'filename': SP['sigmat_ref'],
                'to_world': T.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
            },
            'albedo': {
                'type': 'gridvolume',
                'filename': SP['albedo_ref'],
                'to_world': T.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
            },
            'scale': 40.0,
        },
    },
    'emitter': {'type': 'envmap', 'filename': SP['envmap']}
}
scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)


sigmat_ref = torch.tensor(params[keys['sigmat']], device='cuda').detach()
albedo_ref = torch.tensor(params[keys['albedo']], device='cuda').detach()

sensor_count = SP['sensor_count']
sensors = []

fov = 15

SP['render_res_comparison'] = 512*3

for i in range(sensor_count):
    angle = 360.0 / sensor_count * i - 90.0
    sensor_rotation = T.rotate([0, 1, 0], angle)
    sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': sensor_rotation @ sensor_to_world,
        'film': {
            'type': 'hdrfilm',
            'width': SP['render_res_comparison'], 'height': SP['render_res_comparison'],
            'filter': {'type': 'tent'}
        }
    }))
for i in range(sensor_count):
    angle = 360.0 / sensor_count * i - 90.0
    sensor_rotation = T.rotate([0, 1, 0], angle)
    sensor_to_world = T.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
    sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 45,
        'to_world': sensor_rotation @ sensor_to_world,
        'film': {
            'type': 'hdrfilm',
            'width': SP['render_res_comparison'], 'height': SP['render_res_comparison'],
            'filter': {'type': 'tent'}
        }
    }))

@dr.wrap_ad(source='torch', target='drjit')
def render_single(sigmat_, albedo_, it, sensor_idx, spp=512, spp_grad=1):
    params[keys['sigmat']] = sigmat_
    params[keys['albedo']] = albedo_
    params.update()
    image = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, spp_grad=spp_grad, seed=it)
    return image

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'font.size': 12,
    'pgf.texsystem': 'pdflatex'
})

exp_names = ["adam", "low-pass-filter", "bilateral-filter", "cross-bilateral-filter"]
img_idxs = [0, 10, 20, 30, 40, 50, 190] # janga smoke
K = len(exp_names)
N = len(img_idxs)
sensor_idx = 60
PRINT_SPP = 64 # change this to 512 for final images
fig = plt.figure(figsize=((N + 1)*K , K*4))
with torch.no_grad():
    plt.subplot(K, N + 1, N + 1)
    img_ref = render_single(sigmat_ref, albedo_ref, 0, sensor_idx, spp=PRINT_SPP, spp_grad=1)
    plt.imshow(img_ref.cpu()[...,:3]**(1/2.2))
    plt.xticks([])
    plt.yticks([])
    plt.title('Reference')

    for j, exp_name in enumerate(exp_names):
        RESULT_PATH = os.path.join(RESULT_PARENT_PATH, exp_name)
        for i in range(N):
            ckpt = torch.load(os.path.join(RESULT_PATH, f"{img_idxs[i]}.pt"))
            ckpt = {k: ckpt[k].cuda() for k in ckpt}
            plt.subplot(K, N + 1, 1 + i + (N + 1)*j)
            img_adam = render_single(ckpt['s'], ckpt['a'], 0, sensor_idx, spp=PRINT_SPP, spp_grad=1)
            plt.imshow(img_adam.cpu()[...,:3]**(1/2.2))
            plt.imsave(os.path.join(RESULT_PATH, f"image.png"), (img_adam.cpu()[...,:3]**(1/2.2)).clip(0,1).numpy())

            if j == 0:
                plt.title(f'{img_idxs[i]}')
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel(exp_name)

    sensor_idx += 64
    plt.subplot(K, N + 1, 2*(N + 1))
    img_ref = render_single(sigmat_ref, albedo_ref, 0, sensor_idx, spp=PRINT_SPP, spp_grad=1)
    plt.imshow(img_ref.cpu()[...,:3]**(1/2.2))
    plt.xticks([])
    plt.yticks([])


    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.suptitle('Iterations')

plt.savefig(os.path.join(RESULT_PARENT_PATH, f'volume_crops.pdf'), bbox_inches='tight', pad_inches=0.1)
