# %%
DATA_PATH = "../data/mesh"
RESULT_PATH = "../results/mesh"

import os
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import to_differential, from_differential
from gpytoolbox import remesh_botsch
from igl import hausdorff

mi.set_variant("cuda_ad_rgb")

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from mitsuba import ScalarTransform4f as T
from largesteps_scripts.geometry import (
    remove_duplicates,
    compute_face_normals,
    compute_vertex_normals,
    average_edge_length,
)
from optimizers import SpatioTemporalAdamMesh

N_it = 120
hparams = {
    "scene": "cube",
    "remesh_initial_ratio": 1.0,
    "remesh_it": [],
    "remesh_lr_beta": 0.6,
    "scale": 0.75,
    "source_scale": 1.0,
    "source_trans": [0, 0, 0],
    "mesh_save_interval": 5,
    "sensor_count": 10,
    "lambda_": 19,
    "lr": 1e-2,
    "N_it": N_it,
    "spp": 16,
    "spp_grad": 1,
    "spp_ref": 128,
    "sigma_d": 0.5,
    "sigma_phi": 0.4,
    "prefilter": "smooth",
    "postfilter": "bilateral",
    "betas": (0.9, 1 - (1 - 0.9) ** 2),
    "face_normals": False,
}
hparams["lr_beta"] = 0.6 ** (1 / (N_it * hparams["sensor_count"]))

out_dir = os.path.join(RESULT_PATH, hparams["scene"])
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump({"hparams": hparams}, f, indent=4)

sensor_count = hparams["sensor_count"]
sensors = []

# Fibonacci lattice viewpoints: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/shape_optimization.html
golden_ratio = (1 + 5**0.5) / 2
for i in range(sensor_count):
    theta = 2 * dr.pi * i / golden_ratio
    phi = dr.acos(1 - 2 * (i + 0.5) / sensor_count)

    d = 5
    origin = [
        d * dr.cos(theta) * dr.sin(phi),
        d * dr.sin(theta) * dr.sin(phi),
        d * dr.cos(phi),
    ]

    sensors.append(
        mi.load_dict(
            {
                "type": "perspective",
                "fov": 45,
                "to_world": T.look_at(target=[0, 0, 0], origin=origin, up=[0, 1, 0]),
                "film": {
                    "type": "hdrfilm",
                    "width": 512,
                    "height": 512,
                    "stadam": {"type": "gaussian"},
                    "sample_border": True,
                },
                "sampler": {"type": "independent", "sample_count": hparams["spp"]},
            }
        )
    )

scene_dict = {
    "type": "scene",
    "integrator": {
        "type": "direct_projective",
        "sppi": 0,
        "sppc": hparams["spp_grad"],
        "sppp": hparams["spp_grad"],
    },
    "emitter": {
        "type": "envmap",
        "filename": os.path.join(DATA_PATH, hparams["scene"], "envmap.exr"),
    },
    "shape": {
        "type": "ply",
        "filename": os.path.join(
            DATA_PATH, hparams["scene"], f"{hparams['scene']}.ply"
        ),
        "face_normals": hparams["face_normals"],
        "bsdf": {"type": "twosided", "material": {"type": "diffuse"}},
    },
}


def init_opt_lr_scheduler_ours(v_unique_ours, lr, M):
    v_unique_ours.requires_grad_(True)

    opt_ours = SpatioTemporalAdamMesh(
        [
            {
                "params": v_unique_ours,
                "lr": lr,
                "betas": hparams["betas"],
                "M": M,
            }
        ],
        lr=0.0,
        sigma_d=hparams["sigma_d"],
        sigma_phi=hparams["sigma_phi"],
        prefilter=hparams["prefilter"],
        postfilter=hparams["postfilter"],
    )
    lr_scheduler_ours = torch.optim.lr_scheduler.ExponentialLR(
        opt_ours, hparams["lr_beta"]
    )
    return opt_ours, lr_scheduler_ours


def init_opt_lr_scheduler(u_unique, lr):
    u_unique.requires_grad_(True)
    opt = AdamUniform(
        [
            {
                "params": u_unique,
                "lr": lr,
                "betas": hparams["betas"],
            }
        ],
        lr=0.0,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, hparams["lr_beta"])
    return opt, lr_scheduler


def save_images(images, name):
    for i, img in enumerate(images):
        mi.util.write_bitmap(os.path.join(out_dir, f"{name}_{i}.png"), img)


def render_and_save_images(scene, spp, name):
    images = [mi.render(scene, sensor=sensors[i], spp=spp) for i in range(sensor_count)]
    save_images(images, name)
    return images


# %%
# Render and save initial and reference images

# Use mitsuba builtin cube
if hparams["scene"] == "cube":
    scene_dict["shape"].pop("type")
    scene_dict["shape"].pop("filename")
    scene_dict["shape"]["type"] = "cube"

scene_dict["shape"]["to_world"] = T.scale(hparams["scale"])
scene_target = mi.load_dict(scene_dict)

ref_images = render_and_save_images(scene_target, hparams["spp_ref"], "target")


if hparams["scene"] == "cube":
    scene_dict["shape"].pop("to_world")
    scene_dict["shape"]["type"] = "ply"

scene_dict["shape"].pop("to_world", None)
scene_dict["shape"]["to_world"] = T.translate(
    mi.ScalarPoint3f(
        hparams["source_trans"][0],
        hparams["source_trans"][1],
        hparams["source_trans"][2],
    )
).scale(hparams["source_scale"])
scene_dict["shape"]["filename"] = os.path.join(
    DATA_PATH, hparams["scene"], "ico_10k.ply"
)
scene_source = mi.load_dict(scene_dict)

render_and_save_images(scene_source, hparams["spp_ref"], "init")


# %%
def remesh_and_update_params(v_unique, f_unique, remesh_ratio=0.5):
    v_cpu = v_unique.cpu().numpy()
    f_cpu = f_unique.cpu().numpy()
    # Target edge length
    h = (average_edge_length(v_unique, f_unique)).cpu().numpy() * remesh_ratio

    # Run 5 iterations of the Botsch-Kobbelt remeshing algorithm
    v_new, f_new = remesh_botsch(
        v_cpu.astype(np.double), f_cpu.astype(np.int32), 5, h, True
    )

    v = torch.from_numpy(v_new).cuda().float().contiguous()
    f = torch.from_numpy(f_new).cuda().contiguous()

    v_unique, f_unique, duplicate_idx = remove_duplicates(v, f)

    params["shape.vertex_positions"] = mi.Float(v.float().flatten())
    params["shape.faces"] = mi.Int(f.flatten())
    params.update()

    return v, v_unique, f, f_unique, duplicate_idx


losses = {
    "ls": [],
    "ours": [],
}

# We use a torch -> mitsuba pipeline, where mitsuba is only used to render images
# torch is used for the optimisation
for stadam in [True, False]:
    name = "ours" if stadam else "ls"
    scene_source = mi.load_dict(scene_dict)
    params = mi.traverse(scene_source)

    v = params["shape.vertex_positions"].torch().reshape((-1, 3))
    f = mi.Int32(params["shape.faces"]).torch().reshape((-1, 3))

    # Remove duplicates due to e.g. UV seams or face normals.
    # This is necessary to avoid seams opening up during optimisation
    v_unique, f_unique, duplicate_idx = remove_duplicates(v, f)

    # Remesh initial geometry
    remesh_initial_ratio = hparams["remesh_initial_ratio"]
    if remesh_initial_ratio != 1:
        v, v_unique, f, f_unique, duplicate_idx = remesh_and_update_params(
            v_unique, f_unique, remesh_initial_ratio
        )

    # Set up system matrix, optimizer and learning rate scheduler
    lambda_ = hparams["lambda_"]
    alpha = lambda_ / (1.0 + lambda_)
    M = compute_matrix(v_unique, f_unique, alpha=alpha, lambda_=lambda_)

    if stadam:
        v_unique = v_unique.detach().clone()
        opt, lr_scheduler = init_opt_lr_scheduler_ours(v_unique, hparams["lr"], M)
    else:
        M = M / (1 - alpha)
        u_unique = to_differential(M, v_unique.detach().clone())
        opt, lr_scheduler = init_opt_lr_scheduler(u_unique, hparams["lr"])

    iterations = hparams["N_it"]

    remesh_its = list(hparams["remesh_it"])
    try:
        remesh_it = remesh_its.pop(0)
    except:
        remesh_it = -1

    pbar = tqdm(range(iterations * sensor_count))
    for it in range(iterations):
        total_loss = mi.Float(0.0)

        if it == remesh_it:
            # Remesh
            with torch.no_grad():
                if not stadam:
                    v_unique = from_differential(M, u_unique)

                v, v_unique, f, f_unique, duplicate_idx = remesh_and_update_params(
                    v_unique, f_unique
                )

                M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)

                step_size = lr_scheduler.get_last_lr()[-1] * hparams["remesh_lr_beta"]
                if stadam:
                    opt, lr_scheduler = init_opt_lr_scheduler_ours(
                        v_unique, step_size, M
                    )
                else:
                    M = M / (1 - alpha)
                    u_unique = to_differential(M, v_unique)
                    opt, lr_scheduler = init_opt_lr_scheduler(u_unique, step_size)

            try:
                remesh_it = remesh_its.pop(0)
            except:
                remesh_it = -1

        for sensor_idx in range(sensor_count):
            if not stadam:
                # Retrieve the vertex positions from the latent variable
                v_unique = from_differential(M, u_unique)

            # Get the version of the mesh with the duplicates
            v_opt = v_unique[duplicate_idx]

            params["shape.vertex_positions"] = mi.Float(v_opt.flatten())
            dr.enable_grad(params["shape.vertex_positions"])
            params.update()

            img = mi.render(
                scene_source,
                params,
                sensor=sensors[sensor_idx],
                seed=it * sensor_count + sensor_idx,
            )

            loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
            dr.backward(loss)

            # Propagate gradients from loss (mitsuba) -> v_opt (torch) -> v_unique (torch)
            v_opt.backward(
                dr.grad(params["shape.vertex_positions"]).torch().reshape((-1, 3))
            )

            if stadam:
                # Recompute vertex normals
                with torch.no_grad():
                    face_normals = compute_face_normals(v_unique, f_unique)
                    n_unique = compute_vertex_normals(v_unique, f_unique, face_normals)
                opt.step(n_unique)
            else:
                opt.step()
            opt.zero_grad()
            lr_scheduler.step()

            total_loss += loss[0]

            if sensor_idx == 0:
                img = mi.render(
                    scene_source,
                    sensor=sensors[sensor_idx],
                    seed=it,
                    spp=hparams["spp_ref"],
                )
                mi.util.write_bitmap(
                    os.path.join(out_dir, f"{name}_{it}_{sensor_idx}.png"), img
                )

            pbar.set_postfix({"Loss": f"{loss[0]:6f}"})
            pbar.update()

        losses[name].append(total_loss[0] / sensor_count)

        if (it + 1) % hparams["mesh_save_interval"] == 0:
            torch.save(
                {
                    "v": v_unique[duplicate_idx],
                    "f": f,
                },
                os.path.join(out_dir, f"mesh_{name}_{it}.pt"),
            )

    pbar.close()

    v_opt = v_unique[duplicate_idx]
    params["shape.vertex_positions"] = mi.Float(v_opt.flatten())
    params.update()

    render_and_save_images(scene_source, hparams["spp_ref"], f"{name}_final")

    torch.save(
        {
            "v": v_unique[duplicate_idx],
            "f": f,
        },
        os.path.join(out_dir, f"mesh_{name}_final.pt"),
    )

torch.save(
    {
        "losses": losses,
    },
    os.path.join(out_dir, "losses.pt"),
)
# %%
# Plot rendering loss
fig = plt.figure()
plt.plot(losses["ls"], label="Large Steps")
plt.plot(losses["ours"], label="Ours")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(out_dir, "loss.png"))

# %%
# Plot geometry loss
params = mi.traverse(scene_target)
vb = params["shape.vertex_positions"].torch().reshape((-1, 3)).cpu().numpy()
fb = mi.Int32(params["shape.faces"]).torch().reshape((-1, 3)).cpu().numpy()

N_it = hparams["N_it"]
mesh_save_interval = hparams["mesh_save_interval"]

haus_losses = np.zeros((N_it // mesh_save_interval, 2))

with torch.no_grad():
    for i in range(mesh_save_interval - 1, N_it, mesh_save_interval):
        mesh_ls = torch.load(os.path.join(out_dir, f"mesh_ls_{i}.pt"))
        mesh_ours = torch.load(os.path.join(out_dir, f"mesh_ours_{i}.pt"))
        v_ls = mesh_ls["v"].cpu().numpy()
        f_ls = mesh_ls["f"].cpu().numpy()
        v_ours = mesh_ours["v"].cpu().numpy()
        f_ours = mesh_ours["f"].cpu().numpy()

        haus_losses[i // mesh_save_interval, 0] = hausdorff(
            v_ls, f_ls, vb, fb
        ) + hausdorff(vb, fb, v_ls, f_ls)
        haus_losses[i // mesh_save_interval, 1] = hausdorff(
            v_ours, f_ours, vb, fb
        ) + hausdorff(vb, fb, v_ours, f_ours)

x = np.arange(N_it // mesh_save_interval) * mesh_save_interval
fig = plt.figure()
plt.plot(x, haus_losses[:, 0], label="Large Steps")
plt.plot(x, haus_losses[:, 1], label="Ours")
plt.yscale("log")
plt.xlabel("Iteration")
plt.title("Hausdorff Distance")
plt.legend()
plt.savefig(os.path.join(out_dir, "hausdorff.png"))
