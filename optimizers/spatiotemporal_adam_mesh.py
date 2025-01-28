import torch
from typing import Union, Literal
from largesteps.parameterize import from_differential
from scipy.stats.qmc import Sobol


def matrix_solve(M, v):
    return from_differential(M, v)


class SpatioTemporalAdamMesh(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        sigma_d=0.5,
        sigma_phi=0.4,
        n_filter_samples=32,
        prefilter: Union[
            Literal["off"], Literal["smooth"], Literal["bilateral"]
        ] = "smooth",
        postfilter: Union[
            Literal["off"], Literal["smooth"], Literal["bilateral"]
        ] = "bilateral",
        postfilter_v: Union[
            Literal["uniform"], Literal["smooth"], Literal["bilateral"]
        ] = "uniform",
    ):
        defaults = dict(lr=lr, betas=betas)
        super(SpatioTemporalAdamMesh, self).__init__(params, defaults)

        self.eps = eps
        self.sigma_d = sigma_d
        self.sigma_phi = sigma_phi
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.postfilter_v = postfilter_v

        soboleng = Sobol(d=2, scramble=True, optimization="lloyd", seed=0)
        eta = torch.tensor(
            soboleng.random(n=n_filter_samples), dtype=torch.float32, device="cuda"
        )
        theta = 2 * torch.pi * eta[:, 0]
        phi = torch.acos(1 - 2 * eta[:, 1])
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        self.normal_samples = torch.stack([x, y, z], dim=1)

    def __setstate__(self, state):
        super(SpatioTemporalAdamMesh, self).__setstate__(state)

    @torch.no_grad()
    def step(self, normals=None):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            M = group.get("M", None)
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1

                if p.grad is None:
                    continue

                grad = p.grad.data
                use_filtering = M is not None and normals is not None

                if self.prefilter != "off" and use_filtering:
                    grad = self.filter(
                        grad, filt_type=self.prefilter, normals=normals, M=M
                    )

                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                if self.postfilter != "off" and use_filtering:
                    g1 = self.filter(
                        g1, filt_type=self.postfilter, normals=normals, M=M
                    )
                    if self.postfilter_v == "uniform":
                        g2 = g2.max()
                    else:
                        g2 = self.filter(
                            g2,
                            filt_type=self.postfilter_v,
                            normals=normals,
                            M=M,
                        )

                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))
                gr = m1 / (self.eps + m2.sqrt())
                p.data.sub_(gr, alpha=lr)

    def filter(self, x, filt_type, normals, M):
        if filt_type == "smooth":
            return matrix_solve(M, x)
        elif filt_type == "bilateral":

            def von_mises_fisher_kernel(n1, n2, sigma, clamp=-1):
                return torch.exp((n1 @ n2.t()).clamp(min=clamp) / sigma)

            g_den = von_mises_fisher_kernel(
                normals, self.normal_samples, self.sigma_d
            ).unsqueeze(1)
            g_num = x.unsqueeze(-1) * g_den
            dim = x.shape[-1]
            g = torch.cat([g_num, g_den], dim=1).reshape(x.shape[0], -1)
            g_diffused = matrix_solve(M, g).reshape(x.shape[0], dim + 1, -1)
            phi = von_mises_fisher_kernel(
                normals, self.normal_samples, self.sigma_phi
            ).unsqueeze(1)
            g_phi_diffused = g_diffused * phi
            x_filtered = g_phi_diffused[:, :dim].sum(dim=-1) / g_phi_diffused[
                :, dim:
            ].sum(dim=-1)
            return x_filtered
