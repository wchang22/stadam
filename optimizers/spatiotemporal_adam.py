from typing import List, Optional

import os
import math
import slangpy
import torch
from torch import Tensor
from torch.optim.optimizer import (
    _use_grad_for_differentiable,
    _get_value,
    _dispatch_sqrt,
)


class SpatioTemporalAdam(torch.optim.Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        moment_alpha=0.95,
        eps=1e-8,
        weight_decay=0,
        sigma_l=2e-2,
        postfilter=True,
        prefilter=False,
        mtfilter=True,
        vtfilter=True,
        log_primal=False,
        cross_term_off=False,
        amsgrad=False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        kernel_size=11,
        sigma=1.0,
        stride_levels=5,
    ) -> None:
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.moment_alpha = moment_alpha
        self.stride_levels = stride_levels
        self.sigma_l = sigma_l
        self.log_primal = log_primal
        self.cross_term_off = cross_term_off
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.mtfilter = mtfilter
        self.vtfilter = vtfilter
        self.mt_vt_ratio = []


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            first_moments = []
            second_moments = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            self.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

    def adam(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        grad_scale: Optional[Tensor] = None,
        found_inf: Optional[Tensor] = None,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool,
    ):
        r"""Functional API that performs Adam algorithm computation.
        See :class:`~torch.optim.Adam` for details.
        """

        assert grad_scale is None and found_inf is None

        for i, param in enumerate(params):

            grad = grads[i] if not maximize else -grads[i]
            radius = 2

            if self.prefilter:
                filtered_grad = torch.zeros_like(grad)
                strides = [int(2**(j - 1)) for j in range(self.stride_levels)]
                for j in range(len(strides)):
                    filter2d(
                        input_grad=grad,
                        input_primal=grad if self.cross_term_off else param,
                        output=filtered_grad,
                        radius=radius,
                        stride=strides[j],
                        sigma_l=self.sigma_l,
                        log_primal=self.log_primal
                    )
                    grad, filtered_grad = filtered_grad, grad
            grad = torch.nan_to_num(grad)

            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]


            step_t += 1

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            step = _get_value(step_t)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step


            grad = exp_avg.clone()
            var = exp_avg_sq.clone()

            if self.postfilter:
                strides = [int(2**(i - 1)) for i in range(self.stride_levels)]
                if self.mtfilter:
                    filtered_grad = torch.zeros_like(grad)
                    for i in range(len(strides)):
                        filter2d(
                            input_grad=grad,
                            input_primal=grad if self.cross_term_off else param,
                            output=filtered_grad,
                            radius=radius,
                            stride=strides[i],
                            sigma_l=self.sigma_l,
                            log_primal=self.log_primal
                        )
                        grad, filtered_grad = filtered_grad, grad

                    grad = torch.nan_to_num(grad)

                if self.vtfilter:
                    filtered_var = torch.zeros_like(var)
                    for i in range(len(strides)):
                        filter2d(
                            input_grad=var,
                            input_primal=var if self.cross_term_off else param,
                            output=filtered_var,
                            radius=radius,
                            stride=strides[i],
                            sigma_l=self.sigma_l,
                            log_primal=self.log_primal
                        )
                        var, filtered_var = filtered_var, var

                    var = torch.nan_to_num(var)

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (var.sqrt() / bias_correction2_sqrt).add_(eps)
            # self.mt_vt_ratio.append(((grad / var.sqrt()) > 1.0).float().mean().item())
            self.mt_vt_ratio.append((grad.abs() / bias_correction1 / denom).mean().item())
            param.addcdiv_(grad, denom, value=-step_size)


file_path = os.path.join(os.path.dirname(__file__))
m2d_scalar = slangpy.loadModule(os.path.join(file_path, "filter2d_scalar.slang"))
m2d_rgb = slangpy.loadModule(os.path.join(file_path, "filter2d_rgb.slang"))


def filter2d(input_grad, input_primal, output, radius=2, stride=2, sigma_l=2e-2, log_primal=False):
    h, w, c = output.shape
    block_size = 32
    fn = None
    if c == 1:
        fn = m2d_scalar
    elif c == 3:
        fn = m2d_rgb
    fn.filter(
        input_grad=input_grad,
        # input_primal=torch.log(input_primal) if log_primal else input_primal,
        input_primal=torch.log(input_primal.abs() + 1e-8) if log_primal else input_primal,
        output=output,
        radius=radius,
        stride=stride,
        sigma_l=sigma_l
    ).launchRaw(
        blockSize=(block_size, block_size, 1),
        gridSize=(math.ceil(h / block_size), math.ceil(w / block_size), 1),
    )
