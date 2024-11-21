[AutoPyBindCUDA]
[CUDAKernel]
void filter(
    TensorView<float3> input_grad,
    TensorView<float3> input_primal,
    TensorView<float3> output,
    int radius,
    int stride,
    float sigma_d,
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint4 coord = uint4(dispatchIdx.xyz, 0);
    uint3 dims = uint3(input_grad.size(0), input_grad.size(1), input_grad.size(2));

    // If the thread index is beyond the input size, exit early.
    if (coord.x >= dims.x || coord.y >= dims.y || coord.z >= dims.z)
        return;


    double3 filtered_grad = 0.0f;
    double w_sum = 0.0f;
    float eps = 1e-10;

    float kernel[] = {3.0 / 8, 1.0 / 4, 1.0 / 16};

    int r = radius;
    for (int z = -r; z <= r; ++z) {
        for (int y = -r; y <= r; ++y) {
            for (int x = -r; x <= r ; ++x) {
                int4 offset = int4(x, y, z, 0) * stride;
                int4 neighbor_coord = coord + offset;
                if (neighbor_coord.x < 0 || neighbor_coord.x >= dims.x || neighbor_coord.y < 0 || neighbor_coord.y >= dims.y || neighbor_coord.z < 0 || neighbor_coord.z >= dims.z)
                    continue;

                float3 grad = input_grad[coord];
                float3 grad_neighbor = input_grad[neighbor_coord];

                float3 param = input_primal[coord];
                float3 param_neighbor = input_primal[neighbor_coord];

                double w_l = exp(-length(param - param_neighbor) / (sigma_d + eps));
                double h = kernel[abs(x)] * kernel[abs(y)] * kernel[abs(z)];
                double w = h * w_l;

                filtered_grad += w * grad_neighbor;
                w_sum += w;
            }
        }
    }

        filtered_grad /= w_sum;
        output[coord] = float3(filtered_grad);
}
