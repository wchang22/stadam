[AutoPyBindCUDA]
[CUDAKernel]
void filter(
    TensorView<float> input_grad,
    TensorView<float> input_primal,
    TensorView<float> output,
    int radius,
    int stride,
    float sigma_d,
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint2 coord = dispatchIdx.xy;
    uint2 dims = uint2(input_grad.size(0), input_grad.size(1));

    // If the thread index is beyond the input size, exit early.
    if (coord.x >= dims.x || coord.y >= dims.y)
        return;

    double filtered_grad = 0.0f;
    double w_sum = 0.0f;
    double eps = 1e-30;
    double sigma_d_d = double(sigma_d);

    float kernel[] = {3.0 / 8, 1.0 / 4, 1.0 / 16};

    int r = radius;
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r ; ++x) {
            int2 offset = int2(x, y) * stride;
            int2 neighbor_coord = int2(coord) + offset;
            if (neighbor_coord.x < 0 || neighbor_coord.x >= dims.x || neighbor_coord.y < 0 || neighbor_coord.y >= dims.y)
                continue;

            float grad = input_grad[coord];
            float grad_neighbor = input_grad[neighbor_coord];

            double param = double(input_primal[coord]);
            double param_neighbor = double(input_primal[neighbor_coord]);

            double w_l = exp(-length(param - param_neighbor) / (sigma_d_d + eps));
            double h = kernel[abs(x)] * kernel[abs(y)];
            double w = h * w_l;

            filtered_grad += w * double(grad_neighbor);
            w_sum += w;
        }
    }

    filtered_grad /= w_sum;

    output[coord] = float(filtered_grad);
}
