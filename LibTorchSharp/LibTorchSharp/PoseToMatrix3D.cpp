#include "THSTensor.h"
#include "torch/fft.h"

#include <torch/csrc/autograd/functions/comm.h>

Tensor THSTensor_pose_to_matrix3d(const Tensor pose, const int64_t size)
{
    torch::Tensor tpose = *pose;
    tpose = tpose.unsqueeze(2);
    tpose.print();

    torch::Tensor dir1 = tpose.slice(1, 0, 3);
    torch::Tensor dir1norm = dir1 / (dir1.norm(2, 1, true) + 1e-10f);

    torch::Tensor dir2 = tpose.slice(1, 3, 6);
    dir2 = dir2 - dir1 * (dir1 * dir2).sum(1, true);
    torch::Tensor dir2norm = dir2 / (dir2.norm(2, 1, true) + 1e-10f);

    torch::Tensor dir3 = dir1.cross(dir2, 1);
    
    torch::Tensor shift = tpose.slice(1, 6, 9) / (size / 2);

    torch::Tensor mat = torch::cat({ dir1, dir2, dir3, shift }, 2);

    CATCH_TENSOR(mat);
}