#include "weighted_distance.h"
#include <float.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__host__ __device__ float distance(const int K, const float *coefs, const float *codebook)
{
  float result = 0.;
  for (int i = 0; i < K; i++)
  {
    float diff = coefs[i] - codebook[i];
    result += diff * diff;
  }
  return result;
}

__global__ void distance_reduce(const int N, const int C, const int K,
                                const float *coefs,
                                const float *codebook,
                                float *distances,
                                int64_t *indices)
{

  auto idx = cg::this_grid().thread_rank();
  if (idx >= N)
    return;

  float min_distance = FLT_MAX;
  int64_t min_index;
  for (int64_t i = 0; i < C; i++)
  {
    float d = distance(K, &(coefs[idx * K]), &(codebook[i * K]));
    if (d < min_distance)
    {
      min_distance = d;
      min_index = i;
    }
  }
  distances[idx] = min_distance;
  indices[idx] = min_index;
}

void weighted_distance(const int N, const int C, const int K,
                       const float *coefs,
                       const float *codebook,
                       float *distances,
                       int64_t *indices)
{
  distance_reduce<<<(N + 31) / 32, 32>>>(
      N, C, K,
      coefs,
      codebook,
      distances,
      indices);
}

std::tuple<torch::Tensor, torch::Tensor> weightedDistanceCUDA(
  const torch::Tensor &coefs,
  const torch::Tensor &codebook)
{
  if (coefs.ndimension() != 2 || codebook.ndimension() != 2)
  {
    AT_ERROR("ceofs and codebook must have dimension 2");
  }

  if (codebook.size(1) != coefs.size(1))
  {
    AT_ERROR("coefs and codebook must have same number of channels");
  }

  const int N = coefs.size(0);
  const int C = codebook.size(0);
  const int K = coefs.size(1);

  auto float_opts = coefs.options().dtype(torch::kFloat32);
  auto int_opts = coefs.options().dtype(torch::kInt64);

  torch::Tensor min_distances = torch::zeros({N}, float_opts);
  torch::Tensor min_idx = torch::zeros({N}, int_opts);

  weighted_distance(
      N, C, K,
      coefs.contiguous().data_ptr<float>(),
      codebook.contiguous().data_ptr<float>(),
      // target tensors
      min_distances.contiguous().data_ptr<float>(),
      min_idx.contiguous().data_ptr<int64_t>());

  return std::make_tuple(min_distances, min_idx);
}
