#include <torch/extension.h>
#include "weighted_distance.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weightedDistance", &weightedDistanceCUDA);
}
