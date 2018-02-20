#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"

// Contains inline wrappers around ATen functions which release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using namespace at;
using at::Generator;

${py_nn_function_dispatch}

}} // namespace torch::autograd
