// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "torch/extension.h"

#include "nms.h"

static auto registry = torch::jit::RegisterOperators()
  .op("roi_ops::nms", &nms)
  .op("roi_ops::multi_label_nms", &multi_label_nms);
     
#ifndef NO_PYTHON
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("multi_label_nms", &multi_label_nms, "multi label non-maximum suppression");
}
#endif
