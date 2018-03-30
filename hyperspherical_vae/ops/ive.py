# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The exponentially scaled modified Bessel function of the first kind."""

import numpy as np
import scipy.special

from tensorflow.python.ops import script_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import RegisterGradient


def ive(v, z, require_grad=True):
    """Exponentially scaled modified Bessel function of the first kind."""
    if require_grad:
        with get_default_graph().gradient_override_map({"PyFunc": "CustomIveGrad"}):
            output = script_ops.py_func(__ive_py_func, [v, z], z.dtype, name="PyFunc")
    else:
        output = script_ops.py_func(__ive_py_func, [v, z], z.dtype, name="PyFunc")

    output = array_ops.reshape(output, ops.convert_to_tensor(array_ops.shape(z), dtype=dtypes.int32))
    return output


def __ive_py_func(v, z):
    return np.select(condlist=[v == 0, v == 1],
                     choicelist=[scipy.special.i0e(z, dtype=z.dtype),
                                 scipy.special.i1e(z, dtype=z.dtype)],
                     default=scipy.special.ive(v, z, dtype=z.dtype))


@RegisterGradient("CustomIveGrad")
def __ive_grad(op, grad):
    v, z = op.inputs
    new_grad = (ive(v - 1, z, require_grad=False) - ive(v, z, require_grad=False) * (v + z) / z)
    return None, new_grad * grad
