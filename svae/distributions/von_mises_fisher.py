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
"""The von-Mises-Fisher distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops.distributions.beta import Beta

from svae.ops import ive
from svae.distributions.hyperspherical_uniform import HypersphericalUniform

__all__ = [
    "VonMisesFisher",
]


class VonMisesFisher(distribution.Distribution):
    """The von-Mises-Fisher distribution with location `loc` and `scale` parameters.
    #### Mathematical details
    
    The probability density function (pdf) is,
    
    ```none
    pdf(x; mu, k) = exp(k mu^T x) / Z
    Z = (k ** (m / 2 - 1)) / ((2pi ** m / 2) * besseli(m / 2 - 1, k))
    ```
    where `loc = mu` is the mean, `scale = k` is the concentration, `m` is the dimensionality, and, `Z`
    is the normalization constant.
    
    See https://en.wikipedia.org/wiki/Von_Mises-Fisher distribution for more details on the 
    Von Mises-Fiser distribution.
    
    """

    def __init__(self, loc, scale, validate_args=False, allow_nan_stats=True, name="von-Mises-Fisher"):
        """Construct von-Mises-Fisher distributions with mean and concentration `loc` and `scale`.

        Args:
          loc: Floating point tensor; the mean of the distribution(s).
          scale: Floating point tensor; the concentration of the distribution(s).
            Must contain only non-negative values.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          TypeError: if `loc` and `scale` have different `dtype`.
        """
        parameters = locals()
        with ops.name_scope(name, values=[loc, scale]):
            with ops.control_dependencies([check_ops.assert_positive(scale),
                                           check_ops.assert_near(linalg_ops.norm(loc, axis=-1), 1, atol=1e-7)]
                                          if validate_args else []):
                self._loc = array_ops.identity(loc, name="loc")
                self._scale = array_ops.identity(scale, name="scale")
                check_ops.assert_same_float_dtype([self._loc, self._scale])

        super(VonMisesFisher, self).__init__(
            dtype=self._scale.dtype,
            reparameterization_type=distribution.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._loc, self._scale],
            name=name)

        self.__m = math_ops.cast(self._loc.shape[-1], dtypes.int32)
        self.__mf = math_ops.cast(self.__m, dtype=self.dtype)
        self.__e1 = array_ops.one_hot([0], self.__m, dtype=self.dtype)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(zip(("loc", "scale"), ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32),
                                            ops.convert_to_tensor(sample_shape[:-1].concatenate([1]),
                                                                  dtype=dtypes.int32)])))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for concentration."""
        return self._scale

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self._loc),
            array_ops.shape(self._scale))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
            self._loc.get_shape(),
            self._scale.get_shape())

    def _event_shape_tensor(self):
        return constant_op.constant([], dtype=dtypes.int32)

    def _event_shape(self):
        return tensor_shape.scalar()

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
        w = control_flow_ops.cond(gen_math_ops.equal(self.__m, 3),
                                  lambda: self.__sample_w3(n, seed),
                                  lambda: self.__sample_w_rej(n, seed))

        v = nn_impl.l2_normalize(array_ops.transpose(
            array_ops.transpose(random_ops.random_normal(shape, dtype=self.dtype, seed=seed))[1:]), axis=-1)

        x = array_ops.concat((w, math_ops.sqrt(1 - w ** 2) * v), axis=-1)
        z = self.__householder_rotation(x)

        return z

    def __sample_w3(self, n, seed):
        # This is not working!!  --> NaN
        shape = array_ops.concat(([n], self.batch_shape_tensor()[:-1], [1]), 0)
        u = random_ops.random_uniform(shape, dtype=self.dtype, seed=seed)
        self.__w = 1 + math_ops.log(u + (1 - u) * math_ops.exp(-2 * self.scale)) / self.scale
        return self.__w

    def __sample_w_rej(self, n, seed):
        c = math_ops.sqrt((4 * (self.scale ** 2)) + (self.__mf - 1) ** 2)
        b = (-2 * self.scale + c) / (self.__mf - 1)
        a = (self.__mf - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__mf - 1) * math_ops.log(self.__mf - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, n, seed)
        return self.__w

    def __while_loop(self, b, a, d, n, seed):
        def __cond(w, e, bool_mask, b, a, d):
            return math_ops.reduce_any(bool_mask)

        def __body(w_, e_, bool_mask, b, a, d):
            e = math_ops.cast(Beta((self.__mf - 1) / 2, (self.__mf - 1) / 2).sample(
                shape, seed=seed), dtype=self.dtype)

            u = random_ops.random_uniform(shape, dtype=self.dtype, seed=seed)

            w = (1 - (1 + b) * e) / (1 - (1 - b) * e)
            t = (2 * a * b) / (1 - (1 - b) * e)

            accept = gen_math_ops.greater(((self.__mf - 1) * math_ops.log(t) - t + d), math_ops.log(u))
            reject = gen_math_ops.logical_not(accept)

            w_ = array_ops.where(gen_math_ops.logical_and(bool_mask, accept), w, w_)
            e_ = array_ops.where(gen_math_ops.logical_and(bool_mask, accept), e, e_)
            bool_mask = array_ops.where(gen_math_ops.logical_and(bool_mask, accept), reject, bool_mask)

            return w_, e_, bool_mask, b, a, d

        shape = array_ops.concat([[n], self.batch_shape_tensor()[:-1], [1]], 0)
        b, a, d = [gen_array_ops.tile(array_ops.expand_dims(e, axis=0), [n] + [1] * len(e.shape)) for e in (b, a, d)]

        w, e, bool_mask, b, a, d = control_flow_ops.while_loop(__cond, __body,
                                                               [array_ops.zeros_like(b, dtype=self.dtype),
                                                                array_ops.zeros_like(b, dtype=self.dtype),
                                                                array_ops.ones_like(b, dtypes.bool),
                                                                b, a, d])

        return e, w

    def __householder_rotation(self, x):
        u = nn_impl.l2_normalize(self.__e1 - self._loc, axis=-1)
        z = x - 2 * math_ops.reduce_sum(x * u, axis=-1, keepdims=True) * u
        return z

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))

    def _log_unnormalized_prob(self, x):
        with ops.control_dependencies(
                [check_ops.assert_near(linalg_ops.norm(x, axis=-1), 1, atol=1e-3)] if self.validate_args else []):
            output = self.scale * math_ops.reduce_sum(self._loc * x, axis=-1, keepdims=True)

        return array_ops.reshape(output, ops.convert_to_tensor(array_ops.shape(output)[:-1]))

    def _log_normalization(self):
        output = -((self.__mf / 2 - 1) * math_ops.log(self.scale) - (self.__mf / 2) * math.log(2 * math.pi) - (
                    self.scale + math_ops.log(ive(self.__mf / 2 - 1, self.scale))))

        return array_ops.reshape(output, ops.convert_to_tensor(array_ops.shape(output)[:-1]))

    def _entropy(self):
        return - array_ops.reshape(self.scale * ive(self.__mf / 2, self.scale) / ive((self.__mf / 2) - 1, self.scale),
                                   ops.convert_to_tensor(array_ops.shape(self.scale)[:-1])) + self._log_normalization()

    def _mean(self):
        return self._loc * (ive(self.__mf / 2, self.scale) / ive(self.__mf / 2 - 1, self.scale))

    def _mode(self):
        return self._mean()


@kullback_leibler.RegisterKL(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu, name=None):
    with ops.control_dependencies([check_ops.assert_equal(vmf.loc.shape[-1] - 1, hyu.dim)]):
        with ops.name_scope(name, "_kl_vmf_uniform", [vmf.scale]):
            return - vmf.entropy() + hyu.entropy()
