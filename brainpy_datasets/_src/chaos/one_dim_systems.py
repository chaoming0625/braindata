# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple

import brainpy as bp
from brainpy import math as bm
from brainpy.types import Array
from brainpy_datasets._src.transforms.base import TransformTX
from .base import ChaosDataset

__all__ = [
  'LogisticMap'
]


class OneDimChaosData(ChaosDataset):
  ts: Array
  xs: Array

  def __init__(self,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None):
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.transforms = TransformTX(t_transform, x_transform)

  def __len__(self):
    return self.ts.size

  def __getitem__(self, item: int) -> Tuple[Array, Array]:
    x = self.xs[item]
    t = self.ts
    if self.t_transform is not None:
      t = self.t_transform(t)
    if self.x_transform is not None:
      x = self.x_transform(x)
    return t, x


class LogisticMap(OneDimChaosData):
  r"""The logistic map time series.

  The logistic map is defined by the following equation:

  .. math::

     x_{n+1}=\lambda x_{n}\left(1-x_{n}\right) \quad \text { with } \quad n=0,1,2,3 \ldots

  References
  ----------
  .. [3] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/logistic_map.html
  .. [4] https://en.wikipedia.org/wiki/Logistic_map

  """

  def __init__(self, num_step, mu=3., inits=None, numpy_mon=False,
               transform: Optional[Callable] = None,
               target_transform: Optional[Callable] = None,
               ):
    super().__init__(transform, target_transform)

    if inits is None:
      inits = bm.ones(1) * 0.2
    else:
      inits = bm.asarray(inits)
    runner = bp.DSRunner(_LogisticMap(inits.size, mu=mu),
                         monitors=['x'], dt=1, progress_bar=False,
                         numpy_mon_after_run=numpy_mon)
    runner.run(num_step)
    self.ts = runner.mon['ts']
    self.xs = runner.mon['xs']


class _LogisticMap(bp.DynamicalSystem):
  def __init__(self, num, mu=3.):
    super(_LogisticMap, self).__init__()

    self.mu = mu
    self.x = bm.Variable(bm.ones(num) * 0.2)

  def update(self, t, dt):
    self.x.value = self.mu * self.x * (1 - self.x)
