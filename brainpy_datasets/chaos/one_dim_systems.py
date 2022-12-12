# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple, Any, List

import brainpy as bp
from brainpy import math as bm
from brainpy.types import Array

from .base import ChaosDataset

__all__ = [
  'LogisticMap'
]


class StandardTransform:
  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None
  ) -> None:
    self.t_transform = t_transform
    self.x_transform = x_transform

  def __call__(self, t: Any, x: Any) -> Tuple[Any, Any]:
    if self.t_transform is not None: t = self.t_transform(t)
    if self.x_transform is not None: x = self.x_transform(x)
    return t, x

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.t_transform is not None:
      body += self._format_transform_repr(self.t_transform, "T transform: ")
    if self.x_transform is not None:
      body += self._format_transform_repr(self.x_transform, "X transform: ")
    return "\n".join(body)


class OneDimChaosData(ChaosDataset):
  ts: Array
  xs: Array

  def __init__(self,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None):
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.transforms = StandardTransform(t_transform, x_transform)

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
