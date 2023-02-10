# -*- coding: utf-8 -*-


from typing import Callable, Optional, Tuple

import jax.numpy as jnp

import brainpy as bp
from brainpy import math as bm
from brainpy import odeint, IntegratorRunner
from brainpy.types import Array
from brainpy_datasets._src.transforms.base import TransformTXY
from .base import ChaosDataset

__all__ = [
  'HenonMap',
  'MackeyGlassEq',
  'PWLDuffuingEq',
]




class TwoDimChaosData(ChaosDataset):
  ts: Array
  xs: Array
  ys: Array

  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None,
      y_transform: Optional[Callable] = None,
  ):
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.y_transform = y_transform
    self.transforms = TransformTXY(t_transform, x_transform, y_transform)

  def __len__(self):
    return self.ts.size

  def __getitem__(self, item: int) -> Tuple[Array, Array, Array]:
    x, y = self.xs[item], self.ys[item]
    t = self.ts
    if self.t_transform is not None:
      t = self.t_transform(t)
    if self.x_transform is not None:
      x = self.x_transform(x)
    if self.y_transform is not None:
      y = self.y_transform(y)
    return t, x, y


class HenonMap(TwoDimChaosData):
  r"""The Hénon map time series.

    The Hénon map is a discrete-time dynamical system. It is one of the
    most studied examples of dynamical systems that exhibit chaotic behavior.

    .. math::

      \begin{split}\begin{cases}x_{n+1} = 1-a x_n^2 + y_n\\y_{n+1} = b x_n.\end{cases}\end{split}

    The map depends on two parameters, a and b, which for the classical
    Hénon map have values of a = 1.4 and b = 0.3. For the classical values
    the Hénon map is chaotic.

    References
    ----------

    .. [1] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/henon_map.html
    .. [1] https://en.wikipedia.org/wiki/H%C3%A9non_map
    """

  def __init__(self,
               num_step, a=1.4, b=0.3, inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform)

    if inits is None:
      inits = {'x': bm.zeros(1), 'y': bm.zeros(1)}
    elif isinstance(inits, dict):
      assert 'x' in inits
      assert 'y' in inits
      inits = {'x': bm.asarray(inits['x']), 'y': bm.asarray(inits['y'])}
      assert inits['x'].shape == inits['y'].shape
    else:
      raise ValueError(f'Please provide dict, and do not support {type(inits)}: {inits}')
    map = _HenonMap(inits['x'].size, a=a, b=b)
    runner = bp.DSRunner(map,
                         monitors=['x', 'y'],
                         dt=1,
                         progress_bar=False,
                         numpy_mon_after_run=numpy_mon)
    runner.run(num_step)

    self.ts = runner.mon['ts']
    self.xs = runner.mon['x']
    self.ys = runner.mon['y']


class MackeyGlassEq(TwoDimChaosData):
  """The Mackey-Glass time series.

  Its dynamics is governed by

  .. math::

     \frac{dP(t)}{dt} = \frac{\beta P(t - \tau)}{1 + P(t - \tau)^n} - \gamma P(t)

  where $\beta = 0.2$, $\gamma = 0.1$, $n = 10$, and the time delay $\tau = 17$. $\tau$
  controls the chaotic behaviour of the equations (the higher it is, the more chaotic
  the timeserie becomes.)

  Parameters
  ----------
  duration: int
  dt: float, int, optional
  beta: float, JaxArray
  gamma: float, JaxArray
  tau: float, JaxArray
  n: float, JaxArray
  inits: optional, float, JaxArray
  method: str
  seed: optional, int
  progress_bar: bool

  Returns
  -------
  result: dict
    The time series data which contain

  References
  ----------

  .. [5] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/mackey_glass_eq.html
  """

  def __init__(self,
               duration, dt=0.1, beta=2., gamma=1., tau=2., n=9.65,
               inits=None, method='rk4', seed=None,
               progress_bar=False, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               ):
    with bm.environment(x64=True):
      super().__init__(t_transform, x_transform, y_transform)

      if inits is None:
        inits = bm.ones(1) * 1.2
      elif isinstance(inits, (float, int)):
        inits = bm.asarray([inits], dtype=bm.float_)
      else:
        assert isinstance(inits, (bm.ndarray, jnp.ndarray))

      rng = bm.random.RandomState(seed)
      xdelay = bm.TimeDelay(inits, tau, dt=dt, interp_method='round')
      xdelay.data.value = inits + 0.2 * (rng.random((xdelay.num_delay_step,) + inits.shape) - 0.5)

      @odeint(method=method, state_delays={'x': xdelay})
      def mg_eq(x, t):
        xtau = xdelay(t - tau)
        return beta * xtau / (1 + xtau ** n) - gamma * x

      runner = IntegratorRunner(
        mg_eq,
        inits={'x': inits},
        monitors={'x(tau)': lambda tdi: xdelay(tdi['t'] - tau), 'x': 'x'},
        progress_bar=progress_bar,
        dt=dt,
        numpy_mon_after_run=numpy_mon
      )
      runner.run(duration)

      self.ts = runner.mon['ts']
      self.xs = runner.mon['x']
      self.ys = runner.mon['x(tau)']



class PWLDuffuingEq(TwoDimChaosData):
  """PWL Duffing chaotic attractor.

  References
  ----------
  .. [12] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#PWL-Duffing-chaotic-attractor
  """

  def __init__(
      self,
      duration, dt=0.001, e=0.25, m0=-0.0845, m1=0.66, omega=1, i=-14,
      method='rk4', inits=None, numpy_mon=False,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None,
      y_transform: Optional[Callable] = None,
  ):
    super().__init__(t_transform, x_transform, y_transform)

    gamma = 0.14 + i / 20

    def PWL_duffing_eq(x, y, t):
      dx = y
      dy = -m1 * x - (0.5 * (m0 - m1)) * (abs(x + 1) - abs(x - 1)) - e * y + gamma * bm.cos(omega * t)
      return dx, dy

    self.integrator = odeint(PWL_duffing_eq, method=method)
    r = _two_variable_model(self.integrator,
                            default_inits=dict(x=0, y=0.),
                            duration=duration, dt=dt, inits=inits,
                            numpy_mon=numpy_mon)
    self.ts = r['ts']
    self.xs = r['xs']
    self.ys = r['ys']


def _two_variable_model(integrator, duration, default_inits, inits=None,
                        args=None, dyn_args=None, dt=0.001, numpy_mon=False):
  if inits is None:
    inits = default_inits
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    inits = {'x': bm.asarray(inits['x']).flatten(),
             'y': bm.asarray(inits['y']).flatten()}
    assert inits['x'].shape == inits['y'].shape
  else:
    raise ValueError

  runner = IntegratorRunner(integrator, monitors=['x', 'y'], inits=inits,
                            args=args, dyn_args=dyn_args, dt=dt, progress_bar=False,
                            numpy_mon_after_run=numpy_mon)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon.x,
          'y': runner.mon.y}


class _HenonMap(bp.DynamicalSystem):
  """Hénon map."""

  def __init__(self, num, a=1.4, b=0.3):
    super(_HenonMap, self).__init__()

    # parameters
    self.a = a
    self.b = b
    self.num = num

    # variables
    self.x = bm.Variable(bm.zeros(num))
    self.y = bm.Variable(bm.zeros(num))

  def update(self, t, dt):
    x_new = 1 - self.a * self.x * self.x + self.y
    self.y.value = self.b * self.x
    self.x.value = x_new
