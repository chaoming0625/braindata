# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple

import brainpy as bp
from brainpy import math as bm
from brainpy import odeint, JointEq, IntegratorRunner
from brainpy.types import Array
from brainpy_datasets._src.transforms.base import TransformTXYZ
from .base import ChaosDataset

__all__ = [
  'ModifiedLuChenEq',
  'LorenzEq',
  'RabinovichFabrikantEq',
  'ChenChaoticEq',
  'LuChenEq',
  'ChuaChaoticEq',
  'ModifiedChuaEq',
  'ModifiedLorenzEq',
  'DoubleScrollEq',
]


class ThreeDimChaosData(ChaosDataset):
  ts: Array
  xs: Array
  ys: Array
  zs: Array

  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None,
      y_transform: Optional[Callable] = None,
      z_transform: Optional[Callable] = None,
  ):
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.y_transform = y_transform
    self.z_transform = z_transform
    self.transforms = TransformTXYZ(t_transform, x_transform, y_transform, z_transform)

  def __len__(self):
    return self.ts.size

  def __getitem__(self, item: int) -> Tuple[Array, Array, Array, Array]:
    x, y, z = self.xs[item], self.ys[item], self.zs[item]
    t = self.ts
    if self.t_transform is not None:
      t = self.t_transform(t)
    if self.x_transform is not None:
      x = self.x_transform(x)
    if self.y_transform is not None:
      y = self.y_transform(y)
    if self.z_transform is not None:
      z = self.z_transform(z)
    return t, x, y, z


class ModifiedLuChenEq(ThreeDimChaosData):
  """Modified Lu Chen attractor.

  References
  ----------
  .. [4] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Modified-Lu-Chen-attractor
  """

  def __init__(self,
               duration, dt=0.001, a=36, c=20, b=3, d1=1, d2=0., tau=.2,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    if inits is None:
      inits = {'x': bm.ones(1), 'y': bm.ones(1), 'z': bm.ones(1) * 14}
    elif isinstance(inits, dict):
      assert 'x' in inits
      assert 'y' in inits
      assert 'z' in inits
      inits = {'x': bm.asarray(inits['x']),
               'y': bm.asarray(inits['y']),
               'z': bm.asarray(inits['z'])}
      assert inits['x'].shape == inits['y'].shape == inits['z'].shape
    else:
      raise ValueError
    eq = _ModifiedLuChenSystem(num=inits['x'].size, a=a, b=b, c=c, d1=d1, d2=d2, tau=tau, dt=dt, method=method)
    eq.x[:] = inits['x']
    eq.y[:] = inits['y']
    eq.z[:] = inits['z']
    runner = bp.DSRunner(eq,
                         monitors=['x', 'y', 'z'],
                         dt=dt,
                         progress_bar=False,
                         numpy_mon_after_run=numpy_mon)
    runner.run(duration)

    self.ts = runner.mon['ts']
    self.xs = runner.mon['x']
    self.ys = runner.mon['y']
    self.zs = runner.mon['z']


class LorenzEq(ThreeDimChaosData):
  """The Lorenz system.

  The Lorenz system is a system of ordinary differential equations first
  studied by mathematician and meteorologist Edward Lorenz.


  Returns
  -------
  data: dict
    A dict data with the keys of ``ts``, ``x``, ``y``, and ``z``,
    where ``ts`` is the history time value, ``x, y, z`` are history
    values of the variable in the Lorenz system.

  References
  ----------
  .. [6] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/lorenz_system.html

  """

  def __init__(self,
               duration, dt=0.001, sigma=10, beta=8 / 3, rho=28,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    dx = lambda x, t, y: sigma * (y - x)
    dy = lambda y, t, x, z: x * (rho - z) - y
    dz = lambda z, t, x, y: x * y - beta * z
    integral = odeint(JointEq([dx, dy, dz]), method=method)

    res = _three_variable_model(integral,
                                default_inits={'x': 8, 'y': 1, 'z': 1},
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class RabinovichFabrikantEq(ThreeDimChaosData):
  """Rabinovich-Fabrikant equations.

  The Rabinovich–Fabrikant equations are a set of three coupled ordinary
  differential equations exhibiting chaotic behaviour for certain values
  of the parameters. They are named after Mikhail Rabinovich and Anatoly
  Fabrikant, who described them in 1979.

  References
  ----------
  .. [7] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Rabinovich_Fabrikant_eq.html

  """

  def __init__(self,
               duration, dt=0.001, alpha=1.1, gamma=0.803,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def rf_eqs(x, y, z, t):
      dx = y * (z - 1 + x * x) + gamma * x
      dy = x * (3 * z + 1 - x * x) + gamma * y
      dz = -2 * z * (alpha + x * y)
      return dx, dy, dz

    res = _three_variable_model(rf_eqs,
                                default_inits={'x': -1, 'y': 0, 'z': 0.5},
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class ChenChaoticEq(ThreeDimChaosData):
  """Chen attractor.

  References
  ----------
  .. [7] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Chen-attractor
  """

  def __init__(self,
               duration, dt=0.001, a=40, b=3, c=28,
               method='euler', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def chen_system(x, y, z, t):
      dx = a * (y - x)
      dy = (c - a) * x - x * z + c * y
      dz = x * y - b * z
      return dx, dy, dz

    res = _three_variable_model(chen_system,
                                default_inits=dict(x=-0.1, y=0.5, z=-0.6),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class LuChenEq(ThreeDimChaosData):
  """Lu Chen attractor.

  References
  ----------
  .. [8] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Lu-Chen-attractor
  """

  def __init__(self,
               duration, dt=0.001, a=36, c=20, b=3, u=-15.15,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def lu_chen_system(x, y, z, t):
      dx = a * (y - x)
      dy = x - x * z + c * y + u
      dz = x * y - b * z
      return dx, dy, dz

    res = _three_variable_model(lu_chen_system,
                                default_inits=dict(x=0.1, y=0.3, z=-0.6),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class ChuaChaoticEq(ThreeDimChaosData):
  """Chua’s system.

  References
  ----------
  .. [9] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Chua%E2%80%99s-system
  """

  def __init__(self,
               duration, dt=0.001, alpha=10, beta=14.514,
               gamma=0, a=-1.197, b=-0.646, method='rk4',
               inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def chua_equation(x, y, z, t):
      fx = b * x + 0.5 * (a - b) * (bm.abs(x + 1) - bm.abs(x - 1))
      dx = alpha * (y - x) - alpha * fx
      dy = x - y + z
      dz = -beta * y - gamma * z
      return dx, dy, dz

    res = _three_variable_model(chua_equation,
                                default_inits=dict(x=0.001, y=0, z=0.),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class ModifiedChuaEq(ThreeDimChaosData):
  """Modified Chua chaotic attractor.

  References
  ----------
  .. [10] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Modified-Chua-chaotic-attractor

  """

  def __init__(self,
               duration, dt=0.001, alpha=10.82, beta=14.286, a=1.3, b=.11, d=0,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def modified_chua_system(x, y, z, t):
      dx = alpha * (y + b * bm.sin(bm.pi * x / 2 / a + d))
      dy = x - y + z
      dz = -beta * y
      return dx, dy, dz

    res = _three_variable_model(modified_chua_system,
                                default_inits=dict(x=1, y=1, z=0.),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class ModifiedLorenzEq(ThreeDimChaosData):
  """Modified Lorenz chaotic system.

  References
  ----------
  .. [11] https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/Multiscroll_attractor.html#Modified-Lorenz-chaotic-system
  """

  def __init__(self,
               duration, dt=0.001, a=10, b=8 / 3, c=137 / 5,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def modified_Lorenz(x, y, z, t):
      temp = 3 * bm.sqrt(x * x + y * y)
      dx = (-(a + 1) * x + a - c + z * y) / 3 + ((1 - a) * (x * x - y * y) + (2 * (a + c - z)) * x * y) / temp
      dy = ((c - a - z) * x - (a + 1) * y) / 3 + (2 * (a - 1) * x * y + (a + c - z) * (x * x - y * y)) / temp
      dz = (3 * x * x * y - y * y * y) / 2 - b * z
      return dx, dy, dz

    res = _three_variable_model(modified_Lorenz,
                                default_inits=dict(x=-8, y=4, z=10),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


class DoubleScrollEq(ThreeDimChaosData):
  r"""Double-scroll electronic circuit attractor.

  Its behavior is governed by

  .. math::

     {\dot{V}}_{1} ={V}_{1}/{R}_{1}-\varDelta V/{R}_{2}\,-\,2{I}_{r}\,\sinh (\beta \varDelta V),\\
     \dot{{V}_{2}} =\varDelta V/{R}_{2}+2{I}_{r}\,\sinh (\beta \varDelta V)-I,\\
     \dot{I} ={V}_{2}-{R}_{4}I

  in dimensionless form.

  References
  ----------
  .. [1] Chang, A., Bienfang, J. C., Hall, G. M., Gardner, J. R. &
         Gauthier, D. J. Stabilizing unstable steady states using
         extended time-delay autosynchronization. Chaos 8, 782–790 (1998).
  """

  def __init__(self,
               duration, dt=0.01,
               R1=1.2, R2=3.44, R4=0.193, beta=11.6, Ir=2 * 2.25e-5,
               method='rk4', inits=None, numpy_mon=False,
               t_transform: Optional[Callable] = None,
               x_transform: Optional[Callable] = None,
               y_transform: Optional[Callable] = None,
               z_transform: Optional[Callable] = None,
               ):
    super().__init__(t_transform, x_transform, y_transform, z_transform)

    @odeint(method=method)
    def double_scroll(x, y, z, t):
      delta = x - y
      dV1 = x / R1 - delta / R2 - 2 * Ir * bm.sinh(beta * delta)
      dV2 = delta / R2 + 2 * Ir * bm.sinh(beta * delta) - z
      dI = y - R4 * z
      return dV1, dV2, dI

    res = _three_variable_model(double_scroll,
                                default_inits=dict(x=0.37926545, y=0.058339, z=-0.08167691),
                                duration=duration, dt=dt, inits=inits,
                                numpy_mon=numpy_mon)
    self.ts = res['ts']
    self.xs = res['x']
    self.ys = res['y']
    self.zs = res['z']


def _three_variable_model(integrator, duration, default_inits, inits=None, args=None,
                          dyn_args=None, dt=0.001, numpy_mon=False):
  if inits is None:
    inits = default_inits  # {'x': -1, 'y': 0, 'z': 0.5}
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    assert 'z' in inits
    inits = {'x': bm.asarray(inits['x']).flatten(),
             'y': bm.asarray(inits['y']).flatten(),
             'z': bm.asarray(inits['z']).flatten()}
    assert inits['x'].shape == inits['y'].shape == inits['z'].shape
  else:
    raise ValueError

  runner = IntegratorRunner(integrator, monitors=['x', 'y', 'z'], inits=inits,
                            args=args, dyn_args=dyn_args, dt=dt, progress_bar=False,
                            numpy_mon_after_run=numpy_mon)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon.x,
          'y': runner.mon.y,
          'z': runner.mon.z}


class _ModifiedLuChenSystem(bp.DynamicalSystem):
  def __init__(self, num, a=35, b=3, c=28, d0=1, d1=1, d2=0., tau=.2, dt=0.1, method='rk4'):
    super(_ModifiedLuChenSystem, self).__init__()

    # parameters
    self.a = a
    self.b = b
    self.c = c
    self.d0 = d0
    self.d1 = d1
    self.d2 = d2
    self.tau = tau
    self.delay_len = int(tau / dt)

    # variables
    self.z = bm.Variable(bm.ones(num) * 14.)
    self.z_delay = bm.LengthDelay(self.z, delay_len=self.delay_len, initial_delay_data=14.)
    self.x = bm.Variable(bm.ones(num))
    self.y = bm.Variable(bm.ones(num))

    # functions
    def derivative(x, y, z, t):
      dx = self.a * (y - x)
      z_delay = self.z_delay.retrieve(self.delay_len)
      f = self.d0 * z + self.d1 * z_delay - self.d2 * bm.sin(z_delay)
      dy = (self.c - self.a) * x - x * f + self.c * y
      dz = x * y - self.b * z
      return dx, dy, dz

    self.integral = odeint(derivative, method=method)

  def update(self, t, dt):
    self.x.value, self.y.value, self.z.value = self.integral(
      self.x, self.y, self.z, t, dt)
    self.z_delay.update(self.z)
