from typing import Sequence, Union, Tuple, Any

import brainpy as bp
import numpy as np


def _slice(target):
  if target is None:
    loc = slice(None, None, None)
  elif isinstance(target, int):
    loc = target
  elif isinstance(target, tuple):
    assert len(target) == 2
    loc = slice(target[0], target[1], None)
  else:
    raise ValueError
  return loc


def add(target: np.ndarray,
        value: Any,
        *slices):
  slices = [slice(None, None, None) if s is None else s for s in slices]
  target[slices] += value


def set(target: np.ndarray,
        value: Any,
        *slices):
  slices = [slice(None, None, None) if s is None else s for s in slices]
  target[slices] = value


def interval_of(elem: str, total: Union[dict[str, int], Sequence[Tuple[str, int]]]) -> slice:
  if isinstance(total, dict):
    total = tuple(total.items())
  bp.check.is_sequence(total, elem_type=tuple)
  s = 0
  for k, v in total:
    if k == elem:
      return slice(s, s + v, None)
    else:
      s += v
  else:
    raise ValueError('Not found')


def period_to_arr(periods: dict):
  res = [np.ones(length, dtype=int) * i for i, length in enumerate(periods.values())]
  return np.concatenate(res)


def firing_rate(base, dt, mode):
  if mode == 'rate':
    return base * dt / 1e3
  elif mode == 'spiking':
    if base * dt > 1e3:
      raise ValueError(f'dt is too big, so that dt * fr > 1e3.')
    return base * dt / 1e3
  else:
    raise ValueError('Please set mode as "rate" or "spiking"')

