import numpy as np

__all__ = [
  'TruncExp',
]


class TruncExp(object):
  def __init__(self, mean, min=0, max=np.inf, seed=None):
    self._mean = mean
    self._min = min
    self._max = max
    self._rng = np.random.RandomState(seed)

  def __call__(self):
    if self._min >= self._max:
      return self._max
    else:
      while True:
        v = self._rng.exponential(self._mean)
        if self._min <= v < self._max:
          return v
