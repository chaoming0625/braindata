# -*- coding: utf-8 -*-

from typing import Generic, TypeVar, Optional

import numpy as np

import brainpy as bp

__all__ = [
  'Dataset',
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Dataset(Generic[T_co]):
  r"""An abstract class representing a :class:`Dataset`.

  All datasets that represent a map from keys to data samples should subclass
  it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
  data sample for a given key. Subclasses could also optionally overwrite
  :meth:`__len__`, which is expected to return the size of the dataset by many
  :class:`~.Sampler` implementations and the default options
  of :class:`~.DataLoader`.

  .. note::
    :class:`~.DataLoader` by default constructs a index
    sampler that yields integral indices.  To make it work with a map-style
    dataset with non-integral indices/keys, a custom sampler must be provided.
  """

  def __init__(self, seed: Optional[int] = None):
    self.seed = bp.check.is_integer(seed, allow_none=True)
    self.rng = np.random.RandomState(seed)

  def __len__(self) -> int:
    raise NotImplementedError

  def __getitem__(self, index) -> T_co:
    raise NotImplementedError

