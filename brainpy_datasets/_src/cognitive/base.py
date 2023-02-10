from typing import Callable, Optional, Any, Union, Sequence, Tuple, Dict

import math
import jax
import numpy as np
import jax.numpy as jnp

import brainpy as bp
import brainpy.math as bm
from brainpy_datasets._src.base import Dataset
from brainpy_datasets._src.dataloader import DataLoader
from brainpy_datasets._src.transforms.base import TransformIT


Data = Union[np.ndarray, bm.Array, jax.Array]
# For example:
#   ax0: [('fixation', 10), ('stimulus', 20)]
#   ax1: [('fixation', 1), ('choice', 2)]
AxesInfo = Dict[str, Sequence[Tuple[str, int]]]


class CognitiveTask(Dataset):
  """
  Base class for making a cognitive task.
  It is necessary to override the ``__getitem__`` and ``__len__`` method.

  .. note::

      :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.

  Parameters
  ----------
  input_transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
  target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.

  """

  _repr_indent = 4

  output_features: Sequence[str]
  input_features: Sequence[str]

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      num_trial: int = 1024,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ) -> None:
    self.input_transform = input_transform
    self.target_transform = target_transform
    self.transforms = TransformIT(input_transform, target_transform)
    self.dt = bp.check.is_float(dt, allow_int=True)
    self.num_trial = bp.check.is_integer(num_trial, )

    # seed
    self.seed = bp.check.is_integer(seed, allow_none=True)
    self.rng = np.random.RandomState(seed)

  def __getitem__(self, index: int) -> Tuple[Data, Data]:
    """Get one data.

    Parameters
    ----------
    index: int
      Index.

    Returns
    -------
    out: Any
      Sample and meta data, optionally transformed by the respective transforms.
    """
    x_data, y_data = self.sample_trial(index)
    return x_data[0], y_data[0]

  def sample_trial(self, index) -> Tuple[Tuple[Data, AxesInfo], Tuple[Data, AxesInfo]]:
    raise NotImplementedError

  def __len__(self):
    return self.num_trial

  def __repr__(self) -> str:
    head = "Dataset " + self.__class__.__name__
    body = [f"Number of datapoints: {self.__len__()}"]
    body += self.extra_repr().splitlines()
    if hasattr(self, "transforms") and self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return "\n".join(lines)

  def extra_repr(self) -> str:
    return ""


class FixedLenCogTask(CognitiveTask):
  pass


class VariedLenCogTask(CognitiveTask):
  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 100.,
      num_trial: int = 1024,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ) -> None:
    super().__init__(dt=dt,
                     num_trial=num_trial,
                     seed=seed,
                     input_transform=input_transform,
                     target_transform=target_transform)

    self.max_seq_len = max_seq_len

  def __getitem__(self, index: int) -> Tuple[Data, Data]:
    trial_len = 0
    one_x_data = []
    one_y_data = []
    while trial_len < self.max_seq_len:
      x_data, y_data = self.sample_trial(index)
      trial_len += x_data[0].shape[0]
      one_x_data.append(x_data[0])
      one_y_data.append(y_data[0])
    return np.concatenate(one_x_data, axis=0)[:self.max_seq_len], np.concatenate(one_y_data, axis=0)[:self.max_seq_len]

  __getitem__.__doc__ = CognitiveTask.__getitem__.__doc__


class TaskLoader(DataLoader):
  def __init__(
      self,
      dataset: CognitiveTask,
      max_seq_len: Optional[int] = None,
      **kwargs
  ):
    super().__init__(dataset, **kwargs)

    self.max_seq_len = max_seq_len
    if max_seq_len is None:
      if isinstance(dataset, VariedLenCogTask):
        raise ValueError(f'For instance of {VariedLenCogTask.__name__}, '
                         f'you should provide ``max_seq_len``. ')

    # modify the previous batch number
    self._origin_batch_num = (math.floor(len(dataset) / self.batch_size)
                              if self.drop_last else
                              math.ceil(len(dataset) / self.batch_size))
    self._origin_batch_i = 0
    self._prev_data = None
    self._prev_i = 0

  def get_batch(self):
    if self.max_seq_len is None:
      return super().get_batch()
    else:
      if self._origin_batch_i >= self._origin_batch_num:
        return None

      if self._index >= len(self.dataset):
        self._index = 0
      batch_size = len(self.dataset) - self._index
      if self.drop_last and batch_size < self.batch_size:
        self._index = 0
      batch = self._collate_fn([self._get() for _ in range(self.batch_size)])

      if self._prev_data is None:
        self._prev_data = [np.zeros((b.shape[0], self.max_seq_len) + b.shape[2:])
                           for b in batch]
      seq_len = batch[0].shape[1]
      diff_seq_len = self.max_seq_len - self._prev_i
      if diff_seq_len >= seq_len:
        for i, b in enumerate(batch):
          self._prev_data[i][:, self._prev_i: self._prev_i + seq_len] = b
        self._prev_i += seq_len
      else:
        for i, b in enumerate(batch):
          self._prev_data[i][:, self._prev_i:] = b[:, :diff_seq_len]
        self._prev_i += diff_seq_len

      if self._prev_i >= self.max_seq_len:
        data = self._prev_data
        self._prev_data = None
        self._prev_i = 0
        self._origin_batch_i += 1
        return data
