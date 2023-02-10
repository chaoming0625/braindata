import functools
import math
from typing import Callable, Optional, Union, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

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

TimeDuration = Union[int, float, Callable]


def is_time_duration(x):
  if isinstance(x, (int, float, Callable)):
    return x
  else:
    raise TypeError('A time duration should be an integer/float/function.')


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
    self._max_seq_len = None
    self._return_start_info = False

    # seed
    self.seed = bp.check.is_integer(seed, allow_none=True)
    self.rng = np.random.RandomState(seed)

  def __getitem__(self, index: int) -> Tuple[Data, ...]:
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
    if self._max_seq_len is None:
      x_data, y_data = self.sample_a_trial(index)
      return x_data[0], y_data[0]
    else:
      trial_len = 0
      one_x_data = []
      one_y_data = []
      start_info = []
      while trial_len < self._max_seq_len:
        x_data, y_data = self.sample_a_trial(index)
        trial_len += x_data[0].shape[0]
        one_x_data.append(x_data[0])
        one_y_data.append(y_data[0])
        start = np.zeros(y_data[0].shape[0], dtype=bool)
        start[0] = True
        start_info.append(start)
      X = np.concatenate(one_x_data, axis=0)[:self._max_seq_len]
      Y = np.concatenate(one_y_data, axis=0)[:self._max_seq_len]
      if self._return_start_info:
        S = np.concatenate(start_info, axis=0)[:self._max_seq_len]
        return X, Y, S
      else:
        return X, Y

  def sample_a_trial(self, index) -> Tuple[Tuple[Data, AxesInfo], Tuple[Data, AxesInfo]]:
    raise NotImplementedError

  def __len__(self):
    return self.num_trial

  def __repr__(self) -> str:
    head = "Dataset " + self.__class__.__name__
    body = [f"Number of trials: {self.__len__()}"]
    body += self.extra_repr().splitlines()
    if hasattr(self, "transforms") and self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return "\n".join(lines)

  def extra_repr(self) -> str:
    return ""


def _padding_to_max(max_seq_len, data):
  trial_len = 0
  one_x_data = []
  while trial_len < max_seq_len:
    trial_len += data.shape[0]
    one_x_data.append(data)
  if isinstance(data, (jax.Array, bm.Array)):
    return bm.concatenate(one_x_data, axis=0)[:max_seq_len]
  else:
    return np.concatenate(one_x_data, axis=0)[:max_seq_len]


def collate_fn_repeat(max_seq_len, return_start_info, cls, batch):
  if isinstance(batch[0], np.ndarray):
    if return_start_info:
      starts = [np.zeros((max_seq_len,), dtype=bool) for _ in batch]
      for i, b in enumerate(batch):
        starts[i][0:max_seq_len:len(b)] = True
      data = np.stack([_padding_to_max(max_seq_len, b) for b in batch])
      starts = np.stack(starts)
      if cls.pin_memory:
        data = jnp.asarray(data)
        starts = jnp.asarray(starts)
      return data, starts
    else:
      data = np.stack([_padding_to_max(max_seq_len, b) for b in batch])
      if cls.pin_memory:
        data = jnp.asarray(data)
      return data
  elif isinstance(batch[0], (jax.Array, bm.Array)):
    if return_start_info:
      starts = [bm.zeros((max_seq_len,), dtype=bool) for _ in batch]
      for i, b in enumerate(batch):
        starts[i][0:max_seq_len:len(b)] = True
      data = bm.stack([_padding_to_max(max_seq_len, b) for b in batch])
      starts = bm.stack(starts)
      return data, starts
    else:
      return bm.stack([_padding_to_max(max_seq_len, b) for b in batch])
  elif isinstance(batch[0], (list, tuple)):
    res = tuple(collate_fn_repeat(max_seq_len, return_start_info, cls, var) for var in zip(*batch))
    if return_start_info:
      res = [a[0] for a in res[:-1]] + list(res[-1])
    return res
  else:
    raise ValueError


class TaskLoader(DataLoader):
  def __init__(
      self,
      dataset: CognitiveTask,
      max_seq_len: Optional[int] = None,
      padding: Optional[str] = None,
      return_start_info: bool = False,
      **kwargs
  ):
    assert isinstance(dataset, CognitiveTask)
    super().__init__(dataset, **kwargs)

    self.return_start_info = return_start_info
    self.max_seq_len = max_seq_len
    if max_seq_len is not None:
      assert isinstance(max_seq_len, int)
      padding = 'new_trial' if padding is None else padding
      if padding == 'repeat_trial':
        self._collate_fn = functools.partial(collate_fn_repeat, max_seq_len, return_start_info)
      elif padding == 'new_trial':
        self.dataset._max_seq_len = max_seq_len
        self.dataset._return_start_info = return_start_info
      else:
        raise ValueError(f'Unknown padding: {padding}')
    else:
      assert padding is None, 'No padding support when ``max_seq_len`` is None.'

    # modify the previous batch number
    self._origin_batch_num = (math.floor(len(dataset) / self.batch_size)
                              if self.drop_last else
                              math.ceil(len(dataset) / self.batch_size))
    self._origin_batch_i = 0
    self._prev_data = None
    self._prev_i = 0

  def __next__(self):
    try:
      batch = self.get_batch()
    except ValueError as e:
      raise ValueError('You may provide ``max_seq_len`` if each trial '
                       'has a different length.') from e
    if batch is None:
      raise StopIteration
    else:
      return batch
