import functools
import math
from typing import Callable, Optional, Union, Sequence, Tuple, Dict

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from brainpy_datasets._src.base import Dataset
from brainpy_datasets._src.dataloader import DataLoader
from brainpy_datasets._src.transforms.base import TransformIT


Data = Union[np.ndarray, bm.Array, jax.Array]
# For example:
#   ax0: [('fixation', 10), ('stimulus', 20)]
#   ax1: [('fixation', 1), ('choice', 2)]
AxesInfo = Dict[str, Sequence[Tuple[str, int]]]

TimeDuration = Union[int, float, Callable]


class _FeatureBase(object):
  def __init__(self, r_num, s_num):
    self.start = None
    self.end = None
    self._mode = None
    self._r_num = r_num
    self._s_num = s_num
    self._r_start = 0
    self._r_end = self._r_num
    self._s_start = 0
    self._s_end = self._s_num

  @property
  def mode(self) -> str:
    return self._mode

  @mode.setter
  def mode(self, mode: str):
    assert mode in ['rate', 'spiking']
    self._mode = mode

  @property
  def is_spiking_mode(self):
    return self.mode == 'spiking'

  @property
  def is_rate_mode(self):
    return self.mode == 'rate'

  @property
  def num(self):
    if self.mode == 'rate':
      return self._r_num
    elif self.mode == 'spiking':
      return self._s_num
    else:
      raise ValueError('Please set mode as "rate" or "spiking"')

  @property
  def num_rate(self):
    return self._r_num

  @property
  def num_spike(self):
    return self._s_num

  @property
  def i(self) -> slice:
    """Index of encoding values."""
    if self.mode == 'rate':
      return slice(self._r_start, self._r_end)
    elif self.mode == 'spiking':
      return slice(self._s_start, self._s_end)
    else:
      raise ValueError('Please set mode as "rate" or "spiking"')

  @property
  def i_rate(self):
    return slice(self._r_start, self._r_end)

  @property
  def i_spike(self):
    return slice(self._s_start, self._s_end)

  def __add__(self, other: '_FeatureBase'):
    raise NotImplementedError

  def shift(self, rate_num, spike_num):
    raise NotImplementedError

  def __repr__(self):
    return self.__class__.__name__

  def set_mode(self, mode: str):
    raise NotImplementedError


class _FeatSet(_FeatureBase):
  def __init__(self, *fts):
    self.fts = fts
    self._s = dict()
    for ft in fts:
      if ft.name is None:
        name = bp.tools.get_unique_name(ft.__class__.__name__)
      else:
        name = ft.name
      if name in self._s:
        raise ValueError(f'Duplicate feature name {name}')
      else:
        self._s[name] = ft
    r_num = sum([ft.num_rate for ft in fts])
    s_num = sum([ft.num_spike for ft in fts])
    super().__init__(r_num, s_num)

  def set_mode(self, mode: str):
    self.mode = mode
    for ft in self.fts:
      ft.mode = mode

  def shift(self, r_num, s_num):
    self._r_start += r_num
    self._r_end += r_num
    self._s_start += s_num
    self._s_end += s_num
    for ft in self.fts:
      ft._r_start += self._r_num
      ft._r_end += self._r_num
      ft._s_start += self._s_num
      ft._s_end += self._s_num

  def __getitem__(self, item: str) -> Union['Feature', Tuple['Feature', ...]]:
    if isinstance(item, str):
      return self._s[item]
    elif isinstance(item, (int, slice)):
      return tuple(self._s.values())[item]
    else:
      raise ValueError

  def __add__(self, other: Union['Feature', '_FeatSet']):
    if isinstance(other, Feature):
      other.shift(self._r_num, self._s_num)
      return _FeatSet(*self.fts, other)
    elif isinstance(other, _FeatSet):
      other.shift(self._r_num, self._s_num)
      return _FeatSet(*self.fts, *other.fts)
    else:
      raise ValueError(f'Only support addition with {Feature.__name__} or {_FeatSet.__name__}, '
                       f'but got {type(other)}')

  def __repr__(self):
    names = [ft.name for ft in self.fts]
    rates = [ft.num_rate for ft in self.fts]
    spikes = [ft.num_spike for ft in self.fts]
    return (f'{self.__class__.__name__}(names={names}, '
            f'rates={rates}, spikes={spikes})')


class Feature(_FeatureBase):
  """Object to indicate how to encode features.

  Args:
    name (str): The name of the feature.
    n_rate (int): The number of elements to encode the rate values.
    n_spike (int): The number of elements to encode the spike values.
    fr (int, float): The firing rate.

  """

  def __init__(
      self,
      n_rate: int = 1,
      n_spike: int = 20,
      fr: Union[int, float, Callable] = 30.,
      name: str = None,
  ):
    self.name = name
    self.firing_rate = fr
    super().__init__(n_rate, n_spike)

  def fr(self, dt: Union[int, float]):
    """Get firing rate.

    Args:
      dt (int, float): The time step.
    """
    if callable(self.firing_rate):
      fr = self.firing_rate()
    else:
      fr = self.firing_rate
    if self.mode == 'rate':
      return fr * dt / 1e3
    elif self.mode == 'spiking':
      if fr * dt > 1e3:
        raise ValueError(f'dt is too big, so that dt * fr > 1e3.')
      return fr * dt / 1e3
    else:
      raise ValueError('Please set mode as "rate" or "spiking"')

  def set_name(self, name: str):
    self.name = name
    return self

  def set_mode(self, mode: str):
    self.mode = mode
    return self

  def shift(self, r_num, s_num):
    self._r_start += r_num
    self._r_end += r_num
    self._s_start += s_num
    self._s_end += s_num

  def __add__(self, other: Union['Feature', '_FeatSet']):
    if isinstance(other, Feature):
      other.shift(self.num_rate, self.num_spike)
      return _FeatSet(self, other)
    elif isinstance(other, _FeatSet):
      other.shift(self.num_rate, self.num_spike)
      return _FeatSet(self, *other.fts)
    else:
      raise ValueError(f'Only support addition with {Feature.__name__} or {_FeatSet.__name__}, '
                       f'but got {type(other)}')

  def __repr__(self):
    return f'{self.__class__.__name__}("{self.name}", rate={self.num_rate}, spike={self.num_spike})'


class CircleFeature(Feature):
  def __init__(
      self,
      n_rate: int = 1,
      n_spike: int = 20,
      fr: Union[int, float, Callable] = 30.,
      limits: Tuple[jnp.number, jnp.number] = (0., np.pi * 2),
      name: str = None,
  ):
    super().__init__(n_rate, n_spike, fr, name)
    assert len(limits) == 2
    self.v_min = limits[0]
    self.v_max = limits[1]
    self.v_range = limits[1] - limits[0]


def is_time_duration(x):
  if isinstance(x, (int, float, Callable)):
    return x
  else:
    raise TypeError('A time duration should be an integer/float/function.')


def is_feature(x):
  if isinstance(x, Feature):
    return x
  else:
    raise TypeError(f'A feature should be an instance of {Feature.__name__}.')


class CognitiveTask(Dataset):
  """
  Base class for making a cognitive task.
  It is necessary to override the ``__getitem__`` and ``__len__`` method.

  .. note::

      :attr:`transforms` and the combination of :attr:`input_transform` and
      :attr:`target_transform` are mutually exclusive.

  Parameters
  ----------
  input_transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

  target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.

  """

  times = Sequence[str]
  _repr_indent = 4

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      num_trial: int = 1024,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ) -> None:
    super().__init__(seed=seed)
    self.input_transform = input_transform
    self.target_transform = target_transform
    self.transforms = TransformIT(input_transform, target_transform)
    self.dt = bp.check.is_float(dt, allow_int=True)
    self.num_trial = bp.check.is_integer(num_trial, )

  @property
  def num_inputs(self) -> int:
    raise NotImplementedError

  @property
  def num_outputs(self) -> int:
    raise NotImplementedError

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
    return self.sample_a_trial(index)[:2]

  def sample_a_trial(self, index) -> Tuple:
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


def get_item(self: CognitiveTask,
             index: int,
             max_seq_len: Optional[int],
             return_start_info: bool,
             padding: str):
  if max_seq_len is None:
    res = self.sample_a_trial(index)
    if return_start_info:
      return res[0], res[1], res[2]
    else:
      return res[0], res[1]
  else:
    trial_len = 0
    one_x_data = []
    one_y_data = []
    start_info = []
    if padding == 'new_trial':
      while trial_len < max_seq_len:
        res = self.sample_a_trial(index)
        trial_len += res[0].shape[0]
        one_x_data.append(res[0])
        one_y_data.append(res[1])
        start_info.append(res[2])
    elif padding == 'repeat_trial':
      res = self.sample_a_trial(index)
      while trial_len < max_seq_len:
        trial_len += res[0].shape[0]
        one_x_data.append(res[0])
        one_y_data.append(res[1])
        start_info.append(res[2])
    else:
      raise ValueError
    X = np.concatenate(one_x_data, axis=0)[: max_seq_len]
    Y = np.concatenate(one_y_data, axis=0)[: max_seq_len]
    if return_start_info:
      S = np.concatenate(start_info, axis=0)[: max_seq_len]
      return X, Y, S
    else:
      return X, Y


class TaskLoader(DataLoader):
  """Dataloader for cognitive task.
  
  Args:
    dataset (CognitiveTask): The dataset.
    max_seq_len (int, optional): The maximum sequence length. Default None.
    padding (str, optional): The padding stragety. Currently only supports predefined padding. 
    data_first_axis (str): The first axis of the data.
    
  Other arguments please see :py:class:`~.DataLoader`.
    
  """

  def __init__(
      self,
      dataset: CognitiveTask,
      max_seq_len: Optional[int] = None,
      padding: Optional[str] = None,
      return_start_info: bool = False,  # this argument is experimental
      data_first_axis: str = 'T',
      **kwargs
  ):
    assert isinstance(dataset, CognitiveTask)

    assert data_first_axis in ['B', 'T'], 'Only support data_first_axis of "B" and "T".'
    self.data_first_axis = data_first_axis
    self.return_start_info = return_start_info
    self.max_seq_len = max_seq_len
    self.padding = padding
    if max_seq_len is not None:
      assert isinstance(max_seq_len, int)
      padding = 'new_trial' if padding is None else padding
      assert padding in ['new_trial', 'repeat_trial']
    dataset._get_item = functools.partial(get_item,
                                          dataset,
                                          max_seq_len=max_seq_len,
                                          return_start_info=return_start_info,
                                          padding=padding)
    super().__init__(dataset, **kwargs)

  def get_batch(self):
    batch = super().get_batch()
    if (batch is not None) and (self.data_first_axis == 'T'):
      batch = jax.tree_util.tree_map(_time_first, batch)
    return batch

  @classmethod
  def _get_one_data(cls, dataset, index):
    return dataset._get_item(index)

  def __len__(self):
    if self.drop_last:
      return len(self.dataset) // self.batch_size
    else:
      return math.ceil(len(self.dataset) / self.batch_size)


def _time_first(data):
  if data.ndim > 1:
    if isinstance(data, np.ndarray):
      return np.moveaxis(data, 0, 1)
    elif isinstance(data, jax.Array):
      return jnp.moveaxis(data, 0, 1)
    elif isinstance(data, bm.Array):
      return bm.moveaxis(data, 0, 1)
    else:
      raise TypeError
  else:
    return data
