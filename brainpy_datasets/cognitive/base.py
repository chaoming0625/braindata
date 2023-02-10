from typing import Callable, Optional, Any, Union, Sequence

import numpy as np

import brainpy as bp
from brainpy_datasets.base import Dataset
from brainpy_datasets.transforms.base import TransformIT

__all__ = [
  'CognitiveTask',
]


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

  def __getitem__(self, index: int) -> Any:
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
  pass

