# -*- coding: utf-8 -*-

import os
import os.path
from typing import Any
from typing import Callable, List, Optional

from brainpy_datasets._src.base import Dataset
from brainpy_datasets._src.transforms import TransformIT

__all__ = [
  'VisionDataset'
]


class VisionDataset(Dataset):
  """
  Base Class For making datasets which are compatible with torchvision.
  It is necessary to override the ``__getitem__`` and ``__len__`` method.

  Args:
      root (string): Root directory of dataset.
      input_transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.

  .. note::

      :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
  """

  _repr_indent = 4

  def __init__(
      self,
      root: str,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ) -> None:
    super().__init__()
    if isinstance(root, (str, bytes)):
      root = os.path.expanduser(root)
    self.root = root
    self.input_transform = input_transform
    self.target_transform = target_transform
    self.transforms = TransformIT(input_transform, target_transform)

  def __getitem__(self, index: int) -> Any:
    """
    Args:
        index (int): Index

    Returns:
        (Any): Sample and meta data, optionally transformed by the respective transforms.
    """
    raise NotImplementedError

  def __len__(self) -> int:
    raise NotImplementedError

  def __repr__(self) -> str:
    head = "Dataset " + self.__class__.__name__
    body = [f"Number of datapoints: {self.__len__()}"]
    if self.root is not None:
      body.append(f"Root location: {self.root}")
    body += self.extra_repr().splitlines()
    if self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return "\n".join(lines)

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def extra_repr(self) -> str:
    return ""
