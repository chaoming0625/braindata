# -*- coding: utf-8 -*-

import os
import os.path
from typing import Any
from typing import Callable, List, Optional, Tuple

from brainpy_datasets.base import Dataset

__all__ = [
  'VisionDataset'
]


class StandardTransform:
  def __init__(
      self,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None
  ) -> None:
    self.transform = transform
    self.target_transform = target_transform

  def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
    if self.transform is not None:
      input = self.transform(input)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return input, target

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.transform is not None:
      body += self._format_transform_repr(self.transform, "Transform: ")
    if self.target_transform is not None:
      body += self._format_transform_repr(self.target_transform, "Target transform: ")

    return "\n".join(body)


class VisionDataset(Dataset):
  """
  Base Class For making datasets which are compatible with torchvision.
  It is necessary to override the ``__getitem__`` and ``__len__`` method.

  Args:
      root (string): Root directory of dataset.
      transform (callable, optional): A function/transform that  takes in an PIL image
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
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ) -> None:
    if isinstance(root, (str, bytes)):
      root = os.path.expanduser(root)
    self.root = root

    # for backwards-compatibility
    self.transform = transform
    self.target_transform = target_transform
    self.transforms = StandardTransform(transform, target_transform)

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
    if hasattr(self, "transforms") and self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return "\n".join(lines)

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def extra_repr(self) -> str:
    return ""
