# -*- coding: utf-8 -*-

from typing import Any, Callable, List

from brainpy_datasets._src.base import Dataset

__all__ = [
  'ChaosDataset'
]


class ChaosDataset(Dataset):
  _repr_indent = 4

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
    body += self.extra_repr().splitlines()
    if hasattr(self, "transforms"):
      if self.transforms is not None:
        body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return "\n".join(lines)

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def extra_repr(self) -> str:
    return ""


