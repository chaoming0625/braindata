
from typing import Union, Callable


__all__ = [
  'initialize'
]


def initialize(data: Union[int, float, Callable], allow_none: bool=False):
  if data is None:
    if allow_none:
      return None
    else:
      raise TypeError
  if isinstance(data, (int, float)):
    return data
  elif isinstance(data, Callable):
    return data()
  else:
    raise TypeError
