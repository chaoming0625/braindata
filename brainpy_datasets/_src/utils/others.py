from typing import Union, Callable

__all__ = [
  'initialize'
]


def initialize(data: Union[int, float, Callable], allow_none: bool = False):
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


def initialize2(data: Union[int, float, Callable], dt, allow_none: bool = False):
  if data is None:
    if allow_none:
      return None
    else:
      raise TypeError
  if isinstance(data, (int, float)):
    return int(data / dt)
  elif isinstance(data, Callable):
    return int(data() / dt)
  else:
    raise TypeError
