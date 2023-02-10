from typing import Optional, Callable, Any, Tuple, List

__all__ = [
  'TransformIT',
]


class Transform(object):
  """Base class for transforming dataset."""
  pass


class TransformIT(Transform):
  """The transformation which transform the dataset with ``(I, O)`` pair.

  Parameters
  ----------
  input_transform: Optional, Callable
    The input transform.

  target_transform: Optional, Callable
    The target transform.

  """

  def __init__(
      self,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None
  ) -> None:
    self.input_transform = input_transform
    self.target_transform = target_transform

  def __call__(self, input: Any, output: Any) -> Tuple[Any, Any]:
    if self.input_transform is not None:
      input = self.input_transform(input)
    if self.target_transform is not None:
      output = self.target_transform(output)
    return input, output

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = repr(transform).splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.input_transform is not None:
      body += self._format_transform_repr(self.input_transform, "Input transform: ")
    if self.target_transform is not None:
      body += self._format_transform_repr(self.target_transform, "Target transform: ")
    return "\n".join(body)


class TransformTX(Transform):
  """The transformation which transform the dataset with (X, Y) pair.

  Parameters
  ----------
  t_transform: Optional, Callable
    The transform for the time.

  x_transform: Optional, Callable
    The transform for the x variable.

  """

  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None
  ) -> None:
    self.t_transform = t_transform
    self.x_transform = x_transform

  def __call__(self, t: Any, x: Any) -> Tuple[Any, Any]:
    if self.t_transform is not None: t = self.t_transform(t)
    if self.x_transform is not None: x = self.x_transform(x)
    return t, x

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.t_transform is not None:
      body += self._format_transform_repr(self.t_transform, "T transform: ")
    if self.x_transform is not None:
      body += self._format_transform_repr(self.x_transform, "X transform: ")
    return "\n".join(body)


class TransformTXY(Transform):
  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None,
      y_transform: Optional[Callable] = None,
  ) -> None:
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.y_transform = y_transform

  def __call__(self, t: Any, x: Any, y: Any) -> Tuple[Any, Any, Any]:
    if self.t_transform is not None: t = self.t_transform(t)
    if self.x_transform is not None: x = self.x_transform(x)
    if self.y_transform is not None: y = self.y_transform(y)
    return t, x, y

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.t_transform is not None:
      body += self._format_transform_repr(self.t_transform, "T transform: ")
    if self.x_transform is not None:
      body += self._format_transform_repr(self.x_transform, "X transform: ")
    if self.y_transform is not None:
      body += self._format_transform_repr(self.y_transform, "Y transform: ")
    return "\n".join(body)




class TransformTXYZ:
  def __init__(
      self,
      t_transform: Optional[Callable] = None,
      x_transform: Optional[Callable] = None,
      y_transform: Optional[Callable] = None,
      z_transform: Optional[Callable] = None,
  ) -> None:
    self.t_transform = t_transform
    self.x_transform = x_transform
    self.y_transform = y_transform
    self.z_transform = z_transform

  def __call__(self, t: Any, x: Any, y: Any, z: Any) -> Tuple[Any, Any, Any, Any]:
    if self.t_transform is not None: t = self.t_transform(t)
    if self.x_transform is not None: x = self.x_transform(x)
    if self.y_transform is not None: y = self.y_transform(y)
    if self.z_transform is not None: z = self.z_transform(z)
    return t, x, y, z

  def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
    lines = transform.__repr__().splitlines()
    return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

  def __repr__(self) -> str:
    body = [self.__class__.__name__]
    if self.t_transform is not None:
      body += self._format_transform_repr(self.t_transform, "T transform: ")
    if self.x_transform is not None:
      body += self._format_transform_repr(self.x_transform, "X transform: ")
    if self.y_transform is not None:
      body += self._format_transform_repr(self.y_transform, "Y transform: ")
    if self.z_transform is not None:
      body += self._format_transform_repr(self.z_transform, "Z transform: ")
    return "\n".join(body)

