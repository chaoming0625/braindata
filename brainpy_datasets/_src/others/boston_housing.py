"""Boston housing price regression dataset.

This dataset is inspired by the keras API (https://github.com/keras-team/keras).
"""

import os.path

import numpy as np

from brainpy_datasets._src.utils._data_utils_v1 import get_file
from brainpy_datasets._src.utils._data_utils_v2 import check_integrity
from brainpy_datasets._src.base import Dataset

__all__ = [
  'BostonHousing'
]


class BostonHousing(Dataset):
  """Loads the Boston Housing dataset.

    This is a dataset taken from the StatLib library which is maintained at
    Carnegie Mellon University.

    **WARNING:** This dataset has an ethical problem: the authors of this
    dataset included a variable, "B", that may appear to assume that racial
    self-segregation influences house prices. As such, we strongly discourage
    the use of this dataset, unless in the context of illustrating ethical
    issues in data science and machine learning.

    Samples contain 13 attributes of houses at different locations around the
    Boston suburbs in the late 1970s. Targets are the median values of
    the houses at a location (in k$).

    The attributes themselves are defined in the
    [StatLib website](http://lib.stat.cmu.edu/datasets/boston).

    Args:
      root: path where to cache the dataset locally.
      test_split: fraction of the data to reserve as test set.
      seed: Random seed for shuffling the data
          before computing the test split.

    **x_train, x_test**: numpy arrays with shape `(num_samples, 13)`
      containing either the training samples (for x_train),
      or test samples (for y_train).

    **y_train, y_test**: numpy arrays of shape `(num_samples,)` containing the
      target scalars. The targets are float scalars typically between 10 and
      50 that represent the home prices in k$.
    """

  filename = 'boston_housing.npz'
  url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'

  def __init__(
      self,
      root: str,
      split: str,
      test_split: float = 0.2,
      download: bool = False,
      seed: int = None
  ) -> None:
    if isinstance(root, (str, bytes)):
      root = os.path.expanduser(root)

    assert 0 < test_split < 1
    self.root = root
    self.split = split
    assert split in ['train', 'test']

    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError("Dataset not found. You can use download=True to download it")

    with np.load(os.path.join(self.raw_folder, self.filename), allow_pickle=True) as f:
      x = f["x"]
      y = f["y"]

    if seed is not None:
      indices = np.arange(len(x))
      rng = np.random.RandomState(seed)
      rng.shuffle(indices)
      x = x[indices]
      y = y[indices]

    if split == 'train':
      self.data = np.array(x[: int(len(x) * (1 - test_split))])
      self.targets = np.array(y[: int(len(x) * (1 - test_split))])
    else:
      self.data = np.array(x[int(len(x) * (1 - test_split)):])
      self.targets = np.array(y[int(len(x) * (1 - test_split)):])

  @property
  def raw_folder(self) -> str:
    return os.path.join(self.root, self.__class__.__name__)

  def download(self) -> None:
    if self._check_exists():
      print("Files already downloaded and verified")
      return
    os.makedirs(self.raw_folder, exist_ok=True)
    get_file(
      os.path.join(self.raw_folder, self.filename),
      origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz",
      file_hash=(  # noqa: E501
        "f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5"
      ),
    )

  def _check_exists(self):
    return check_integrity(os.path.join(self.raw_folder, self.filename))

