# -*- coding: utf-8 -*-
"""
MNIST dataset, which is rewritten according to torchvision APIs.
"""


import codecs
import os
import os.path
import shutil
import string
import sys
from typing import Any
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import URLError

import numpy as np

from .base import VisionDataset
from brainpy_datasets._src.utils._data_utils_v2 import (download_and_extract_archive,
                                                        extract_archive,
                                                        verify_str_arg,
                                                        check_integrity)

__all__ = [
  'MNIST',
  'FashionMNIST',
  'KMNIST',
  'EMNIST',
  'QMNIST',
]


class MNIST(VisionDataset):
  """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

  Args:
      root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
          and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
      split (str, optional): If 'train', creates dataset from ``train-images-idx3-ubyte``,
          otherwise from ``t10k-images-idx3-ubyte``.
      download (bool, optional): If True, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
      input_transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
  """

  _mirrors = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
  ]

  _resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
  ]

  _train_file = "train.pt"
  _test_file = "test.pt"
  classes = [
    "0 - zero",
    "1 - one",
    "2 - two",
    "3 - three",
    "4 - four",
    "5 - five",
    "6 - six",
    "7 - seven",
    "8 - eight",
    "9 - nine",
  ]

  def __init__(
      self,
      root: str,
      split: str = 'train',
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      download: bool = False,
  ) -> None:
    super().__init__(root, input_transform=input_transform, target_transform=target_transform)
    self.split = split  # training set or test set

    if self._check_legacy_exist():
      self.data, self.targets = self._load_legacy_data()
      return

    if download:
      self._download()

    if not self._check_exists():
      raise RuntimeError("Dataset not found. You can use download=True to download it")

    self.data, self.targets = self._load_data()

  def _check_legacy_exist(self):
    processed_folder_exists = os.path.exists(self._processed_folder)
    if not processed_folder_exists:
      return False
    return all(check_integrity(os.path.join(self._processed_folder, file))
               for file in (self._train_file, self._test_file))

  def _load_legacy_data(self):
    assert self.split in ['train', 'test']
    data_file = self._train_file if self.split == 'train' else self._test_file
    return np.load(os.path.join(self._processed_folder, data_file))

  def _load_data(self):
    assert self.split in ['train', 'test']

    image_file = f"{'train' if self.split == 'train' else 't10k'}-images-idx3-ubyte"
    data = read_image_file(os.path.join(self._raw_folder, image_file))

    label_file = f"{'train' if self.split == 'train' else 't10k'}-labels-idx1-ubyte"
    targets = read_label_file(os.path.join(self._raw_folder, label_file))

    return data, targets

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img = self.data[index]
    target = self.targets[index]

    if self.input_transform is not None:
      img = self.input_transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self) -> int:
    return len(self.data)

  @property
  def _raw_folder(self) -> str:
    return os.path.join(self.root, self.__class__.__name__, "raw")

  @property
  def _processed_folder(self) -> str:
    return os.path.join(self.root, self.__class__.__name__, "processed")

  @property
  def class_to_idx(self) -> Dict[str, int]:
    return {_class: i for i, _class in enumerate(self.classes)}

  def _check_exists(self) -> bool:
    return all(check_integrity(os.path.join(self._raw_folder, os.path.splitext(os.path.basename(url))[0]))
               for url, _ in self._resources)

  def _download(self) -> None:
    """Download the MNIST data if it doesn't exist already."""

    if self._check_exists():
      return

    os.makedirs(self._raw_folder, exist_ok=True)

    # download files
    for filename, md5 in self._resources:
      for mirror in self._mirrors:
        url = f"{mirror}{filename}"
        try:
          print(f"Downloading {url}")
          download_and_extract_archive(url, download_root=self._raw_folder, filename=filename, md5=md5)
        except URLError as error:
          print(f"Failed to download (trying next):\n{error}")
          continue
        finally:
          print()
        break
      else:
        raise RuntimeError(f"Error downloading {filename}")

  def extra_repr(self) -> str:
    return f"Split: {self.split}"


class FashionMNIST(MNIST):
  """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

  Args:
      root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
          and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
      split (str, optional): If 'train', creates dataset from ``train-images-idx3-ubyte``,
          otherwise from ``t10k-images-idx3-ubyte``.
      download (bool, optional): If True, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
  """

  _mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

  _resources = [
    ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
  ]
  classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class KMNIST(MNIST):
  """`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset.

  Args:
      root (string): Root directory of dataset where ``KMNIST/raw/train-images-idx3-ubyte``
          and  ``KMNIST/raw/t10k-images-idx3-ubyte`` exist.
      split (str, optional): If 'train', creates dataset from ``train-images-idx3-ubyte``,
          otherwise from ``t10k-images-idx3-ubyte``.
      download (bool, optional): If True, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
  """

  _mirrors = ["http://codh.rois.ac.jp/kmnist/dataset/kmnist/"]

  _resources = [
    ("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
    ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
    ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
    ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134"),
  ]
  classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


class EMNIST(MNIST):
  """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

  Args:
      root (string): Root directory of dataset where ``EMNIST/raw/train-images-idx3-ubyte``
          and  ``EMNIST/raw/t10k-images-idx3-ubyte`` exist.
      category (string): The dataset has 6 different categories: ``byclass``, ``bymerge``,
          ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
          which one to use.
      split (string): If 'train', creates dataset from ``train.pt``,
          otherwise from ``test.pt``.
      download (bool, optional): If True, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
      input_transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
  """

  url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
  md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
  categories = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
  # Merged Classes assumes Same structure for both uppercase and lowercase version
  _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
  _all_classes = set(string.digits + string.ascii_letters)
  classes_split_dict = {
    "byclass": sorted(list(_all_classes)),
    "bymerge": sorted(list(_all_classes - _merged_classes)),
    "balanced": sorted(list(_all_classes - _merged_classes)),
    "letters": ["N/A"] + list(string.ascii_lowercase),
    "digits": list(string.digits),
    "mnist": list(string.digits),
  }

  def __init__(
      self,
      root: str,
      category: str,
      split: str,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      download: bool = False,
  ):
    self.category = verify_str_arg(category, "category", self.categories)
    self.train_file = f"train_{split}.pt"
    self.test_file = f"test_{split}.pt"
    super().__init__(root,
                     split=split,
                     input_transform=input_transform,
                     target_transform=target_transform,
                     download=download)
    self.classes = self.classes_split_dict[self.category]

  @property
  def _file_prefix(self) -> str:
    return f"emnist-{self.category}-{self.split}"

  @property
  def images_file(self) -> str:
    return os.path.join(self._raw_folder, f"{self._file_prefix}-images-idx3-ubyte")

  @property
  def labels_file(self) -> str:
    return os.path.join(self._raw_folder, f"{self._file_prefix}-labels-idx1-ubyte")

  def _load_data(self):
    return read_image_file(self.images_file), read_label_file(self.labels_file)

  def _check_exists(self) -> bool:
    return all(check_integrity(file) for file in (self.images_file, self.labels_file))

  def _download(self) -> None:
    """Download the EMNIST data if it doesn't exist already."""
    if self._check_exists():
      return
    os.makedirs(self._raw_folder, exist_ok=True)
    download_and_extract_archive(self.url, download_root=self._raw_folder, md5=self.md5)
    gzip_folder = os.path.join(self._raw_folder, "gzip")
    for gzip_file in os.listdir(gzip_folder):
      if gzip_file.endswith(".gz"):
        extract_archive(os.path.join(gzip_folder, gzip_file), self._raw_folder)
    shutil.rmtree(gzip_folder)

  def extra_repr(self) -> str:
    return f"Split: {self.category}-{self.split}"


class QMNIST(MNIST):
  """`QMNIST <https://github.com/facebookresearch/qmnist>`_ Dataset.

  Args:
      root (string): Root directory of dataset whose ``raw``
          subdir contains binary files of the datasets.
      split (string): Can be 'train', 'test', 'test10k',
          'test50k', or 'nist' for respectively the mnist compatible
          training set, the 60k qmnist testing set, the 10k qmnist
          examples that match the mnist testing set, the 50k
          remaining qmnist testing examples, or all the nist
          digits. The default is to select 'train' or 'test'
          according to the compatibility argument 'train'.
      compat (bool,optional): A boolean that says whether the target
          for each example is class number (for compatibility with
          the MNIST dataloader) or a torch vector containing the
          full qmnist information. Default=True.
      download (bool, optional): If True, downloads the dataset from
          the internet and puts it in root directory. If dataset is
          already downloaded, it is not downloaded again.
      input_transform (callable, optional): A function/transform that
          takes in an PIL image and returns a transformed
          version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform
          that takes in the target and transforms it.
  """

  subsets = {"train": "train",
             "test": "test",
             "test10k": "test",
             "test50k": "test",
             "nist": "nist"}
  _resources: Dict[str, List[Tuple[str, str]]] = {  # type: ignore[assignment]
    "train": [
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz",
        "ed72d4157d28c017586c42bc6afe6370",
      ),
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz",
        "0058f8dd561b90ffdd0f734c6a30e5e4",
      ),
    ],
    "test": [
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz",
        "1394631089c404de565df7b7aeaf9412",
      ),
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz",
        "5b5b05890a5e13444e108efe57b788aa",
      ),
    ],
    "nist": [
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz",
        "7f124b3b8ab81486c9d8c2749c17f834",
      ),
      (
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz",
        "5ed0e788978e45d4a8bd4b7caec3d79d",
      ),
    ],
  }
  classes = [
    "0 - zero",
    "1 - one",
    "2 - two",
    "3 - three",
    "4 - four",
    "5 - five",
    "6 - six",
    "7 - seven",
    "8 - eight",
    "9 - nine",
  ]

  def __init__(
      self,
      root: str,
      split: str,
      compat: bool = True,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      download: bool = False,
  ) -> None:
    self.category = verify_str_arg(split, "split", tuple(self.subsets.keys()))
    self.compat = compat
    self.data_file = split + ".pt"
    self.train_file = self.data_file
    self.test_file = self.data_file
    super().__init__(root,
                     split,
                     input_transform=input_transform,
                     target_transform=target_transform,
                     download=download)

  @property
  def images_file(self) -> str:
    (url, _), _ = self._resources[self.subsets[self.category]]
    return os.path.join(self._raw_folder, os.path.splitext(os.path.basename(url))[0])

  @property
  def labels_file(self) -> str:
    _, (url, _) = self._resources[self.subsets[self.category]]
    return os.path.join(self._raw_folder, os.path.splitext(os.path.basename(url))[0])

  def _check_exists(self) -> bool:
    return all(check_integrity(file) for file in (self.images_file, self.labels_file))

  def _load_data(self):
    data = read_sn3_pascalvincent_tensor(self.images_file)
    assert data.dtype == np.uint8
    assert data.ndim == 3

    targets = read_sn3_pascalvincent_tensor(self.labels_file).astype(np.int64)
    assert targets.ndim == 2

    if self.category == "test10k":
      data = data[0:10000].copy()
      targets = targets[0:10000].copy()
    elif self.category == "test50k":
      data = data[10000:].copy()
      targets = targets[10000:].copy()

    return data, targets

  def _download(self) -> None:
    """Download the QMNIST data if it doesn't exist already.
    Note that we only download category has been asked for (argument 'category').
    """
    if self._check_exists():
      return

    os.makedirs(self._raw_folder, exist_ok=True)
    split = self._resources[self.subsets[self.category]]

    for url, md5 in split:
      download_and_extract_archive(url, self._raw_folder, md5=md5)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    # redefined to handle the compat flag
    img, target = self.data[index], self.targets[index]
    if self.input_transform is not None:
      img = self.input_transform(img)
    if self.compat:
      target = int(target[0])
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target

  def extra_repr(self) -> str:
    return f"Split: {self.category}"


def get_int(b: bytes) -> int:
  return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
  8: np.uint8,
  9: np.int8,
  11: np.int16,
  12: np.int32,
  13: np.float32,
  14: np.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> np.ndarray:
  """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
  Argument may be a filename, compressed filename, or file object.
  """
  # read
  with open(path, "rb") as f:
    data = f.read()
  # parse
  magic = get_int(data[0:4])
  nd = magic % 256
  ty = magic // 256
  assert 1 <= nd <= 3
  assert 8 <= ty <= 14
  dtype = SN3_PASCALVINCENT_TYPEMAP[ty]
  s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]

  num_bytes_per_value = np.iinfo(dtype).bits // 8
  # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
  # we need to reverse the bytes before we can read them with .frombuffer().
  needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
  parsed = np.frombuffer(bytearray(data), dtype=dtype, offset=(4 * (nd + 1)))
  if needs_byte_reversal:
    parsed = np.flip(parsed, 0)
  assert parsed.shape[0] == np.prod(s) or not strict
  return parsed.reshape(*s)


def read_label_file(path: str) -> np.ndarray:
  x = read_sn3_pascalvincent_tensor(path, strict=False)
  assert x.dtype == np.uint8
  assert x.ndim == 1
  return x.astype(np.float_)


def read_image_file(path: str) -> np.ndarray:
  x = read_sn3_pascalvincent_tensor(path, strict=False)
  assert x.dtype == np.uint8
  assert x.ndim == 3
  return x
