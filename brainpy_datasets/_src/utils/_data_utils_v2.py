# -*- coding: utf-8 -*-

"""
This util file is rewritten from pytorch code.
"""

import bz2
import ctypes
import errno
import gzip
import hashlib
import importlib.machinery
import itertools
import lzma
import os
import os.path
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
import urllib
import urllib.error
import urllib.request
import warnings
import zipfile
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple, Iterator
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import requests
from brainpy import math as bm
from tqdm import tqdm

ENV_HOME = 'BRAINPY_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

USER_AGENT = "brainpy/datasets"


def _get_home():
  home = os.path.expanduser(os.getenv(ENV_HOME,
                                      os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                                             DEFAULT_CACHE_DIR),
                                                   'brainpy')))
  return home


# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_HOME = os.path.join(_get_home(), "datasets", "")
_USE_SHARDED_DATASETS = False


def _download_file_from_remote_location(fpath: str, url: str) -> None:
  pass


def _is_remote_location_available() -> bool:
  return False


def get_dir():
  r"""
  Get the Hub cache directory used for storing downloaded models & weights.

  If :func:`~torch.hub.set_dir` is not called, default path is ``$BRAINPY_HOME/hub`` where
  environment variable ``$BRAINPY_HOME`` defaults to ``$XDG_CACHE_HOME/brainpy``.
  ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
  filesystem layout, with a default value ``~/.cache`` if the environment
  variable is not set.
  """
  # Issue warning to move data if old env is set
  return os.path.join(_get_home(), 'hub')


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
  r"""Loads the serialized object at the given URL.

  If downloaded file is a zip file, it will be automatically
  decompressed.

  If the object is already present in `model_dir`, it's deserialized and
  returned.
  The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
  ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

  Args:
      url (string): URL of the object to download
      model_dir (string, optional): directory in which to save the object
      map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
      progress (bool, optional): whether or not to display a progress bar to stderr.
          Default: True
      check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
          ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
          digits of the SHA256 hash of the contents of the file. The hash is used to
          ensure unique names and to verify the contents of the file.
          Default: False
      file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

  """
  # Issue warning to move data if old env is set
  if os.getenv('BRAINPY_MODEL_ZOO'):
    warnings.warn('BRAINPY_MODEL_ZOO is deprecated, please use env BRAINPY_HOME instead')

  if model_dir is None:
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

  try:
    os.makedirs(model_dir)
  except OSError as e:
    if e.errno == errno.EEXIST:
      # Directory already exists, ignore.
      pass
    else:
      # Unexpected OSError, re-raise.
      raise

  parts = urlparse(url)
  filename = os.path.basename(parts.path)
  if file_name is not None:
    filename = file_name
  cached_file = os.path.join(model_dir, filename)
  if not os.path.exists(cached_file):
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    hash_prefix = None
    if check_hash:
      r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
      hash_prefix = r.group(1) if r else None
    download_url_to_file(url, cached_file, hash_prefix, progress=progress)

  if _is_legacy_zip_format(cached_file):
    return _legacy_zip_load(cached_file, model_dir, map_location)
  return bm.load(cached_file, map_location=map_location)


def _legacy_zip_load(filename, model_dir, map_location):
  warnings.warn('Falling back to the old format < 1.6. This support will be '
                'deprecated in favor of default zipfile format introduced in 1.6. '
                'Please redo torch.save() to save it in the new zipfile format.')
  # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
  #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
  #       E.g. resnet18-5c106cde.pth which is widely used.
  with zipfile.ZipFile(filename) as f:
    members = f.infolist()
    if len(members) != 1:
      raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
    f.extractall(model_dir)
    extraced_name = members[0].filename
    extracted_file = os.path.join(model_dir, extraced_name)
  return bm.load(extracted_file, map_location=map_location)


# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename):
  if zipfile.is_zipfile(filename):
    infolist = zipfile.ZipFile(filename).infolist()
    return len(infolist) == 1 and not infolist[0].is_dir()
  return False


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
  r"""Download object at the given URL to a local path.

  Args:
      url (string): URL of the object to download
      dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
      hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
          Default: None
      progress (bool, optional): whether or not to display a progress bar to stderr
          Default: True

  """
  file_size = None
  req = Request(url, headers={"User-Agent": "brainpy.hub"})
  u = urlopen(req)
  meta = u.info()
  if hasattr(meta, 'getheaders'):
    content_length = meta.getheaders("Content-Length")
  else:
    content_length = meta.get_all("Content-Length")
  if content_length is not None and len(content_length) > 0:
    file_size = int(content_length[0])

  # We deliberately save it in a temp file and move it after
  # download is complete. This prevents a local working checkpoint
  # being overridden by a broken download.
  dst = os.path.expanduser(dst)
  dst_dir = os.path.dirname(dst)
  f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

  try:
    if hash_prefix is not None:
      sha256 = hashlib.sha256()
    with tqdm(total=file_size, disable=not progress,
              unit='B', unit_scale=True, unit_divisor=1024) as pbar:
      while True:
        buffer = u.read(8192)
        if len(buffer) == 0:
          break
        f.write(buffer)
        if hash_prefix is not None:
          sha256.update(buffer)
        pbar.update(len(buffer))

    f.close()
    if hash_prefix is not None:
      digest = sha256.hexdigest()
      if digest[:len(hash_prefix)] != hash_prefix:
        raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                           .format(hash_prefix, digest))
    shutil.move(f.name, dst)
  finally:
    f.close()
    if os.path.exists(f.name):
      os.remove(f.name)


def _get_extension_path(lib_name):
  lib_dir = os.path.dirname(__file__)
  if os.name == "nt":
    # Register the main library location on the default DLL path
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    if with_load_library_flags:
      kernel32.AddDllDirectory.restype = ctypes.c_void_p

    if sys.version_info >= (3, 8):
      os.add_dll_directory(lib_dir)
    elif with_load_library_flags:
      res = kernel32.AddDllDirectory(lib_dir)
      if res is None:
        err = ctypes.WinError(ctypes.get_last_error())
        err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
        raise err

    kernel32.SetErrorMode(prev_error_mode)

  loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

  extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
  ext_specs = extfinder.find_spec(lib_name)
  if ext_specs is None:
    raise ImportError

  return ext_specs.origin

def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
  with open(filename, "wb") as fh:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
      with tqdm(total=response.length) as pbar:
        for chunk in iter(lambda: response.read(chunk_size), ""):
          if not chunk:
            break
          pbar.update(chunk_size)
          fh.write(chunk)


def gen_bar_updater() -> Callable[[int, int, int], None]:
  pbar = tqdm(total=None)

  def bar_update(count, block_size, total_size):
    if pbar.total is None and total_size:
      pbar.total = total_size
    progress_bytes = count * block_size
    pbar.update(progress_bytes - pbar.n)

  return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
  md5 = hashlib.md5()
  with open(fpath, "rb") as f:
    for chunk in iter(lambda: f.read(chunk_size), b""):
      md5.update(chunk)
  return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
  return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
  if not os.path.isfile(fpath):
    return False
  if md5 is None:
    return True
  return check_md5(fpath, md5)


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
  initial_url = url
  headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

  for _ in range(max_hops + 1):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
      if response.url == url or response.url is None:
        return url

      url = response.url
  else:
    raise RecursionError(
      f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
    )


def _get_google_drive_file_id(url: str) -> Optional[str]:
  parts = urlparse(url)

  if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
    return None

  match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
  if match is None:
    return None

  return match.group("id")


def download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
  """Download a file from a url and place it in root.

  Args:
      url (str): URL to download file from
      root (str): Directory to place downloaded file in
      filename (str, optional): Name to save the file under. If None, use the basename of the URL
      md5 (str, optional): MD5 checksum of the download. If None, do not check
      max_redirect_hops (int, optional): Maximum number of redirect hops allowed
  """
  root = os.path.expanduser(root)
  if not filename:
    filename = os.path.basename(url)
  fpath = os.path.join(root, filename)

  os.makedirs(root, exist_ok=True)

  # check if file is already present locally
  if check_integrity(fpath, md5):
    print("Using downloaded and verified file: " + fpath)
    return

  if _is_remote_location_available():
    _download_file_from_remote_location(fpath, url)
  else:
    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
      return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
      print("Downloading " + url + " to " + fpath)
      _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
      if url[:5] == "https":
        url = url.replace("https:", "http:")
        print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
      else:
        raise e

  # check integrity of downloaded file
  if not check_integrity(fpath, md5):
    raise RuntimeError("File not found or corrupted.")


def list_dir(root: str, prefix: bool = False) -> List[str]:
  """List all directories at a given root

  Args:
      root (str): Path to directory whose folders need to be listed
      prefix (bool, optional): If true, prepends the path to each result, otherwise
          only returns the name of the directories found
  """
  root = os.path.expanduser(root)
  directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
  if prefix is True:
    directories = [os.path.join(root, d) for d in directories]
  return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
  """List all files ending with a suffix at a given root

  Args:
      root (str): Path to directory whose folders need to be listed
      suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
          It uses the Python "str.endswith" method and is passed directly
      prefix (bool, optional): If true, prepends the path to each result, otherwise
          only returns the name of the files found
  """
  root = os.path.expanduser(root)
  files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
  if prefix is True:
    files = [os.path.join(root, d) for d in files]
  return files


def _quota_exceeded(first_chunk: bytes) -> bool:
  try:
    return "Google Drive - Quota exceeded" in first_chunk.decode()
  except UnicodeDecodeError:
    return False


def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
  """Download a Google Drive file from  and place it in root.

  Args:
      file_id (str): id of file to be downloaded
      root (str): Directory to place downloaded file in
      filename (str, optional): Name to save the file under. If None, use the id of the file.
      md5 (str, optional): MD5 checksum of the download. If None, do not check
  """
  # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

  url = "https://docs.google.com/uc?export=download"

  root = os.path.expanduser(root)
  if not filename:
    filename = file_id
  fpath = os.path.join(root, filename)

  os.makedirs(root, exist_ok=True)

  if os.path.isfile(fpath) and check_integrity(fpath, md5):
    print("Using downloaded and verified file: " + fpath)
  else:
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
      params = {"id": file_id, "confirm": token}
      response = session.get(url, params=params, stream=True)

    # Ideally, one would use response.status_code to check for quota limits, but google drive is not consistent
    # with their own API, refer https://github.com/pytorch/vision/issues/2992#issuecomment-730614517.
    # Should this be fixed at some place in future, one could refactor the following to no longer rely on decoding
    # the first_chunk of the payload
    response_content_generator = response.iter_content(32768)
    first_chunk = None
    while not first_chunk:  # filter out keep-alive new chunks
      first_chunk = next(response_content_generator)

    if _quota_exceeded(first_chunk):
      msg = (
        f"The daily quota of the file {filename} is exceeded and it "
        f"can't be downloaded. This is a limitation of Google Drive "
        f"and can only be overcome by trying again later."
      )
      raise RuntimeError(msg)

    _save_response_content(itertools.chain((first_chunk,), response_content_generator), fpath)
    response.close()


def _get_confirm_token(response) -> Optional[str]:
  for key, value in response.cookies.items():
    if key.startswith("download_warning"):
      return value

  return None


def _save_response_content(
    response_gen: Iterator[bytes],
    destination: str,
) -> None:
  with open(destination, "wb") as f:
    pbar = tqdm(total=None)
    progress = 0

    for chunk in response_gen:
      if chunk:  # filter out keep-alive new chunks
        f.write(chunk)
        progress += len(chunk)
        pbar.update(progress - pbar.n)
    pbar.close()


def _extract_tar(from_path: str, to_path: str, compression: Optional[str]) -> None:
  with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
    tar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
  ".bz2": zipfile.ZIP_BZIP2,
  ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: str, to_path: str, compression: Optional[str]) -> None:
  with zipfile.ZipFile(
      from_path, "r", compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED
  ) as zip:
    zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[str, str, Optional[str]], None]] = {
  ".tar": _extract_tar,
  ".zip": _extract_zip,
}
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {
  ".bz2": bz2.open,
  ".gz": gzip.open,
  ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
  ".tbz": (".tar", ".bz2"),
  ".tbz2": (".tar", ".bz2"),
  ".tgz": (".tar", ".gz"),
}


def _detect_file_type(file: str) -> Tuple[str, Optional[str], Optional[str]]:
  """Detect the archive type and/or compression of a file.

  Args:
      file (str): the filename

  Returns:
      (tuple): tuple of suffix, archive type, and compression

  Raises:
      RuntimeError: if file has no suffix or suffix is not supported
  """
  suffixes = pathlib.Path(file).suffixes
  if not suffixes:
    raise RuntimeError(
      f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
    )
  suffix = suffixes[-1]

  # check if the suffix is a known alias
  if suffix in _FILE_TYPE_ALIASES:
    return (suffix, *_FILE_TYPE_ALIASES[suffix])

  # check if the suffix is an archive type
  if suffix in _ARCHIVE_EXTRACTORS:
    return suffix, suffix, None

  # check if the suffix is a compression
  if suffix in _COMPRESSED_FILE_OPENERS:
    # check for suffix hierarchy
    if len(suffixes) > 1:
      suffix2 = suffixes[-2]

      # check if the suffix2 is an archive type
      if suffix2 in _ARCHIVE_EXTRACTORS:
        return suffix2 + suffix, suffix2, suffix

    return suffix, None, suffix

  valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
  raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


def _decompress(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
  r"""Decompress a file.

  The compression is automatically detected from the file name.

  Args:
      from_path (str): Path to the file to be decompressed.
      to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
      remove_finished (bool): If ``True``, remove the file after the extraction.

  Returns:
      (str): Path to the decompressed file.
  """
  suffix, archive_type, compression = _detect_file_type(from_path)
  if not compression:
    raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

  if to_path is None:
    to_path = from_path.replace(suffix, archive_type if archive_type is not None else "")

  # We don't need to check for a missing key here, since this was already done in _detect_file_type()
  compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

  with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
    wfh.write(rfh.read())

  if remove_finished:
    os.remove(from_path)

  return to_path


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
  """Extract an archive.

  The archive type and a possible compression is automatically detected from the file name. If the file is compressed
  but not an archive the call is dispatched to :func:`decompress`.

  Args:
      from_path (str): Path to the file to be extracted.
      to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
          used.
      remove_finished (bool): If ``True``, remove the file after the extraction.

  Returns:
      (str): Path to the directory the file was extracted to.
  """
  if to_path is None:
    to_path = os.path.dirname(from_path)

  suffix, archive_type, compression = _detect_file_type(from_path)
  if not archive_type:
    return _decompress(
      from_path,
      os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
      remove_finished=remove_finished,
    )

  # We don't need to check for a missing key here, since this was already done in _detect_file_type()
  extractor = _ARCHIVE_EXTRACTORS[archive_type]

  extractor(from_path, to_path, compression)
  if remove_finished:
    os.remove(from_path)

  return to_path


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
  download_root = os.path.expanduser(download_root)
  if extract_root is None:
    extract_root = download_root
  if not filename:
    filename = os.path.basename(url)

  download_url(url, download_root, filename, md5)

  archive = os.path.join(download_root, filename)
  print(f"Extracting {archive} to {extract_root}")
  extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable: Iterable) -> str:
  return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Iterable[T] = None,
    custom_msg: Optional[str] = None,
) -> T:
  if not isinstance(value, (str, bytes)):
    if arg is None:
      msg = "Expected type str, but got type {type}."
    else:
      msg = "Expected type str for argument {arg}, but got type {type}."
    msg = msg.format(type=type(value), arg=arg)
    raise ValueError(msg)

  if valid_values is None:
    return value

  if value not in valid_values:
    if custom_msg is not None:
      msg = custom_msg
    else:
      msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
      msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
    raise ValueError(msg)

  return value
