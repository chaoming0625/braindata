import multiprocessing
import queue
from itertools import cycle
from typing import TypeVar, Callable

import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from .base import Dataset

T_co = TypeVar('T_co', covariant=True)

__all__ = [
  'DataLoader',
]


class DataLoader(object):
  """Data loader.

  Args:
    dataset (Dataset): dataset from which to load the data.
    batch_size (int): how many samples per batch to load (default: ``1``).
    shuffle (bool): set to ``True`` to have the data reshuffled
        at every epoch (default: ``False``).
    num_workers (int): how many subprocesses to use for data
        loading. ``0`` means that the data will be loaded in the main process.
        (default: ``0``)
    pin_memory (bool): If ``True``, the data loader will copy Tensors
        into device/CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        see the example below.
    drop_last (bool): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    prefetch_factor (int): Number of batches loaded
        in advance by each worker. ``2`` means there will be a total of
        2 * num_workers batches prefetched across all workers. (default: ``2``)
    collate_fn (Callable, optional): merges a list of samples to form a
        mini-batch of Tensor(s).  Used when using batched loading from a
        map-style dataset.

  """

  def __init__(
      self,
      dataset: Dataset,
      batch_size: int = 1,
      num_workers: int = 0,
      shuffle: bool = False,
      drop_last: bool = False,
      pin_memory: bool = False,
      prefetch_factor: int = 2,
      collate_fn: Callable = None,
  ):
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.pin_memory = pin_memory
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor
    self._collate_fn = default_collate_fn if collate_fn is None else collate_fn

    self._num_data = len(dataset)
    self._all_indices = np.arange(self._num_data)
    if self.shuffle:
      self._all_indices = np.random.permutation(self._all_indices)
    self._index = 0
    self._index_prefetch = 0
    self._index_queues = []
    self._data_workers = []
    self._cache = {}
    self._event = multiprocessing.Event()
    if num_workers > 0:
      self._output_queue = multiprocessing.Queue()
      self._worker_cycle = cycle(range(num_workers))
      for _ in range(num_workers):
        index_queue = multiprocessing.Queue()
        if hasattr(dataset, 'rng'):
          if isinstance(getattr(dataset, 'rng'), (np.random.RandomState, bm.random.RandomState)):
            dataset.rng.seed(dataset.rng.randint(100000))
        worker = multiprocessing.Process(
          target=self._worker_fn,
          args=(dataset, index_queue, self._output_queue, self._event)
        )
        worker.daemon = True
        worker.start()
        self._data_workers.append(worker)
        self._index_queues.append(index_queue)

    self._prefetch()

  def suspend_subprocess(self):
    """Suspend all sub-processes to wait for data loading jobs."""
    if self._event.is_set():
      self._event.clear()

  def resume_subprocess(self):
    """Resume all sub-processes for data loading."""
    if not self._event.is_set():
      self._event.set()

  def get_batch(self):
    if self._index >= self._num_data:
      return None
    batch_size = self._num_data - self._index
    if self.drop_last and batch_size < self.batch_size:
      return None
    batch_size = min(batch_size, self.batch_size)
    return self._collate_fn(self, [self._get() for _ in range(batch_size)])

  def __iter__(self):
    if self.shuffle:
      self._all_indices = np.random.permutation(self._all_indices)
    self._index = 0
    self._index_prefetch = 0
    self._cache = {}
    self.resume_subprocess()
    self._prefetch()
    while True:
      try:
        batch = self.get_batch()
      except Exception as e:
        self.suspend_subprocess()
        raise e
      if batch is None:
        break
      else:
        yield batch
    self.suspend_subprocess()

  def __del__(self):
    try:
      for i, w in enumerate(self._data_workers):
        self._index_queues[i].put(None)
        w.join(timeout=5.0)
      if self.num_workers > 0:
        self._output_queue.cancel_join_thread()
        self._output_queue.close()
      for q in self._index_queues:
        q.cancel_join_thread()
        q.close()
    except:
      pass
    finally:
      for w in self._data_workers:
        if w.is_alive():
          w.terminate()

  def _get(self):
    self._prefetch()
    if self.num_workers > 0:
      if self._index in self._cache:
        data = self._cache[self._index]
        del self._cache[self._index]
      else:
        while True:
          try:
            (index, data_index, thread_data) = self._output_queue.get(timeout=0)
          except queue.Empty:  # output queue empty, keep trying
            continue
          if index == self._index:  # found our item, ready to return
            data = thread_data
            break
          else:  # item isn't the one we want, cache for later
            self._cache[index] = thread_data
    else:
      data = self._get_one_data(self.dataset, self._all_indices[self._index])
    self._index += 1
    return data

  def _prefetch(self):
    while ((self.num_workers > 0) and
           (self._index_prefetch < self._num_data) and
           (self._index_prefetch < self._index + 2 * self.num_workers * self.batch_size)):
      # if the prefetch_index hasn't reached the end of the dataset
      # and it is not 2 batches ahead, add indexes to the index queues
      true_index = self._all_indices[self._index_prefetch]
      index = self._index_prefetch
      self._index_queues[next(self._worker_cycle)].put((index, true_index))
      self._index_prefetch += 1

  @classmethod
  def _get_one_data(cls, dataset, index):
    return dataset[index]

  @classmethod
  def _worker_fn(cls, dataset, index_queue, output_queue, event):
    while True:
      event.wait()
      # Worker function, simply reads indices from index_queue, and adds the
      # dataset element to the output_queue
      try:
        index = index_queue.get(timeout=0)
      except queue.Empty:
        # time.sleep(1.)
        continue
      if index is None:
        break
      index, data_index = index
      data = cls._get_one_data(dataset, data_index)
      output_queue.put((index, data_index, data))


def default_collate_fn(cls, batch):
  if isinstance(batch[0], np.ndarray):
    data = np.stack(batch)
    if cls.pin_memory:
      data = jnp.asarray(data)
    return data
  elif isinstance(batch[0], (jax.Array, bm.Array)):
    data = bm.stack(batch)
    return data
  elif isinstance(batch[0], (int, float)):
    data = np.array(batch)
    if cls.pin_memory:
      data = jnp.asarray(data)
    return data
  elif isinstance(batch[0], (list, tuple)):
    return tuple(default_collate_fn(cls, var) for var in zip(*batch))
  else:
    raise TypeError
