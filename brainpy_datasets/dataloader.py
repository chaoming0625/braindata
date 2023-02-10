import multiprocessing
import queue
from itertools import cycle
from typing import TypeVar

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
    batch_size (int, optional): how many samples per batch to load (default: ``1``).
    shuffle (bool, optional): set to ``True`` to have the data reshuffled
        at every epoch (default: ``False``).
    num_workers (int, optional): how many subprocesses to use for data
        loading. ``0`` means that the data will be loaded in the main process.
        (default: ``0``)
    pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
        into device/CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        see the example below.
    drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
        in advance by each worker. ``2`` means there will be a total of
        2 * num_workers batches prefetched across all workers. (default: ``2``)

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
  ):
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.pin_memory = pin_memory
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor

    self._index = 0
    self._index_prefetch = 0
    self._index_queues = []
    self._data_workers = []
    self._cache = {}
    if num_workers > 0:
      self._output_queue = multiprocessing.Queue()
      self._worker_cycle = cycle(range(num_workers))

      for _ in range(num_workers):
        index_queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
          target=_worker_fn,
          args=(dataset, index_queue, self._output_queue)
        )
        worker.daemon = True
        worker.start()
        self._data_workers.append(worker)
        self._index_queues.append(index_queue)

    self._prefetch()

  def __next__(self):
    if self._index >= len(self.dataset):
      raise StopIteration
    batch_size = len(self.dataset) - self._index
    if self.drop_last and batch_size < self.batch_size:
      raise StopIteration
    batch_size = min(batch_size, self.batch_size)
    return self._collate_fn([self._get() for _ in range(batch_size)])

  def __iter__(self):
    self._index = 0
    self._cache = {}
    self._index_prefetch = 0
    self._prefetch()
    return self

  def __del__(self):
    try:
      for i, w in enumerate(self._data_workers):
        self._index_queues[i].put(None)
        w.join(timeout=5.0)
      for q in self._index_queues:
        q.cancel_join_thread()
        q.close()
      if self.num_workers > 0:
        self._output_queue.cancel_join_thread()
        self._output_queue.close()
    finally:
      for w in self._data_workers:
        if w.is_alive():
          w.terminate()

  def _get(self):
    if self.num_workers > 0:
      self._prefetch()
      if self._index in self._cache:
        item = self._cache[self._index]
        del self._cache[self._index]
      else:
        while True:
          try:
            (index, data) = self._output_queue.get(timeout=0)
          except queue.Empty:  # output queue empty, keep trying
            continue
          if index == self._index:  # found our item, ready to return
            item = data
            break
          else:  # item isn't the one we want, cache for later
            self._cache[index] = data
    else:
      item = self.dataset[self._index]
    self._index += 1
    return item

  def _prefetch(self):
    while ((self.num_workers > 0) and (self._index_prefetch < len(self.dataset)) and
           (self._index_prefetch < self._index + 2 * self.num_workers * self.batch_size)):
      # if the prefetch_index hasn't reached the end of the dataset
      # and it is not 2 batches ahead, add indexes to the index queues
      self._index_queues[next(self._worker_cycle)].put(self._index_prefetch)
      self._index_prefetch += 1

  def _collate_fn(self, batch):
    if isinstance(batch[0], np.ndarray):
      data = np.stack(batch)
      if self.pin_memory:
        data = jnp.asarray(data)
      return data
    if isinstance(batch[0], (int, float)):
      data = np.array(batch)
      if self.pin_memory:
        data = jnp.asarray(data)
      return data
    if isinstance(batch[0], (list, tuple)):
      return tuple(self._collate_fn(var) for var in zip(*batch))


def _worker_fn(dataset, index_queue, output_queue):
    while True:
      # Worker function, simply reads indices from index_queue, and adds the
      # dataset element to the output_queue
      try:
        index = index_queue.get(timeout=0)
      except queue.Empty:
        continue
      if index is None:
        break
      data = dataset[index]
      output_queue.put((index, data))

