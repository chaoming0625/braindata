from absl.testing import parameterized

from brainpy_datasets._src.cognitive import decision_making
from brainpy_datasets._src.cognitive import working_memory
from brainpy_datasets._src.cognitive import reasoning
from brainpy_datasets._src.cognitive import others
from brainpy_datasets._src.cognitive.base import TaskLoader


class TestRun(parameterized.TestCase):
  @parameterized.product(model=decision_making.__all__,
                         padding=['repeat_trial', 'new_trial'],
                         return_start_info=[True, False])
  def test_decision_making(self, model, padding, return_start_info):
    loader = TaskLoader(getattr(decision_making, model)(),
                        max_seq_len=100,
                        padding=padding,
                        batch_size=32,
                        return_start_info=return_start_info)
    if return_start_info:
      x, y, s = next(iter(loader))
      print(x.shape, y.shape, s.shape)
    else:
      x, y = next(iter(loader))
      print(x.shape, y.shape)

  @parameterized.product(model=working_memory.__all__,
                         padding=['repeat_trial', 'new_trial'],
                         return_start_info=[True, False])
  def test_working_memory(self, model, padding, return_start_info):

    print(model)
    loader = TaskLoader(getattr(working_memory, model)(),
                        max_seq_len=100,
                        padding=padding,
                        batch_size=32,
                        return_start_info=return_start_info)
    if return_start_info:
      x, y, s = next(iter(loader))
      print(x.shape, y.shape, s.shape)
    else:
      x, y = next(iter(loader))
      print(x.shape, y.shape)

  @parameterized.product(model=reasoning.__all__,
                         padding=['repeat_trial', 'new_trial'],
                         return_start_info=[True, False])
  def test_reasoning(self, model, padding, return_start_info):

    print(model)
    loader = TaskLoader(getattr(reasoning, model)(),
                        max_seq_len=100,
                        padding=padding,
                        batch_size=32,
                        return_start_info=return_start_info)
    if return_start_info:
      x, y, s = next(iter(loader))
      print(x.shape, y.shape, s.shape)
    else:
      x, y = next(iter(loader))
      print(x.shape, y.shape)

  @parameterized.product(model=others.__all__,
                         padding=['repeat_trial', 'new_trial'],
                         return_start_info=[True, False])
  def test_others(self, model, padding, return_start_info):

    print(model)
    loader = TaskLoader(getattr(others, model)(),
                        max_seq_len=100,
                        padding=padding,
                        batch_size=32,
                        return_start_info=return_start_info)
    if return_start_info:
      x, y, s = next(iter(loader))
      print(x.shape, y.shape, s.shape)
    else:
      x, y = next(iter(loader))
      print(x.shape, y.shape)
