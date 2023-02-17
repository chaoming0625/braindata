
import brainpy_datasets as bd
from brainpy_datasets._src.cognitive.base import Feature, _FeatSet
import unittest



class TestFeature(unittest.TestCase):
  def test(self):
    fixation = Feature('fixation', 1, 20)
    noise = Feature('noise', 1, 20)
    left = Feature('left', 1, 20)
    right = Feature('left', 1, 20)
    print(fixation, noise, left, right)

    r = fixation + noise + left + right
    print(r)
    self.assertTrue(r._r_num == 4)
    self.assertTrue(r._s_num == 80)
    for i, stage in enumerate(r.fts):
      self.assertTrue(stage._r_start == 1 * i)
      self.assertTrue(stage._r_end == 1 * (i + 1))
      self.assertTrue(stage._s_start == 20 * i)
      self.assertTrue(stage._s_end == 20 * (i + 1))






