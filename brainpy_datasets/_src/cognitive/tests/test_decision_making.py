from absl.testing import parameterized

from brainpy_datasets._src.cognitive import decision_making


class TestContextDecisionMakingTask(parameterized.TestCase):
  @parameterized.product(
    t_fixation=[500, 600],
    t_decision=[500, 600],
  )
  def test_ContextDecisionMaking(self, t_fixation, t_decision):
    task = decision_making.ContextDecisionMaking(100, t_decision=t_decision, t_fixation=t_fixation)
    X, Y = task[0]
    print(X.shape, Y.shape)
    self.assertTrue(X.shape[0] == Y.shape[0])

  @parameterized.product(t_delay=[0, 100], )
  def test_SingleContextDecisionMaking(self, t_delay):
    task = decision_making.SingleContextDecisionMaking(100, t_delay=t_delay)
    X, Y = task[0]
    print(X.shape, Y.shape)
    self.assertTrue(X.shape[0] == Y.shape[0])


class TestPerceptualDecisionMakingTask(parameterized.TestCase):
  @parameterized.product(
    t_cue=[10, 20],
    t_bin=[200, 300],
    t_fixation=[500, 600],
    t_decision=[500, 600],
    n_bin=[4, 6, 8],
  )
  def test_PulseDecisionMaking(self, t_cue, t_bin, t_fixation, t_decision, n_bin):
    task = decision_making.PulseDecisionMaking(100,
                                               t_cue=t_cue,
                                               t_decision=t_decision,
                                               t_fixation=t_fixation,
                                               t_bin=t_bin,
                                               n_bin=n_bin,
                                               )
    X, Y = task.sample_trial(0)

    n = int(((t_cue + t_bin) * n_bin + t_fixation + t_decision) / 10)
    self.assertTrue(X[0].shape[0] == n)
    self.assertTrue(Y[0].shape[0] == n)

  @parameterized.product(t_delay=[0, 100], )
  def test_PerceptualDecisionMaking(self, t_delay):
    task = decision_making.PerceptualDecisionMaking(t_delay=t_delay)
    X, Y = task[0]
    print(X.shape)
    print(Y.shape)
