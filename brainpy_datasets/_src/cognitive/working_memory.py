from typing import Union, Optional, Callable

import numpy as np

import brainpy as bp
from brainpy_datasets._src.cognitive.base import FixedLenCogTask
from brainpy_datasets._src.cognitive.base import VariedLenCogTask
from brainpy_datasets._src.cognitive.utils import interval_of
from brainpy_datasets._src.utils.others import initialize
from brainpy_datasets._src.utils.random import TruncExp

__all__ = [
  'DelayComparison',
  'DelayMatchCategory',
  'DelayMatchSample',
  'DelayPairedAssociation',
  'DualDelayMatchSample',
  'GoNoGo',
  'IntervalDiscrimination',
  'PostDecisionWager',
  'ReadySetGo',
]


class DelayComparison(FixedLenCogTask):
  """Delayed comparison.

  The agent needs to compare the magnitude of two stimuli are separated by a
  delay period. The agent reports its decision of the stronger stimulus
  during the decision period.
  """
  metadata = {
    'paper_link': 'https://www.jneurosci.org/content/30/28/9424',
    'paper_name': 'Neuronal Population Coding of Parametric Working Memory',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      vpairs: Optional[np.ndarray] = None,
      t_fixation: Union[int, float] = 500.,
      t_stimulus1: Union[int, float] = 500.,
      t_stimulus2: Union[int, float] = 500.,
      t_delay: Union[int, float] = 1000.,
      t_decision: Union[int, float] = 100.,
      num_trial: int = 1024,
      noise_sigma: float = 1.0,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # Inputs
    if vpairs is None:
      self.vpairs = np.asarray([(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)])
    else:
      self.vpairs = vpairs
    self._vmin = np.min(self.vpairs)
    self._vmax = np.max(self.vpairs)
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_stimulus1 = bp.check.is_float(t_stimulus1, min_bound=0., allow_int=True)
    self.t_stimulus2 = bp.check.is_float(t_stimulus2, min_bound=0., allow_int=True)
    self.t_delay = bp.check.is_float(t_delay, min_bound=0., allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_stimulus1 = int(self.t_stimulus1 / self.dt)
    n_stimulus2 = int(self.t_stimulus2 / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'stimulus1': n_stimulus1,
                          'delay': n_delay,
                          'stimulus2': n_stimulus2,
                          'decision': n_decision, }

    # features
    self._choices = np.asarray([1, 2])
    self._feature_periods = {'fixation': 1, 'choice': 1}

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation', 'stimulus']

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, 2))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)
    v1, v2 = self.vpairs[self.rng.choice(len(self.vpairs))]
    if ground_truth == 2:
      v1, v2 = v2, v1

    ax0_stim1 = interval_of('stimulus1', self._time_periods)
    ax0_stim2 = interval_of('stimulus2', self._time_periods)
    ax0_decision = interval_of('decision', self._time_periods)

    X[:, 0] += 1.
    X[ax0_stim1, 1] += (1 + (v1 - self._vmin) / (self._vmax - self._vmin)) / 2
    X[ax0_stim1, 1] += self.rng.randn(self._time_periods['stimulus1']) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_stim2, 1] += (1 + (v2 - self._vmin) / (self._vmax - self._vmin)) / 2
    X[ax0_stim2, 1] += self.rng.randn(self._time_periods['stimulus2']) * self.noise_sigma / np.sqrt(self.dt)

    Y[ax0_decision] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [(f, 1) for f in self.input_features]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


class DelayMatchCategory(FixedLenCogTask):
  r"""Delayed match-to-category task.

  A sample stimulus is shown during the sample period. The stimulus is
  characterized by a one-dimensional variable, such as its orientation
  between 0 and 360 degree. This one-dimensional variable is separated
  into two categories (for example, 0-180 degree and 180-360 degree).
  After a delay period, a test stimulus is shown. The agent needs to
  determine whether the sample and the test stimuli belong to the same
  category, and report that decision during the decision period.
  """

  metadata = {
    'paper_link': 'https://www.nature.com/articles/nature05078',
    'paper_name': 'Experience-dependent representation of visual categories in parietal cortex',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 500.,
      t_sample: Union[int, float] = 650.,
      t_delay: Union[int, float] = 1000.,
      t_test: Union[int, float] = 650.,
      num_trial: int = 1024,
      num_choice: int = 2,
      noise_sigma: float = 1.0,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # Inputs

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_sample = bp.check.is_float(t_sample, min_bound=0., allow_int=True)
    self.t_delay = bp.check.is_float(t_delay, min_bound=0., allow_int=True)
    self.t_test = bp.check.is_float(t_test, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_sample = int(self.t_sample / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_test = int(self.t_test / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'sample': n_sample,
                          'delay': n_delay,
                          'test': n_test, }

    # features
    self.num_choice = bp.check.is_integer(num_choice)
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    self._choices = np.asarray([1, 2])
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._feature_periods = {'fixation': 1, 'stimulus': num_choice}

    # input / output information
    self.output_features = ['fixation', 'match', 'non-match']
    self.input_features = ['fixation'] + [f'stimulus {i}' for i in range(num_choice)]

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)
    sample_category = self.rng.choice([0, 1])
    test_category = sample_category if ground_truth == 1 else (1 - sample_category)

    sample_theta = (sample_category + self.rng.rand()) * np.pi
    test_theta = (test_category + self.rng.rand()) * np.pi

    stim_sample = np.cos(self._features - sample_theta) * 0.5 + 0.5
    stim_test = np.cos(self._features - test_theta) * 0.5 + 0.5

    ax0_sample = interval_of('sample', self._time_periods)
    ax0_test = interval_of('test', self._time_periods)
    ax1_stim = interval_of('stimulus', self._feature_periods)

    X[:, 0] += 1.
    X[ax0_test, 0] = 0.
    X[ax0_sample, ax1_stim] += stim_sample
    X[ax0_sample, ax1_stim] += self.rng.randn(self._time_periods['sample']) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_test, ax1_stim] += stim_test
    X[ax0_test, ax1_stim] += self.rng.randn(self._time_periods['test']) * self.noise_sigma / np.sqrt(self.dt)

    Y[ax0_test] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [('fixation', 1), ('stimulus', self.num_choice)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


class DelayMatchSample(FixedLenCogTask):
  r"""Delayed match-to-sample task.

  A sample stimulus is shown during the sample period. The stimulus is
  characterized by a one-dimensional variable, such as its orientation
  between 0 and 360 degree. After a delay period, a test stimulus is
  shown. The agent needs to determine whether the sample and the test
  stimuli are equal, and report that decision during the decision period.
  """

  metadata = {
    'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf',
    'paper_name': 'Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 300.,
      t_sample: Union[int, float] = 500.,
      t_delay: Union[int, float] = 1000.,
      t_test: Union[int, float] = 500.,
      t_decision: Union[int, float] = 900.,
      num_trial: int = 1024,
      num_choice: int = 2,
      noise_sigma: float = 1.0,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # Inputs

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_sample = bp.check.is_float(t_sample, min_bound=0., allow_int=True)
    self.t_delay = bp.check.is_float(t_delay, min_bound=0., allow_int=True)
    self.t_test = bp.check.is_float(t_test, min_bound=0., allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_sample = int(self.t_sample / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_test = int(self.t_test / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'sample': n_sample,
                          'delay': n_delay,
                          'test': n_test,
                          'decision': n_decision}

    # features
    self.num_choice = bp.check.is_integer(num_choice)
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    self._choices = np.asarray([1, 2])
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._feature_periods = {'fixation': 1, 'stimulus': num_choice}

    # input / output information
    self.output_features = ['fixation', 'match', 'non-match']
    self.input_features = ['fixation'] + [f'stimulus {i}' for i in range(num_choice)]

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)
    sample_theta = self.rng.choice(self._features)
    test_theta = sample_theta if ground_truth == 1 else np.mod(sample_theta + np.pi, 2 * np.pi)

    stim_sample = np.cos(self._features - sample_theta) * 0.5 + 0.5
    stim_test = np.cos(self._features - test_theta) * 0.5 + 0.5

    ax0_sample = interval_of('sample', self._time_periods)
    ax0_test = interval_of('test', self._time_periods)
    ax0_decision = interval_of('decision', self._time_periods)
    ax1_stim = interval_of('stimulus', self._feature_periods)

    X[:, 0] += 1.
    X[ax0_decision, 0] = 0.
    X[ax0_sample, ax1_stim] += stim_sample
    X[ax0_sample, ax1_stim] += self.rng.randn(self._time_periods['sample']) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_test, ax1_stim] += stim_test
    X[ax0_test, ax1_stim] += self.rng.randn(self._time_periods['test']) * self.noise_sigma / np.sqrt(self.dt)

    Y[ax0_decision] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [('fixation', 1), ('stimulus', self.num_choice)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


class DelayPairedAssociation(FixedLenCogTask):
  r"""Delayed paired-association task.

  The agent is shown a pair of two stimuli separated by a delay period. For
  half of the stimuli-pairs shown, the agent should choose the Go response.
  The agent is rewarded if it chose the Go response correctly.
  """

  metadata = {
    'paper_link': 'https://elifesciences.org/articles/43191',
    'paper_name': 'Active information maintenance in working memory by a sensory cortex',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 0.,
      t_stim1: Union[int, float] = 1000.,
      t_delay1: Union[int, float] = 1000.,
      t_stim2: Union[int, float] = 1000.,
      t_delay2: Union[int, float] = 1000.,
      t_decision: Union[int, float] = 500.,
      num_trial: int = 1024,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_stim1 = bp.check.is_float(t_stim1, min_bound=0., allow_int=True)
    self.t_delay1 = bp.check.is_float(t_delay1, min_bound=0., allow_int=True)
    self.t_stim2 = bp.check.is_float(t_stim2, min_bound=0., allow_int=True)
    self.t_delay2 = bp.check.is_float(t_delay2, min_bound=0., allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_stim1 = int(self.t_stim1 / self.dt)
    n_delay1 = int(self.t_delay1 / self.dt)
    n_stim2 = int(self.t_stim2 / self.dt)
    n_delay2 = int(self.t_delay2 / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'stim1': n_stim1,
                          'delay1': n_delay1,
                          'stim2': n_stim2,
                          'delay2': n_delay2,
                          'decision': n_decision}

    # features
    self._feature_periods = {'fixation': 1, 'stimulus': 4}
    self.pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
    self.association = 0

    # input / output information
    self.output_features = ['fixation', 'go']
    self.input_features = ['fixation'] + [f'stimulus {i}' for i in range(4)]

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    pair = self.pairs[self.rng.choice(len(self.pairs))]
    ground_truth = int(np.diff(pair)[0] % 2 == self.association)

    ax0_stim1 = interval_of('stim1', self._time_periods)
    ax0_stim2 = interval_of('stim2', self._time_periods)
    ax0_decision = interval_of('decision', self._time_periods)

    X[:, 0] += 1.
    X[ax0_stim1, pair[0]] += 1.
    X[ax0_stim2, pair[1]] += 1.
    X[ax0_decision] = 0.

    Y[ax0_decision] = ground_truth

    # # if trial is GO the reward is set to R_MISS and  to 0 otherwise
    # self.r_tmax = self.rewards['miss'] * ground_truth
    # self.performance = 1 - ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [('fixation', 1), ('stimulus', 4)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]




class DualDelayMatchSample(FixedLenCogTask):
  r"""Two-item Delay-match-to-sample.

  The trial starts with a fixation period. Then during the sample period,
  two sample stimuli are shown simultaneously. Followed by the first delay
  period, a cue is shown, indicating which sample stimulus will be tested.
  Then the first test stimulus is shown and the agent needs to report whether
  this test stimulus matches the cued sample stimulus. Then another delay
  and then test period follows, and the agent needs to report whether the
  other sample stimulus matches the second test stimulus.
  """
  metadata = {
    'paper_link': 'https://science.sciencemag.org/content/354/6316/1136',
    'paper_name': 'Reactivation of latent working memories with transcranial magnetic stimulation',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 500.,
      t_sample: Union[int, float] = 500.,
      t_delay1: Union[int, float] = 500.,
      t_cue1: Union[int, float] = 500.,
      t_test1: Union[int, float] = 500.,
      t_delay2: Union[int, float] = 500.,
      t_cue2: Union[int, float] = 500.,
      t_test2: Union[int, float] = 500.,
      num_trial: int = 1024,
      noise_sigma: float = 1.0,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_sample = bp.check.is_float(t_sample, min_bound=0., allow_int=True)
    self.t_delay1 = bp.check.is_float(t_delay1, min_bound=0., allow_int=True)
    self.t_cue1 = bp.check.is_float(t_cue1, min_bound=0., allow_int=True)
    self.t_test1 = bp.check.is_float(t_test1, min_bound=0., allow_int=True)
    self.t_delay2 = bp.check.is_float(t_delay2, min_bound=0., allow_int=True)
    self.t_cue2 = bp.check.is_float(t_cue2, min_bound=0., allow_int=True)
    self.t_test2 = bp.check.is_float(t_test2, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_sample = int(self.t_sample / self.dt)
    n_delay1 = int(self.t_delay1 / self.dt)
    n_cue1 = int(self.t_cue1 / self.dt)
    n_test1 = int(self.t_test1 / self.dt)
    n_delay2 = int(self.t_delay2 / self.dt)
    n_cue2 = int(self.t_cue2 / self.dt)
    n_test2 = int(self.t_test2 / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'sample': n_sample,
                          'delay1': n_delay1, 'cue1': n_cue1, 'test1': n_test1,
                          'delay2': n_delay2, 'n_cue2': n_cue2, 'n_test2': n_test2}

    # features
    self.choices = np.asarray([1, 2])
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    self._feature_periods = {'fixation': 1, 'stimulus1': 2, 'stimulus2': 2, 'cue1': 1, 'cue2': 1}

    # input / output information
    self.output_features = ['fixation', 'match', 'non-match']
    self.input_features = ['fixation',
                           'stimulus1-0', 'stimulus1-1',
                           'stimulus2-0', 'stimulus2-1',
                           'cue1', 'cue2']

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth1 = self.rng.choice(self.choices)
    ground_truth2 = self.rng.choice(self.choices)
    sample1 = self.rng.choice([0, 0.5])
    sample2 = self.rng.choice([0, 0.5])
    test_order = self.rng.choice([0, 1])

    test1 = sample1 if ground_truth1 == 1 else 0.5 - sample1
    test2 = sample2 if ground_truth2 == 1 else 0.5 - sample2

    if test_order == 0:
      stim_test1_period, stim_test2_period = 'test1', 'test2'
      cue1_period, cue2_period = 'cue1', 'cue2'
    else:
      stim_test1_period, stim_test2_period = 'test2', 'test1'
      cue1_period, cue2_period = 'cue2', 'cue1'

    sample_theta, test_theta = sample1 * np.pi, test1 * np.pi
    stim_sample1 = [np.cos(sample_theta), np.sin(sample_theta)]
    stim_test1 = [np.cos(test_theta), np.sin(test_theta)]

    sample_theta, test_theta = sample2 * np.pi, test2 * np.pi
    stim_sample2 = [np.cos(sample_theta), np.sin(sample_theta)]
    stim_test2 = [np.cos(test_theta), np.sin(test_theta)]

    ax0_sample = interval_of('sample', self._time_periods)
    ax0_test1 = interval_of('test1', self._time_periods)
    ax0_test2 = interval_of('test2', self._time_periods)
    ax1_fixation = interval_of('fixation', self._feature_periods)
    ax1_stim1 = interval_of('stimulus1', self._feature_periods)
    ax1_stim2 = interval_of('stimulus2', self._feature_periods)
    ax1_cue1 = interval_of('cue1', self._feature_periods)
    ax1_cue2 = interval_of('cue2', self._feature_periods)

    X[:, ax1_fixation] += 1.
    X[ax0_sample, ax1_stim1] += stim_sample1
    X[ax0_sample, ax1_stim2] += stim_sample2
    ax0_cue1 = interval_of(cue1_period, self._time_periods)
    ax0_cue2 = interval_of(cue2_period, self._time_periods)
    X[ax0_cue1, ax1_cue1] += 1.
    X[ax0_cue2, ax1_cue2] += 1.

    ax0_stim_test1 = interval_of(stim_test1_period, self._time_periods)
    ax0_stim_test2 = interval_of(stim_test2_period, self._time_periods)
    X[ax0_stim_test1, ax1_stim1] += stim_test1
    X[ax0_stim_test2, ax1_stim2] += stim_test2

    X[ax0_sample] += self.rng.randn(self._time_periods['sample']) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_test1] += self.rng.randn(self._time_periods['test1']) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_test2] += self.rng.randn(self._time_periods['test2']) * self.noise_sigma / np.sqrt(self.dt)

    Y[ax0_stim_test1] = ground_truth1
    Y[ax0_stim_test2] = ground_truth2

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [('fixation', 1), ('stimulus1', 2), ('stimulus2', 2), ('cue', 2)]
    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]




class GoNoGo(FixedLenCogTask):
  r"""Delayed match-to-sample task.

  A sample stimulus is shown during the sample period. The stimulus is
  characterized by a one-dimensional variable, such as its orientation
  between 0 and 360 degree. After a delay period, a test stimulus is
  shown. The agent needs to determine whether the sample and the test
  stimuli are equal, and report that decision during the decision period.
  """

  metadata = {
    'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf',
    'paper_name': 'Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 0.,
      t_stimulus: Union[int, float] = 500.,
      t_delay: Union[int, float] = 500.,
      t_decision: Union[int, float] = 900.,
      num_trial: int = 1024,
      num_choice: int = 2,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # Inputs

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_stimulus = bp.check.is_float(t_stimulus, min_bound=0., allow_int=True)
    self.t_delay = bp.check.is_float(t_delay, min_bound=0., allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_sample = int(self.t_stimulus / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'stimulus': n_sample,
                          'delay': n_delay,
                          'decision': n_decision}

    # features
    self.num_choice = bp.check.is_integer(num_choice)
    self._choices = np.asarray([0, 1])

    # input / output information
    self.output_features = ['fixation', 'go']
    self.input_features = ['fixation', 'nogo', 'go']

  def sample_trial(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)

    ax0_decision = interval_of('decision', self._time_periods)
    ax0_stim = interval_of('stimulus', self._time_periods)

    X[:, 0] += 1.
    X[ax0_stim, ground_truth + 1] += 1.
    X[ax0_decision] = 0.

    Y[ax0_decision] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(self._time_periods.items())
    ax1 = [('fixation', 1), ('nogo', 1), ('go', 1)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


class IntervalDiscrimination(VariedLenCogTask):
  r"""Comparing the time length of two stimuli.

  Two stimuli are shown sequentially, separated by a delay period. The
  duration of each stimulus is randomly sampled on each trial. The
  subject needs to judge which stimulus has a longer duration, and reports
  its decision during the decision period by choosing one of the two
  choice options.
  """
  metadata = {
    'paper_link': 'https://www.sciencedirect.com/science/article/pii/S0896627309004887',
    'paper_name': 'Feature- and Order-Based Timing Representations in the Frontal Cortex',
  }

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float, Callable] = 300,
      t_stim1: Union[int, float, Callable] = None,
      t_delay1: Union[int, float, Callable] = None,
      t_stim2: Union[int, float, Callable] = None,
      t_delay2: Union[int, float, Callable] = 500.,
      t_decision: Union[int, float, Callable] = 300.,
      num_trial: int = 1024,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     max_seq_len=max_seq_len,
                     num_trial=num_trial,
                     seed=seed)
    if t_stim1 is None:
      t_stim1 = lambda: self.rng.uniform(300, 600)
    if t_delay1 is None:
      t_delay1 = lambda: self.rng.uniform(800, 1500)
    if t_stim2 is None:
      t_stim2 = lambda: self.rng.uniform(300, 600)

    # time related
    assert isinstance(t_fixation, (int, float, Callable))
    assert isinstance(t_stim1, (int, float, Callable))
    assert isinstance(t_delay1, (int, float, Callable))
    assert isinstance(t_stim2, (int, float, Callable))
    assert isinstance(t_delay2, (int, float, Callable))
    assert isinstance(t_decision, (int, float, Callable))
    self.t_fixation = t_fixation
    self.t_stim1 = t_stim1
    self.t_delay1 = t_delay1
    self.t_stim2 = t_stim2
    self.t_delay2 = t_delay2
    self.t_decision = t_decision

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation', 'stimulus 0', 'stimulus 1']

  def sample_trial(self, item):
    t_fixation = initialize(self.t_fixation)
    t_stim1 = initialize(self.t_stim1)
    t_delay1 = initialize(self.t_delay1)
    t_stim2 = initialize(self.t_stim2)
    t_delay2 = initialize(self.t_delay2)
    t_decision = initialize(self.t_decision)
    time_info = {'fixation': t_fixation,
                 'stim1': t_stim1,
                 'delay1': t_delay1,
                 'stim2': t_stim2,
                 'delay2': t_delay2,
                 'decision': t_decision}
    ground_truth = 1 if t_stim1 > t_stim2 else 2

    n_total = sum(time_info.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros((n_total,), dtype=int)

    ax1_fixation = 0
    ax1_stim1 = 1
    ax1_stim2 = 2
    ax0_stim1 = interval_of('stim1', time_info)
    ax0_stim2 = interval_of('stim2', time_info)
    ax0_decision = interval_of('decision', time_info)

    X[:, ax1_fixation] += 1.
    X[ax0_stim1, ax1_stim1] += 1.
    X[ax0_stim2, ax1_stim2] += 1.
    X[ax0_decision] = 0.

    Y[ax0_decision] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(time_info.items())
    ax1 = [('fixation', 1), ('stimulus', 2)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]




class PostDecisionWager(VariedLenCogTask):
  r"""Post-decision wagering task assessing confidence.

  The agent first performs a perceptual discrimination task (see for more
  details the PerceptualDecisionMaking task). On a random half of the
  trials, the agent is given the option to abort the sensory
  discrimination and to choose instead a sure-bet option that guarantees a
  small reward. Therefore, the agent is encouraged to choose the sure-bet
  option when it is uncertain about its perceptual decision.
  """
  metadata = {
    'paper_link': 'https://science.sciencemag.org/content/324/5928/759.long',
    'paper_name': 'Representation of Confidence Associated with a Decision by Neurons in the Parietal Cortex',
  }

  cohs = np.asarray([0, 3.2, 6.4, 12.8, 25.6, 51.2])

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 80.,
      t_fixation: Union[int, float, Callable] = 300,
      t_stimulus: Union[int, float, Callable] = TruncExp(180, 100, 900),
      t_delay: Union[int, float, Callable] = TruncExp(1350, 1200, 1800),
      t_pre_sure: Union[int, float, Callable] = None,
      t_decision: Union[int, float, Callable] = 300.,
      num_trial: int = 1024,
      num_choice: int = 2,
      seed: Optional[int] = None,
      noise_sigma: float = 1.0,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     max_seq_len=max_seq_len,
                     num_trial=num_trial,
                     seed=seed)

    if t_pre_sure is None:
      t_pre_sure = lambda: self.rng.uniform(500, 750)
    assert isinstance(t_fixation, (int, float, Callable))
    assert isinstance(t_stimulus, (int, float, Callable))
    assert isinstance(t_delay, (int, float, Callable))
    assert isinstance(t_pre_sure, (int, float, Callable))
    assert isinstance(t_decision, (int, float, Callable))
    self.t_fixation = t_fixation
    self.t_stimulus = t_stimulus
    self.t_delay = t_delay
    self.t_pre_sure = t_pre_sure
    self.t_decision = t_decision

    # inputs
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    self.num_choice = bp.check.is_integer(num_choice)
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._choices = np.arange(num_choice)

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1', 'sure']
    self.input_features = ['fixation', 'stimulus 0', 'stimulus 1', 'sure']

  def sample_trial(self, item):
    t_fixation = initialize(self.t_fixation)
    t_stimulus = initialize(self.t_stimulus)
    t_delay = initialize(self.t_delay)
    t_pre_sure = initialize(self.t_pre_sure)
    t_decision = initialize(self.t_decision)

    wager = self.rng.random() > 0.5
    if wager:
      time_info = {'fixation': t_fixation,
                   'stimulus': t_stimulus,
                   'pre_sure': t_pre_sure,
                   'delay': t_delay,
                   'decision': t_decision}
      ax0_pre_sure = interval_of('pre_sure', time_info)
    else:
      time_info = {'fixation': t_fixation,
                   'stimulus': t_stimulus,
                   'delay': t_delay,
                   'decision': t_decision}
    n_total = sum(time_info.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros((n_total,), dtype=int)

    ground_truth = self.rng.choice(self._choices)
    coh = self.rng.choice(self.cohs)
    stim_theta = self._features[ground_truth]

    ax0_fixation = interval_of('fixation', time_info)
    ax0_stimulus = interval_of('stimulus', time_info)
    ax0_delay = interval_of('delay', time_info)
    ax0_decision = interval_of('decision', time_info)
    ax1_fixation = 0
    ax1_stimulus = slice(1, 3)
    ax1_sure = 3

    X[ax0_fixation, ax1_fixation] += 1.
    X[ax0_stimulus, ax1_fixation] += 1.
    X[ax0_delay, ax1_fixation] += 1.

    stim = np.cos(self._features - stim_theta) * (coh / 200) + 0.5
    X[ax0_stimulus, ax1_stimulus] += stim
    X[ax0_stimulus] += self.rng.randn((t_stimulus, len(self.input_features))) * self.noise_sigma / np.sqrt(self.dt)
    if wager:
      X[ax0_delay, ax1_sure] += 1.
      X[ax0_pre_sure, ax1_sure] = 0.

    Y[ax0_decision] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(time_info.items())
    ax1 = [('fixation', 1), ('stimulus', self.num_choice), ('sure', 1)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


class ReadySetGo(VariedLenCogTask):
  r"""Agents have to measure and produce different time intervals.

  A stimulus is briefly shown during a ready period, then again during a
  set period. The ready and set periods are separated by a measure period,
  the duration of which is randomly sampled on each trial. The agent is
  required to produce a response after the set cue such that the interval
  between the response and the set cue is as close as possible to the
  duration of the measure period.

  Args:
      gain: Controls the measure that the agent has to produce. (def: 1, int)
      prod_margin: controls the interval around the ground truth production
          time within which the agent receives proportional reward
  """
  metadata = {
    'paper_link': 'https://www.sciencedirect.com/science/article/pii/S0896627318304185',
    'paper_name': 'Flexible Sensorimotor Computations through Rapid Reconfiguration of Cortical Dynamics',
  }

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 80.,
      t_fixation: Union[int, float, Callable] = 100.,
      t_ready: Union[int, float, Callable] = 80.,
      t_measure: Union[int, float, Callable] = None,
      t_set: Union[int, float, Callable] = 80,
      num_trial: int = 1024,
      gain=1,
      prod_margin=0.2,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     max_seq_len=max_seq_len,
                     num_trial=num_trial,
                     seed=seed)

    self.gain = gain
    self.prod_margin = prod_margin

    # time
    assert isinstance(t_fixation, (int, float, Callable))
    assert isinstance(t_ready, (int, float, Callable))
    assert isinstance(t_measure, (int, float, Callable))
    assert isinstance(t_set, (int, float, Callable))
    self.t_fixation = t_fixation
    self.t_reach = t_ready
    self.t_measure = t_measure
    self.t_set = t_set

    # input / output information
    self.output_features = ['fixation', 'go']
    self.input_features = ['fixation', 'ready', 'set']

  def sample_trial(self, item):
    t_fixation = int(initialize(self.t_fixation) / self.dt)
    t_reach = int(initialize(self.t_reach) / self.dt)
    t_measure = int(initialize(self.t_measure) / self.dt)
    t_set = int(initialize(self.t_set) / self.dt)
    t_production = int(t_measure * self.gain * 2)
    time_info = {'fixation': t_fixation,
                 'measure': t_measure,
                 'reach': t_reach,
                 'set': t_set,
                 'production': t_production}

    n_total = sum(time_info.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros((n_total,), dtype=int)

    ax1_fixation = 0
    ax1_ready = 1
    ax1_set = 2
    ax0_production = interval_of('production', time_info)
    ax0_ready = interval_of('ready', time_info)
    ax0_set = interval_of('set', time_info)

    X[:, ax1_fixation] += 1.
    X[ax0_production, ax1_fixation] = 0.
    X[ax0_ready, ax1_ready] += 1.
    X[ax0_set, ax1_set] += 1.

    gt = np.zeros(t_production)
    gt[int(t_measure * self.gain)] = 1
    Y[ax0_production] = gt

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(time_info.items())
    ax1 = [('fixation', 1), ('ready', 1), ('set', 1)]

    return [X, dict(ax0=ax0, ax1=ax1)], [Y, dict(ax0=ax0)]


