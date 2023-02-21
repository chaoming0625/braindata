from typing import Union, Callable, Optional

import brainpy as bp
import numpy as np

from brainpy_datasets._src.cognitive.base import (CognitiveTask, TimeDuration)
from brainpy_datasets._src.cognitive._utils import interval_of, period_to_arr
from brainpy_datasets._src.utils.others import initialize
from brainpy_datasets._src.utils.random import TruncExp

__all__ = [
  'RateHierarchicalReasoning',
  'RateProbabilisticReasoning',
]


class RateHierarchicalReasoning(CognitiveTask):
  """Hierarchical reasoning of rules.

  On each trial, the subject receives two flashes separated by a delay
  period. The subject needs to judge whether the duration of this delay
  period is shorter than a threshold. Both flashes appear at the
  same location on each trial. For one trial type, the network should
  report its decision by going to the location of the flashes if the delay is
  shorter than the threshold. In another trial type, the network should go to
  the opposite direction of the flashes if the delay is short.

  The two types of trials are alternated across blocks, and the block
  transtion is unannouced.
  """
  metadata = {
    'paper_link': 'https://science.sciencemag.org/content/364/6441/eaav8911',
    'paper_name': "Hierarchical reasoning by neural circuits in the frontal cortex",
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: TimeDuration = TruncExp(600, 400, 800),
      t_rule_target: TimeDuration = 1000.,
      t_fixation2: TimeDuration = TruncExp(600, 400, 900),
      t_flash1: TimeDuration = 100,
      t_delay: TimeDuration = None,
      t_flash2: TimeDuration = 100.,
      t_decision: TimeDuration = 700.,
      noise_sigma: float = 1.0,
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
    if t_delay is None:
      t_delay = lambda: self.rng.choice([530, 610, 690, 770, 850, 930, 1010, 1090, 1170])

    # time related
    assert isinstance(t_fixation, (int, float, Callable))
    assert isinstance(t_rule_target, (int, float, Callable))
    assert isinstance(t_fixation2, (int, float, Callable))
    assert isinstance(t_flash1, (int, float, Callable))
    assert isinstance(t_delay, (int, float, Callable))
    assert isinstance(t_flash2, (int, float, Callable))
    assert isinstance(t_decision, (int, float, Callable))
    self.t_fixation = t_fixation
    self.t_rule_target = t_rule_target
    self.t_fixation2 = t_fixation2
    self.t_flash1 = t_flash1
    self.t_delay = t_delay
    self.t_flash2 = t_flash2
    self.t_decision = t_decision

    # other
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    self.choices = [0, 1]

    # input / output information
    self.output_features = ['fixation', 'rule 0', 'rule 1', 'choice 0', 'choice 1']
    self.input_features = ['fixation', 'rule 0', 'rule 1', 'stimulus 0', 'stimulus 1']
    self._feature_info = {'fixation': 1, 'rule': 2, 'choice': 2}

  @property
  def num_inputs(self) -> int:
    return len(self.input_features)

  @property
  def num_outputs(self) -> int:
    return len(self.output_features)

  def sample_a_trial(self, index):
    t_fixation = int(initialize(self.t_fixation) / self.dt)
    t_rule_target = int(initialize(self.t_rule_target) / self.dt)
    t_fixation2 = int(initialize(self.t_fixation2) / self.dt)
    t_flash1 = int(initialize(self.t_flash1) / self.dt)
    t_delay = int(initialize(self.t_delay) / self.dt)
    t_flash2 = int(initialize(self.t_flash2) / self.dt)
    t_decision = int(initialize(self.t_decision) / self.dt)
    time_info = {'fixation': t_fixation,
                 'rule_target': t_rule_target,
                 'fixation2': t_fixation2,
                 'flash1': t_flash1,
                 'delay': t_delay,
                 'flash2': t_flash2,
                 'decision': t_decision}
    n_total = sum(time_info.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros((n_total,), dtype=int)

    stimulus = self.rng.choice(self.choices)
    rule = index % 2

    # Is interval long? When interval == mid_delay, randomly assign
    long_interval = (t_delay > 610) + (self.rng.rand() - 0.5)
    # Is the response pro or anti?
    pro_choice = int(long_interval) == rule
    if pro_choice:
      choice = stimulus
    else:
      choice = 1 - stimulus

    ax1_stim = stimulus + 3
    ax1_fixation = 0
    ax1_rule = interval_of('rule', self._feature_info)
    ax0_decision = interval_of('decision', time_info)
    ax0_rule_target = interval_of('rule_target', time_info)
    ax0_flash1 = interval_of('flash1', time_info)
    ax0_flash2 = interval_of('flash2', time_info)

    X[:, ax1_fixation] += 1.
    X[ax0_decision, ax1_fixation] = 0.
    X[ax0_rule_target, ax1_rule] += 1.
    X[ax0_flash1, ax1_stim] += 1.
    X[ax0_flash2, ax1_stim] += 1.

    Y[ax0_decision] = choice + 3
    Y[ax0_rule_target] = rule + 1

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y, period_to_arr(time_info)


class RateProbabilisticReasoning(CognitiveTask):
  """Probabilistic reasoning.

  The agent is shown a sequence of stimuli. Each stimulus is associated
  with a certain log-likelihood of the correct response being one choice
  versus the other. The final log-likelihood of the target response being,
  for example, option 1, is the sum of all log-likelihood associated with
  the presented stimuli. A delay period separates each stimulus, so the
  agent is encouraged to lean the log-likelihood association and integrate
  these values over time within a trial.

  Args:
      shape_weight: array-like, evidence weight of each shape
      num_loc: int, number of location of show shapes
  """

  metadata = {
    'paper_link': 'https://www.nature.com/articles/nature05852',
    'paper_name': 'Probabilistic reasoning by neurons',
  }

  # The evidence weight of each stimulus
  shape_weight = np.asarray([-10, -0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9, 10])

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: TimeDuration = 500,
      t_stimulus: TimeDuration = 500,
      t_delay: TimeDuration = None,
      t_decision: TimeDuration = 700.,
      num_trial: int = 1024,
      num_loc: int = 4,
      seed: Optional[int] = None,
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)
    if t_delay is None:
      t_delay = lambda: self.rng.uniform(450, 550)

    # time related
    assert isinstance(t_fixation, (int, float, Callable))
    assert isinstance(t_stimulus, (int, float, Callable))
    assert isinstance(t_delay, (int, float, Callable))
    assert isinstance(t_decision, (int, float, Callable))
    self.t_fixation = t_fixation
    self.t_stimulus = t_stimulus
    self.t_delay = t_delay
    self.t_decision = t_decision

    # other
    self.choices = [0, 1]
    self.num_shape = len(self.shape_weight)
    dim_shape = self.num_shape
    # Shape representation needs to be fixed cross-platform
    self.shapes = np.eye(self.num_shape, dim_shape)
    self.num_loc = num_loc

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation', ] + [f'loc{i}-{j}' for j in range(dim_shape) for i in range(num_loc)]
    self._feature_info = {'fixation': 1}
    for i in range(num_loc):
      self._feature_info[f'loc{i}'] = dim_shape

  @property
  def num_inputs(self) -> int:
    return len(self.input_features)

  @property
  def num_outputs(self) -> int:
    return len(self.output_features)

  def sample_a_trial(self, index):
    t_fixation = int(initialize(self.t_fixation) / self.dt)
    t_stimulus = int(initialize(self.t_stimulus) / self.dt)
    t_delay = int(initialize(self.t_delay) / self.dt)
    t_decision = int(initialize(self.t_decision) / self.dt)
    time_info = {'fixation': t_fixation}
    for i in range(self.num_loc):
      time_info[f'stimulus{i}'] = t_stimulus
    time_info['delay'] = t_delay
    time_info['decision'] = t_decision
    n_total = sum(time_info.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros((n_total,), dtype=int)

    locs = self.rng.choice(range(self.num_loc), size=self.num_loc, replace=False)
    shapes = self.rng.choice(range(self.num_shape), size=self.num_loc, replace=True)

    log_odd = sum([self.shape_weight[shape] for shape in shapes])
    p = 1. / (10 ** (-log_odd) + 1.)
    ground_truth = int(self.rng.rand() < p)

    ax1_fixation = 0
    ax0_decision = interval_of('decision', time_info)

    X[:, ax1_fixation] += 1.
    X[ax0_decision, ax1_fixation] = 0.
    for i_loc in range(self.num_loc):
      loc = locs[i_loc]
      ax1_loc = interval_of('loc' + str(loc), self._feature_info)
      shape = shapes[i_loc]
      for j in range(i_loc, self.num_loc):
        ax0_stim = interval_of('stimulus' + str(j), time_info)
        X[ax0_stim, ax1_loc] += self.shapes[shape]

    Y[ax0_decision] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y, period_to_arr(time_info)
