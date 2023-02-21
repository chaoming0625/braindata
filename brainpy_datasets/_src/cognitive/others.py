from typing import Union, Optional, Callable

import numpy as np

import brainpy as bp
from brainpy_datasets._src.cognitive.base import (CognitiveTask,
                                                  TimeDuration,
                                                  Feature,
                                                  is_time_duration)
from brainpy_datasets._src.cognitive._utils import interval_of, period_to_arr
from brainpy_datasets._src.utils.others import initialize

__all__ = [
  'RateAntiReach',
  'RateReaching1D',
  'EvidenceAccumulation',
]


class RateAntiReach(CognitiveTask):
  """Anti-response task.

  During the fixation period, the agent fixates on a fixation point.
  During the following stimulus period, the agent is then shown a stimulus away
  from the fixation point. Finally, the agent needs to respond in the
  opposite direction of the stimulus during the decision period.

  Args:
    anti: bool, if True, requires an anti-response. If False, requires a
        pro-response, i.e. response towards the stimulus.
  """
  metadata = {
    'paper_link': 'https://www.nature.com/articles/nrn1345',
    'paper_name': 'Look away: the anti-saccade task and the voluntary control of eye movement',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      anti: bool = True,
      t_fixation: TimeDuration = 500.,
      t_stimulus: TimeDuration = 500.,
      t_delay: TimeDuration = 0.,
      t_decision: TimeDuration = 500.,
      num_choice: int = 32,
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
    self.t_fixation = is_time_duration(t_fixation)
    self.t_stimulus = is_time_duration(t_stimulus)
    self.t_delay = is_time_duration(t_delay)
    self.t_decision = is_time_duration(t_decision)

    # features
    self.num_choice = bp.check.is_integer(num_choice, )
    self._features = np.arange(0, 2 * np.pi, 2 * np.pi / num_choice)
    self._choices = np.arange(self.num_choice)
    self._feature_periods = {'fixation': 1, 'choice': num_choice}

    # others
    self.anti = anti

    # input / output information
    self.output_features = ['fixation'] + [f'choice {i}' for i in range(num_choice)]
    self.input_features = ['fixation'] + [f'stimulus {i}' for i in range(num_choice)]

  @property
  def num_inputs(self) -> int:
    return 1 + self.num_choice

  @property
  def num_outputs(self) -> int:
    return 1 + self.num_choice

  def sample_a_trial(self, item):
    n_fixation = int(initialize(self.t_fixation) / self.dt)
    n_stimulus = int(initialize(self.t_stimulus) / self.dt)
    n_delay = int(initialize(self.t_delay) / self.dt)
    n_decision = int(initialize(self.t_decision) / self.dt)
    _time_periods = {'fixation': n_fixation,
                     'stimulus': n_stimulus,
                     'delay': n_delay,
                     'decision': n_decision, }
    n_total = sum(_time_periods.values())
    X = np.zeros((n_total, self.num_choice + 1))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)
    if self.anti:
      stim_theta = np.mod(self._features[ground_truth] + np.pi, 2 * np.pi)
    else:
      stim_theta = self._features[ground_truth]

    ax0_fixation = interval_of('fixation', _time_periods)
    ax0_stimulus = interval_of('stimulus', _time_periods)
    ax0_delay = interval_of('delay', _time_periods)
    ax1_fixation = interval_of('fixation', self._feature_periods)
    ax1_choice = interval_of('choice', self._feature_periods)

    X[ax0_fixation, ax1_fixation] += 1.
    X[ax0_stimulus, ax1_fixation] += 1.
    X[ax0_delay, ax1_fixation] += 1.

    stim = np.cos(self._features - stim_theta)
    X[ax0_stimulus, ax1_choice] += stim

    Y[interval_of('decision', _time_periods)] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y, period_to_arr(_time_periods)


class RateReaching1D(CognitiveTask):
  r"""Reaching to the stimulus.

  The agent is shown a stimulus during the fixation period. The stimulus
  encodes a one-dimensional variable such as a movement direction. At the
  end of the fixation period, the agent needs to respond by reaching
  towards the stimulus direction.
  """
  metadata = {
    'paper_link': 'https://science.sciencemag.org/content/233/4771/1416',
    'paper_name': 'Neuronal population coding of movement direction',
  }

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: TimeDuration = 500.,
      t_reach: TimeDuration = 500.,
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

    # time
    self.t_fixation = is_time_duration(t_fixation)
    self.t_reach = is_time_duration(t_reach)

    # features
    self.num_choice = bp.check.is_integer(num_choice)
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._feature_periods = {'target': num_choice, 'self': num_choice}

    # input / output information
    self.output_features = ['fixation', 'left', 'right']
    self.input_features = [f'target{i}' for i in range(num_choice)] + [f'self{i}' for i in range(num_choice)]

  @property
  def num_inputs(self) -> int:
    return 1 + self.num_choice

  @property
  def num_outputs(self) -> int:
    return len(self.output_features)

  def sample_a_trial(self, item):
    n_fixation = int(self.t_fixation / self.dt)
    n_reach = int(self.t_reach / self.dt)
    _time_periods = {'fixation': n_fixation, 'reach': n_reach, }
    n_total = sum(_time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.uniform(0, np.pi * 2)

    ax0_fixation = interval_of('fixation', _time_periods)
    ax0_reach = interval_of('reach', _time_periods)
    ax1_target = interval_of('target', self._feature_periods)

    target = np.cos(self._features - ground_truth)
    X[ax0_reach, ax1_target] += target

    Y[ax0_fixation] = np.pi
    Y[ax0_reach] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y, period_to_arr(_time_periods)


class EvidenceAccumulation(CognitiveTask):
  metadata = {
    'paper_link': 'https://doi.org/10.1038/nn.4403',
    'paper_name': 'History-dependent variability in population dynamics during evidence accumulation in cortex',
  }

  def __init__(
      self,
      # time
      dt: Union[int, float] = 1.,
      t_interval: TimeDuration = 50.,
      t_cue: TimeDuration = 100.,
      t_delay: TimeDuration = 1000.,
      t_recall: TimeDuration = 150.,
      # features
      ft_left: Feature = Feature(1, 25, 40.),
      ft_right: Feature = Feature(1, 25, 40.),
      ft_recall: Feature = Feature(1, 25, 40.),
      ft_noise: Feature = Feature(1, 25, 10.),
      prob: float = 0.3,
      num_trial: int = 1024,
      num_cue: int = 7,
      seed: Optional[int] = None,
      mode: str = 'spiking',
      input_transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
  ):
    super().__init__(input_transform=input_transform,
                     target_transform=target_transform,
                     dt=dt,
                     num_trial=num_trial,
                     seed=seed)

    # time
    self.t_interval = is_time_duration(t_interval)
    self.t_cue = is_time_duration(t_cue)
    self.t_delay = is_time_duration(t_delay)
    self.t_recall = is_time_duration(t_recall)

    # feature
    ft_left.name = 'left'
    ft_right.name = 'right'
    ft_noise.name = 'noise'
    ft_recall.name = 'recall'
    self.features = ft_left + ft_right + ft_recall + ft_noise
    self.features.set_mode(mode)

    # features
    self.prob = bp.check.is_float(prob)
    self.num_cue = bp.check.is_integer(num_cue)

    # input / output information
    periods = []
    for i in range(self.num_cue):
      periods.append(f'interval {i}')
      periods.append(f'cue {i}')
    periods += ['delay', 'recall']
    self.periods = periods
    self.output_features = ['left', 'right']

  @property
  def num_inputs(self) -> int:
    return self.features.num

  @property
  def num_outputs(self) -> int:
    return len(self.output_features)

  def sample_a_trial(self, item):
    t_interval = int(initialize(self.t_interval) / self.dt)
    t_cue = int(initialize(self.t_cue) / self.dt)
    t_delay = int(initialize(self.t_delay) / self.dt)
    t_recall = int(initialize(self.t_recall) / self.dt)
    _time_periods = dict()
    for i in range(self.num_cue):
      _time_periods[f'interval {i}'] = t_interval
      _time_periods[f'cue {i}'] = t_cue
    _time_periods['delay'] = t_delay
    _time_periods['recall'] = t_recall
    t_total = sum(_time_periods.values())
    X = np.zeros((t_total, self.features.num))

    # assign input spike probability
    ground_truth = self.rng.rand() < 0.5
    prob = self.prob if ground_truth else (1 - self.prob)

    # for each example in batch, draw which cues are going to be active (left or right)
    cue_assignments = np.asarray(self.rng.random(self.num_cue) > prob, dtype=np.int_)

    # generate input spikes
    for k in range(self.num_cue):
      # input channels only fire when they are selected (left or right)
      choice = 'left' if cue_assignments[k] else 'right'
      # reverse order of cues
      i_seq = t_interval + k * (t_interval + t_cue)
      X[i_seq:i_seq + t_cue, self.features[choice].i] = self.features[choice].fr(self.dt)

    # recall cue
    X[-t_recall:, self.features['recall'].i] = self.features['recall'].fr(self.dt)

    # background noise
    X[:, self.features['noise'].i] = self.features['noise'].fr(self.dt)

    # generate inputs and targets
    if self.features.is_spiking_mode:
      X = self.rng.rand(*X.shape) < X
    Y = np.asarray(np.sum(cue_assignments) > (self.num_cue / 2), dtype=int)

    # transform
    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    dim0 = period_to_arr(_time_periods)
    return X, Y, dim0

