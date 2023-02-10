from typing import Union, Optional, Callable

import numpy as np

import brainpy as bp
from brainpy_datasets._src.cognitive.base import FixedLenCogTask
from brainpy_datasets._src.cognitive.utils import interval_of

__all__ = [
  'AntiReach',
  'Reaching1D',
]


class AntiReach(FixedLenCogTask):
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
      t_fixation: Union[int, float] = 500.,
      t_stimulus: Union[int, float] = 500.,
      t_delay: Union[int, float] = 0.,
      t_decision: Union[int, float] = 500.,
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
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_stimulus = bp.check.is_float(t_stimulus, min_bound=0., allow_int=True)
    self.t_delay = bp.check.is_float(t_delay, min_bound=0., allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_stimulus = int(self.t_stimulus / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = {'fixation': n_fixation,
                          'stimulus': n_stimulus,
                          'delay': n_delay,
                          'decision': n_decision, }

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

  def __getitem__(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, self.num_choice + 1))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.choice(self._choices)
    if self.anti:
      stim_theta = np.mod(self._features[ground_truth] + np.pi, 2 * np.pi)
    else:
      stim_theta = self._features[ground_truth]

    ax0_fixation = interval_of('fixation', self._time_periods)
    ax0_stimulus = interval_of('stimulus', self._time_periods)
    ax0_delay = interval_of('delay', self._time_periods)
    ax1_fixation = interval_of('fixation', self._feature_periods)
    ax1_choice = interval_of('choice', self._feature_periods)

    X[ax0_fixation, ax1_fixation] += 1.
    X[ax0_stimulus, ax1_fixation] += 1.
    X[ax0_delay, ax1_fixation] += 1.

    stim = np.cos(self._features - stim_theta)
    X[ax0_stimulus, ax1_choice] += stim

    Y[interval_of('decision', self._time_periods)] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y


class Reaching1D(FixedLenCogTask):
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
      t_fixation: Union[int, float] = 500.,
      t_reach: Union[int, float] = 500.,
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
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0., allow_int=True)
    self.t_reach = bp.check.is_float(t_reach, min_bound=0., allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_reach = int(self.t_reach / self.dt)
    self._time_periods = {'fixation': n_fixation, 'reach': n_reach, }

    # features
    self.num_choice = bp.check.is_integer(num_choice)
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._feature_periods = {'target': num_choice, 'self': num_choice}

    # input / output information
    self.output_features = ['fixation', 'left', 'right']
    self.input_features = [f'target{i}' for i in range(num_choice)] + [f'self{i}' for i in range(num_choice)]

  def __getitem__(self, item):
    n_total = sum(self._time_periods.values())
    X = np.zeros((n_total, len(self.input_features)))
    Y = np.zeros(n_total, dtype=int)

    ground_truth = self.rng.uniform(0, np.pi * 2)

    ax0_fixation = interval_of('fixation', self._time_periods)
    ax1_target = interval_of('target', self._feature_periods)
    ax0_reach = interval_of('reach', self._time_periods)

    target = np.cos(self._features - ground_truth)
    X[ax0_reach, ax1_target] += target

    Y[ax0_fixation] = np.pi
    Y[ax0_reach] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    return X, Y

