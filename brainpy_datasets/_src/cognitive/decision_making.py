from typing import Union, Callable, Optional, Tuple

import numpy as np

import brainpy as bp
from brainpy_datasets._src.cognitive.base import VariedLenCogTask, FixedLenCogTask
from brainpy_datasets._src.cognitive.utils import interval_of
from brainpy_datasets._src.utils.random import TruncExp
from brainpy_datasets._src.utils.others import initialize

__all__ = [
  'SingleContextDecisionMaking',
  'ContextDecisionMaking',
  'PerceptualDecisionMaking',
  'PulseDecisionMaking',
  'PerceptualDecisionMakingDelayResponse',
]


class SingleContextDecisionMaking(VariedLenCogTask):
  """Context-dependent decision-making task.

  The agent simultaneously receives stimulus inputs from two modalities (
  for example, a colored random dot motion pattern with color and motion
  modalities). The agent needs to make a perceptual decision based on only
  one of the two modalities, while ignoring the other. The agent reports
  its decision during the decision period, with an optional delay period
  in between the stimulus period and the decision period. The relevant
  modality is not explicitly signaled.

  """

  metadata = {
    'paper_link': 'https://www.nature.com/articles/nature12742',
    'paper_name': 'Context-dependent computation by recurrent dynamics in prefrontal cortex',
  }

  coherence = np.asarray([5., 15., 50.])

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 300.,
      t_stimulus: Union[int, float] = 750.,
      t_delay: Union[int, float, Callable] = TruncExp(600, 300, 3000),
      t_decision: Union[int, float] = 100.,
      context: int = 0,
      num_choice: int = 2,
      noise_sigma: float = 1.0,
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

    # time related
    self.t_fixation = bp.check.is_float(t_fixation, allow_int=True)
    self.t_stimulus = bp.check.is_float(t_stimulus, allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, allow_int=True)
    self._n_fixation = int(self.t_fixation / self.dt)
    self._n_stimulus = int(self.t_stimulus / self.dt)
    self._n_decision = int(self.t_decision / self.dt)
    assert isinstance(t_delay, (int, float, Callable))
    self.t_delay = t_delay

    self.num_choice = bp.check.is_integer(num_choice, )
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)
    assert context in (0, 1)
    self.context = context

    # set action and observation space
    self._features = np.linspace(0, 2 * np.pi, num_choice + 1)[:-1]
    self._choices = np.arange(num_choice)

    # input / output information
    self.output_features = ['fixation'] + [f'choice {i}' for i in range(num_choice)]
    self.input_features = (
        ['fixation', ] +
        [f'stimulus 0-{i}' for i in range(num_choice)] +
        [f'stimulus 1-{i}' for i in range(num_choice)]
    )
    self._feature_info = {'fixation': 1, 'stimulus 0': num_choice, 'stimulus 1': num_choice}

  @property
  def num_input_feature(self):
    return 1 + self.num_choice * 2

  def sample_trial(self, item):
    t_delay = initialize(self.t_delay)
    _n_delay = int(t_delay / self.dt)
    n_total = self._n_fixation + self._n_stimulus + _n_delay + self._n_decision
    X = np.zeros((n_total, self.num_input_feature))
    Y = np.zeros((n_total,), dtype=int)

    _time_info = {'fixation': self._n_fixation,
                  'stimulus': self._n_stimulus,
                  'delay': _n_delay,
                  'decision': self._n_decision}

    choice_0 = ground_truth = self.rng.choice(self._choices)
    choice_1 = self.rng.choice(self._choices)
    coh_0 = self.rng.choice(self.coherence)
    coh_1 = self.rng.choice(self.coherence)
    if self.context == 1:
      choice_1, choice_0 = choice_0, choice_1
    stim_theta_0 = self._features[choice_0]
    stim_theta_1 = self._features[choice_1]

    X[:, 0] = 1.
    stim = np.cos(self._features - stim_theta_0) * (coh_0 / 200) + 0.5
    X[self._n_fixation: self._n_fixation + self._n_stimulus, 1:self.num_choice + 1] += stim
    stim = np.cos(self._features - stim_theta_1) * (coh_1 / 200) + 0.5
    X[self._n_fixation: self._n_fixation + self._n_stimulus, self.num_choice + 1:] += stim
    rand = self.rng.randn(self._n_stimulus, self.num_input_feature - 1) * self.noise_sigma / np.sqrt(self.dt)
    X[self._n_fixation: self._n_fixation + self._n_stimulus, 1:] += rand
    X[self._n_fixation + self._n_stimulus + _n_delay:] = 0.

    Y[self._n_fixation + self._n_stimulus + _n_delay:] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(_time_info.items())
    ax1 = tuple(self._feature_info.items())
    return [X, {'ax0': ax0, 'ax1': ax1}], [Y, {'ax0': ax0}]


class ContextDecisionMaking(VariedLenCogTask):
  """Context-dependent decision-making task.

  The agent simultaneously receives stimulus inputs from two modalities (
  for example, a colored random dot motion pattern with color and motion
  modalities). The agent needs to make a perceptual decision based on
  only one of the two modalities, while ignoring the other. The relevant
  modality is explicitly indicated by a rule signal.
  """
  metadata = {
    'paper_link': 'https://www.nature.com/articles/nature12742',
    'paper_name': 'Context-dependent computation by recurrent dynamics in prefrontal cortex',
  }

  coherence = np.array([5., 15., 50.])
  contexts = np.asarray([0, 1])  # index for context inputs
  choices = np.asarray([1, 2])  # left, right choice

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 300.,
      t_stimulus: Union[int, float] = 750.,
      t_delay: Union[int, float, Callable] = TruncExp(600, 300, 3000),
      t_decision: Union[int, float] = 100.,
      noise_sigma: float = 1.0,
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

    # time related
    self.t_fixation = bp.check.is_float(t_fixation, allow_int=True)
    self.t_stimulus = bp.check.is_float(t_stimulus, allow_int=True)
    self.t_decision = bp.check.is_float(t_decision, allow_int=True)
    self._n_fixation = int(self.t_fixation / self.dt)
    self._n_stimulus = int(self.t_stimulus / self.dt)
    self._n_decision = int(self.t_decision / self.dt)
    assert isinstance(t_delay, (int, float, Callable))
    self.t_delay = t_delay
    self.noise_sigma = bp.check.is_float(noise_sigma, min_bound=0., allow_int=True)

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation',
                           'stim1_mod1', 'stim2_mod1', 'stim1_mod2', 'stim2_mod2',
                           'context1', 'context2']

  @property
  def num_input_feature(self):
    return len(self.input_features)

  def sample_trial(self, item):
    if isinstance(self.t_delay, (int, float)):
      t_delay = self.t_delay
    elif isinstance(self.t_delay, Callable):
      t_delay = self.t_delay()
    else:
      raise ValueError
    _n_delay = int(t_delay / self.dt)
    n_total = self._n_fixation + self._n_stimulus + _n_delay + self._n_decision
    X = np.zeros((n_total, self.num_input_feature))
    Y = np.zeros((n_total,), dtype=int)

    time_info = {'fixation': self._n_fixation,
                    'stimulus': self._n_stimulus,
                    'delay': _n_delay,
                    'decision': self._n_decision}
    feature_info = {f: 1 for f in self.input_features}

    choice_0 = ground_truth = self.rng.choice(self.choices)
    choice_1 = self.rng.choice(self.choices)
    coh_0 = self.rng.choice(self.coherence)
    coh_1 = self.rng.choice(self.coherence)
    context = self.rng.choice(self.contexts)
    if context == 1:
      choice_1, choice_0 = choice_0, choice_1
    signed_coh_0 = coh_0 if choice_0 == 1 else -coh_0
    signed_coh_1 = coh_1 if choice_1 == 1 else -coh_1

    X[:, interval_of('fixation', feature_info)] += 1.

    ax0_stimulus = interval_of('stimulus', time_info)
    stim = (1 + signed_coh_0 / 100) / 2
    X[ax0_stimulus, interval_of('stim1_mod1', feature_info)] += stim

    stim = (1 - signed_coh_0 / 100) / 2
    X[ax0_stimulus, interval_of('stim2_mod1', feature_info)] += stim

    stim = (1 + signed_coh_1 / 100) / 2
    X[ax0_stimulus, interval_of('stim1_mod2', feature_info)] += stim

    stim = (1 - signed_coh_1 / 100) / 2
    X[ax0_stimulus, interval_of('stim2_mod2', feature_info)] += stim

    rand = self.rng.randn(self._n_stimulus, self.num_input_feature - 1) * self.noise_sigma / np.sqrt(self.dt)
    X[ax0_stimulus, 1:] += rand

    if context == 0:
      X[:, interval_of('context1', feature_info)] += 1.
    else:
      X[:, interval_of('context2', feature_info)] += 1.

    Y[interval_of('decision', time_info)] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    ax0 = tuple(time_info.items())
    ax1 = tuple(feature_info.items())

    return [X, {'ax0': ax0, 'ax1': ax1}], [Y, {'ax0': ax0}]



class PerceptualDecisionMaking(FixedLenCogTask):
  """"
  Two-alternative forced choice task in which the subject has to
  integrate two stimuli to decide which one is higher on average.

  A noisy stimulus is shown during the stimulus period. The strength (
  coherence) of the stimulus is randomly sampled every trial. Because the
  stimulus is noisy, the agent is encouraged to integrate the stimulus
  over time.

  Args:
      dt: Time step.
      t_fixation: Time of eye fixation.
      t_stimulus: Time of stimulus period.
      t_delay: Time of delay period.
      t_decision: Time of decision period.
      noise_sigma: float, input noise level.
      num_choice: int. Number of input choices.
      num_trial: int. The total number of trial in one epoch.
      seed: int. Random seed.
      input_transform:
      target_transform:
  """

  metadata = {
    'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
    'paper_name': 'The analysis of visual motion: a comparison of neuronal and psychophysical performance',
  }

  coherence = np.array([0, 6.4, 12.8, 25.6, 51.2])
  '''Coherence levels controlling the difficulty of the task.'''

  def __init__(
      self,
      dt: Union[int, float] = 100.,
      t_fixation: Union[int, float] = 100.,
      t_stimulus: Union[int, float] = 2000.,
      t_delay: Union[int, float] = 0.,
      t_decision: Union[int, float] = 100.,
      noise_sigma: float = 1.0,
      num_choice: int = 2,
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
    self.noise_sigma = bp.check.is_float(noise_sigma, allow_int=True)
    n_fixation = int(self.t_fixation / self.dt)
    n_stimulus = int(self.t_stimulus / self.dt)
    n_delay = int(self.t_delay / self.dt)
    n_decision = int(self.t_decision / self.dt)
    self._time_periods = dict()
    self._time_periods['fixation'] = n_fixation
    self._time_periods['stimulus'] = n_stimulus
    self._time_periods['delay'] = n_delay
    self._time_periods['decision'] = n_decision

    # features
    self.num_choice = bp.check.is_integer(num_choice, )
    self._feature_periods = {'fixation': 1, 'choice': num_choice}
    self._features = np.linspace(0, 2 * np.pi, self.num_choice + 1)[:-1]
    self._choices = np.arange(self.num_choice)

    # input / output information
    self.output_features = ['fixation'] + [f'choice {i}' for i in range(num_choice)]
    self.input_features = ['fixation'] + [f'choice {i}' for i in range(num_choice)]

  @property
  def num_input_feature(self):
    return 1 + self.num_choice

  def sample_trial(self, item):
    total = sum(self._time_periods.values())
    X = np.zeros((total, self.num_input_feature))
    Y = np.zeros((total,), dtype=np.int_)

    coherence = self.rng.choice(self.coherence)
    ground_truth = self.rng.choice(self._choices)
    feature = self._features[ground_truth]

    ax1_fixation = interval_of('fixation', self._feature_periods)
    ax0_fixation = interval_of('fixation', self._time_periods)
    delay = interval_of('delay', self._time_periods)
    ax1_choice = interval_of('choice', self._feature_periods)
    stimulus = interval_of('stimulus', self._time_periods)
    decision = interval_of('decision', self._time_periods)

    X[ax0_fixation, ax1_fixation] += 1.
    X[stimulus, ax1_fixation] += 1.
    X[delay, ax1_fixation] += 1.
    X[stimulus, ax1_choice] += np.cos(self._features - feature) * (coherence / 200) + 0.5
    noise = self.rng.randn(self._time_periods['stimulus'], self.num_choice) * self.noise_sigma / np.sqrt(self.dt)
    X[stimulus, ax1_choice] += noise

    Y[decision] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    dim0 = tuple(self._time_periods.items())
    dim1 = tuple(self._feature_periods.items())

    return [X, {'ax0': dim0, 'ax1': dim1}], [Y, {'ax0': dim0}]


class PulseDecisionMaking(VariedLenCogTask):
  """Pulse-based decision-making task.

  Discrete stimuli are presented briefly as pulses.

  Args:
      p_pulse: array-like, probability of pulses for each choice
      n_bin: int, number of stimulus bins
  """

  metadata = {
    'paper_link': 'https://elifesciences.org/articles/11308',
    'paper_name': 'Sources of noise during accumulation of evidence '
                  'in unrestrained and voluntarily head-restrained rats',
  }

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 10.,
      t_fixation: Union[int, float] = 500.,
      t_decision: Union[int, float] = 500.,
      t_cue: Union[int, float] = 10.,
      t_bin: Union[int, float] = 240.,
      num_trial: int = 1024,
      p_pulse: Tuple[float, float] = (0.3, 0.7),
      n_bin: int = 6,
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

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0.)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0.)
    self.t_cue = bp.check.is_float(t_cue, min_bound=0.)
    self.t_bin = bp.check.is_float(t_bin, min_bound=0.)
    self._n_fixation = int(self.t_fixation / self.dt)
    self._n_cue = int(self.t_cue / self.dt)
    self._n_bin = int(self.t_bin / self.dt)
    self._n_decision = int(self.t_decision / self.dt)
    self._n_total = self._n_fixation + (self._n_cue + self._n_bin) * n_bin + self._n_decision
    self._time_info = {'fixation': self._n_fixation, }
    for i in range(n_bin):
      self._time_info[f'cue{i}'] = self._n_cue
      self._time_info[f'bin{i}'] = self._n_bin
    self._time_info['decision'] = self._n_decision

    self.p_pulse = bp.check.is_sequence(p_pulse, elem_type=float)
    self.n_bin = bp.check.is_integer(n_bin, allow_none=False)

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation', 'stimulus 0', 'stimulus 1']

  def sample_trial(self, item):
    X = np.zeros((self._n_total, len(self.input_features)))
    Y = np.zeros(self._n_total, dtype=int)

    p1, p2 = self.p_pulse
    if self.rng.random() < 0.5:
      p1, p2 = p2, p1
    pulse1 = np.asarray(self.rng.random(self.n_bin) < p1, dtype=float)
    pulse2 = np.asarray(self.rng.random(self.n_bin) < p2, dtype=float)
    n_pulse1 = sum(pulse1)
    n_pulse2 = sum(pulse2) + self.rng.uniform(-0.1, 0.1)
    ground_truth = int(n_pulse1 < n_pulse2)

    X[:, 0] += 1.
    for i in range(self.n_bin):
      start = self._n_fixation + (self._n_cue + self._n_bin) * i
      end = start + self._n_cue
      X[start: end, 1] += pulse1[i]
      X[start: end, 2] += pulse2[i]
    X[self._n_total - self._n_decision:] = 0

    Y[self._n_total - self._n_decision:] = ground_truth + 1

    if self.input_transform is not None:
      X = self.input_transform(X)
    if self.target_transform is not None:
      Y = self.target_transform(Y)

    dim0 = tuple(self._time_info.items())
    dim1 = [(feat, 1) for feat in self.input_features]
    return [X, dict(ax0=dim0, ax1=dim1)], [Y, dict(ax0=dim0)]


class PerceptualDecisionMakingDelayResponse(VariedLenCogTask):
  """Perceptual decision-making with delayed responses.

  Agents have to integrate two stimuli and report which one is
  larger on average after a delay.

  """
  metadata = {
    'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
    'paper_name': 'Discrete attractor dynamics underlies persistent activity in the frontal cortex',
  }

  coherence = np.asarray([0, 6.4, 12.8, 25.6, 51.2])  # specifies the amount of evidence
  choices = np.asarray([1, 2])

  def __init__(
      self,
      max_seq_len: int,
      dt: Union[int, float] = 10.,
      t_fixation: Union[int, float] = 0.,
      t_stimulus: Union[int, float] = 1150.,
      t_delay: Union[int, float, Callable] = TruncExp(600, 300, 4000),
      t_decision: Union[int, float] = 1500.,
      num_trial: int = 1024,
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

    # time
    self.t_fixation = bp.check.is_float(t_fixation, min_bound=0.)
    self.t_decision = bp.check.is_float(t_decision, min_bound=0.)
    self.t_stimulus = bp.check.is_float(t_stimulus, min_bound=0.)
    assert isinstance(t_delay, (int, float, Callable))
    self.t_delay = t_delay
    self._n_fixation = int(self.t_fixation / self.dt)
    self._n_decision = int(self.t_decision / self.dt)
    self._n_stimulus = int(self.t_stimulus / self.dt)

    self.noise_sigma = bp.check.is_float(noise_sigma, allow_int=True)

    # input / output information
    self.output_features = ['fixation', 'choice 0', 'choice 1']
    self.input_features = ['fixation', 'stimulus 0', 'stimulus 1']

  @property
  def num_input_feature(self):
    return 3

  def sample_trial(self, item):
    if isinstance(self.t_delay, (int, float)):
      t_delay = self.t_delay
    elif isinstance(self.t_delay, Callable):
      t_delay = self.t_delay()
    else:
      raise ValueError
    _n_delay = int(t_delay / self.dt)
    n_total = self._n_fixation + self._n_stimulus + _n_delay + self._n_decision
    X = np.zeros((n_total, self.num_input_feature))
    Y = np.zeros((n_total,), dtype=int)

    time_periods = {'fixation': self._n_fixation,
                    'stimulus': self._n_stimulus,
                    'delay': _n_delay,
                    'decision': self._n_decision}

    ground_truth = self.rng.choice(self.choices)
    coherence = self.rng.choice(self.coherence)

    ax0_fixation = interval_of('fixation', time_periods)
    X[ax0_fixation] = [1, 0, 0]

    ax0_stim = interval_of('stimulus', time_periods)
    X[ax0_stim, 0] = 1
    X[ax0_stim, 1:] = (1 - coherence / 100) / 2
    X[ax0_stim, ground_truth] = (1 + coherence / 100) / 2
    X[ax0_stim, 1:] += self.rng.randn(self._n_stimulus, 2) * self.noise_sigma / np.sqrt(self.dt)

    ax0_delay = interval_of('delay', time_periods)
    X[ax0_delay] = [1, 0, 0]

    ax0_decision = interval_of('decision', time_periods)
    Y[ax0_decision] = ground_truth

    if self.input_transform is not None:
      X = self.input_transform(X)

    if self.target_transform is not None:
      Y = self.target_transform(Y)

    dim0 = tuple(time_periods.items())
    dim1 = [(feat, 1) for feat in self.input_features]
    return [X, dict(ax0=dim0, ax1=dim1)], [Y, dict(ax0=dim0)]


