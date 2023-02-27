
# rate tasks
from brainpy_datasets._src.cognitive.base import (
  CognitiveTask as CognitiveTask,
  TaskLoader as TaskLoader,
  Feature as Feature,
  CircleFeature as CircleFeature,
)
from brainpy_datasets._src.cognitive.decision_making import (
  RateSingleContextDecisionMaking,
  RateContextDecisionMaking,
  RatePerceptualDecisionMaking,
  RatePulseDecisionMaking,
  RatePerceptualDecisionMakingDelayResponse,
)
from brainpy_datasets._src.cognitive.others import (
  RateAntiReach,
  RateReaching1D,
)
from brainpy_datasets._src.cognitive.reasoning import (
  RateHierarchicalReasoning,
  RateProbabilisticReasoning,
)
from brainpy_datasets._src.cognitive.working_memory import (
  RateDelayComparison,
  RateDelayMatchCategory,
  RateDelayMatchSample,
  RateDelayPairedAssociation,
  RateDualDelayMatchSample,
  RateGoNoGo,
  RateIntervalDiscrimination,
  RatePostDecisionWager,
  RateReadySetGo,
)


# rate/spiking tasks
from brainpy_datasets._src.cognitive.working_memory import (
  DelayComparison,
  CircleFeatDMS,
  CircleFeatDMS_recall,
)
from brainpy_datasets._src.cognitive.others import (
  EvidenceAccumulation,
)


from brainpy_datasets._src.cognitive._utils import firing_rate

