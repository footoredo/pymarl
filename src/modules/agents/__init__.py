REGISTRY = {}

from .rnn_agent import RNNAgent
from .att_rnn_agent import ATTRNNAgent
from .att_rnn_agent_compatible import ATTRNNAgentCompatible
from .multi_att_rnn_agent import MULTIATTRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["att_rnn_compat"] = ATTRNNAgentCompatible
REGISTRY["multi_att_rnn"] = MULTIATTRNNAgent
