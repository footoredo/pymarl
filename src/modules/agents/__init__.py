REGISTRY = {}

from .rnn_agent import RNNAgent
from .att_rnn_agent import ATTRNNAgent
from .att_rnn_agent_compatible import ATTRNNAgentCompatible
REGISTRY["rnn"] = RNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["att_rnn_compat"] = ATTRNNAgentCompatible
