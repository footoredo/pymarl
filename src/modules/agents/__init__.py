REGISTRY = {}

from .rnn_agent import RNNAgent
from .att_rnn_agent import ATTRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent
