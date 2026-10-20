import torch
import torch.optim as optim
import abc

from collections import namedtuple
import os
import pickle

from .model import StepLRMargin, StepResetLR
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model

Transition = namedtuple(
    "Transition", ["s", "t", "a", "r", "s_", "t_", "done", "info"]
)


class DDQN(abc.ABC):
  """
  The parent class for DDQNSingle. Implements the
  basic utils functions and defines abstract functions to be implemented in
  the child class.
  """

  def __init__(self, CONFIG):
    """Initializes DDQN with a configuration file."""
    self.CONFIG = CONFIG
    self.saved = False
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)

    # == PARAM ==
    # Exploration-exploitation tradeoff.
    self.EpsilonScheduler = StepResetLR(
        initValue=CONFIG.EPSILON,
        period=CONFIG.EPS_PERIOD,
        resetPeriod=CONFIG.EPS_RESET_PERIOD,
        decay=CONFIG.EPS_DECAY,
        endValue=CONFIG.EPS_END,
    )
    self.EPSILON = self.EpsilonScheduler.get_variable()

    # Learning rate of updating the Q-network
    self.LR_C = CONFIG.LR_C
    self.LR_C_PERIOD = CONFIG.LR_C_PERIOD
    self.LR_C_DECAY = CONFIG.LR_C_DECAY
    self.LR_C_END = CONFIG.LR_C_END

    # Neural network related : batch size, maximal number of NNs stored
    self.BATCH_SIZE = CONFIG.BATCH_SIZE
    self.MAX_MODEL = CONFIG.MAX_MODEL
    self.device = CONFIG.DEVICE

    # Discount factor: anneal to one
    self.GammaScheduler = StepLRMargin(
        initValue=CONFIG.GAMMA,
        period=CONFIG.GAMMA_PERIOD,
        decay=CONFIG.GAMMA_DECAY,
        endValue=CONFIG.GAMMA_END,
        goalValue=1.0,
    )
    self.GAMMA = self.GammaScheduler.get_variable()

    # Target network update: also check `update_target_network()`
    self.double_network = CONFIG.DOUBLE  # bool: double DQN if True
    self.SOFT_UPDATE = CONFIG.SOFT_UPDATE  # bool, use soft_update if True
    self.TAU = CONFIG.TAU  # float, soft-update coefficient
    self.HARD_UPDATE = CONFIG.HARD_UPDATE  # int, period for hard_update

  @abc.abstractmethod
  def build_network(self):
    raise NotImplementedError

  def build_optimizer(self):
    """
    Builds optimizer for the Q_network and construct a scheduler for
    learning rate and reset counter for updates.
    """
    self.optimizer = torch.optim.AdamW(
        self.Q_network.parameters(), lr=self.LR_C, weight_decay=1e-3
    )
    self.scheduler = optim.lr_scheduler.StepLR(
        self.optimizer, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY
    )
    self.max_grad_norm = 1
    self.cntUpdate = 0

  @abc.abstractmethod
  def update(self):
    raise NotImplementedError

  @abc.abstractmethod
  def initBuffer(self, env):
    raise NotImplementedError

  @abc.abstractmethod
  def initQ(self):
    raise NotImplementedError

  @abc.abstractmethod
  def learn(self):
    raise NotImplementedError

  def update_target_network(self):
    """
    Updates the target network periodically.
    """
    if self.SOFT_UPDATE:
      # target = TAU * Q_network + (1-TAU) * target
      soft_update(self.target_network, self.Q_network, self.TAU)
    elif self.cntUpdate % self.HARD_UPDATE == 0:
      # Hard Replace: copy the Q-network into the target nework every
      self.target_network.load_state_dict(self.Q_network.state_dict())

  def updateHyperParam(self):
    """
    Updates the hypewr-parameters, such as learning rate, discount factor
    (GAMMA) and exploration-exploitation tradeoff (EPSILON)
    """
    lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
    if (lr <= self.LR_C_END):
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.LR_C_END
    else:
      self.scheduler.step()

    self.EpsilonScheduler.step()
    self.EPSILON = self.EpsilonScheduler.get_variable()
    self.GammaScheduler.step()
    self.GAMMA = self.GammaScheduler.get_variable()

  @abc.abstractmethod
  def select_action(self):
    raise NotImplementedError

  def store_transition(self, s, a, r, s_, info):
    """Stores a transition and explicitly records time metadata.

    The state itself may already include time. Keeping t/t_/done as separate
    replay fields makes the time-varying reach-avoid data contract explicit
    while preserving callers that pass the original five arguments.
    """
    t = info.get("t", s[-1] if len(s) > 2 else 0.0)
    if s_ is None:
      t_ = info.get("t_next", t)
    else:
      t_ = info.get("t_next", s_[-1] if len(s_) > 2 else t)
    done = info.get("done", s_ is None)
    self.memory.update(Transition(s, t, a, r, s_, t_, done, info))

  def save(self, step, logs_path):
    """Saves the model weights and save the configuration file in first call."""

    save_model(self.Q_network, step, logs_path, "Q", self.MAX_MODEL)
    if not self.saved:
      config_path = os.path.join(logs_path, "CONFIG.pkl")
      pickle.dump(self.CONFIG, open(config_path, "wb"))
      self.saved = True

  def restore(self, step, logs_path, verbose=True):
    """Restores the model weights from the given model path."""

    logs_path = os.path.join(logs_path, "model", "Q-{}.pth".format(step))
    self.Q_network.load_state_dict(
        torch.load(logs_path, map_location=self.device)
    )
    self.target_network.load_state_dict(
        torch.load(logs_path, map_location=self.device)
    )
    if verbose:
      print("  => Restore {}".format(logs_path))

  def unpack_batch(self, batch):
    """Decomposes the batch into different variables."""

    non_final_mask = torch.logical_not(
        torch.BoolTensor(batch.done).to(self.device)
    )
    non_final_state_nxt = torch.FloatTensor([
        s for s in batch.s_ if s is not None
    ]).to(self.device)
    state = torch.FloatTensor(batch.s).to(self.device)
    action = torch.LongTensor(batch.a).to(self.device).view(-1, 1)
    reward = torch.FloatTensor(batch.r).to(self.device)

    g_x = torch.FloatTensor([info["g_x"] for info in batch.info])
    g_x = g_x.to(self.device).view(-1)

    l_x = torch.FloatTensor([info["l_x"] for info in batch.info])
    l_x = l_x.to(self.device).view(-1)

    return (
        non_final_mask, non_final_state_nxt, state, action, reward, g_x, l_x
    )
