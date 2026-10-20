class config(object):
  """Contains basic hyper-parameters for reinforcement learning algorithms.
  """

  def __init__(
      self, ENV_NAME="Pendulum-v0", DEVICE="cpu", SEED=0, MAX_UPDATES=2000000,
      MAX_EP_STEPS=200, EPSILON=0.95, EPS_END=0.05, EPS_PERIOD=1,
      EPS_DECAY=0.5, EPS_RESET_PERIOD=100, LR_C=1e-3, LR_C_END=1e-4,
      LR_C_PERIOD=1, LR_C_DECAY=0.5, GAMMA=0.9, GAMMA_END=0.99999999,
      GAMMA_PERIOD=200, GAMMA_DECAY=0.5, MEMORY_CAPACITY=10000, BATCH_SIZE=32,
      RENDER=False, MAX_MODEL=5, ARCHITECTURE=None, ACTIVATION="Tanh",
      SKIP=False, REWARD=-1, PENALTY=1
  ):

    if ARCHITECTURE is None:
      ARCHITECTURE = [512, 512, 512]
    self.MAX_UPDATES = MAX_UPDATES
    self.MAX_EP_STEPS = MAX_EP_STEPS

    self.EPSILON = EPSILON
    self.EPS_END = EPS_END
    self.EPS_PERIOD = EPS_PERIOD
    self.EPS_DECAY = EPS_DECAY
    self.EPS_RESET_PERIOD = EPS_RESET_PERIOD

    self.LR_C = LR_C
    self.LR_C_END = LR_C_END
    self.LR_C_PERIOD = LR_C_PERIOD
    self.LR_C_DECAY = LR_C_DECAY

    self.GAMMA = GAMMA
    self.GAMMA_END = GAMMA_END
    self.GAMMA_PERIOD = GAMMA_PERIOD
    self.GAMMA_DECAY = GAMMA_DECAY

    self.MEMORY_CAPACITY = MEMORY_CAPACITY
    self.BATCH_SIZE = BATCH_SIZE

    self.RENDER = RENDER
    self.ENV_NAME = ENV_NAME
    self.SEED = SEED

    self.MAX_MODEL = MAX_MODEL
    self.DEVICE = DEVICE

    self.ARCHITECTURE = ARCHITECTURE
    self.ACTIVATION = ACTIVATION
    self.SKIP = SKIP

    self.REWARD = REWARD
    self.PENALTY = PENALTY


class dqnConfig(config):
  """Contains hyper-parameters for (double) deep Q-network.
  """

  def __init__(
      self, ENV_NAME="Pendulum-v0", DEVICE="cpu", SEED=0, MAX_UPDATES=2000000,
      MAX_EP_STEPS=200, EPSILON=0.95, EPS_END=0.05, EPS_PERIOD=1,
      EPS_DECAY=0.5, EPS_RESET_PERIOD=100, LR_C=1e-3, LR_C_END=1e-4,
      LR_C_PERIOD=1, LR_C_DECAY=0.5, GAMMA=0.9, GAMMA_END=0.99999999,
      GAMMA_PERIOD=200, GAMMA_DECAY=0.5, TAU=0.01, HARD_UPDATE=1,
      SOFT_UPDATE=True, MEMORY_CAPACITY=10000, BATCH_SIZE=32, RENDER=False,
      MAX_MODEL=10, DOUBLE=True, ARCHITECTURE=None, ACTIVATION="Tanh",
      SKIP=False, REWARD=-1, PENALTY=1
  ):
    """
    Initializes a configuration object for (double) deep Q-network with the
    specified attributes.
    """
    if ARCHITECTURE is None:
      ARCHITECTURE = [512, 512, 512]
    super().__init__(
        ENV_NAME=ENV_NAME,
        DEVICE=DEVICE,
        SEED=SEED,
        MAX_UPDATES=MAX_UPDATES,
        MAX_EP_STEPS=MAX_EP_STEPS,
        EPSILON=EPSILON,
        EPS_END=EPS_END,
        EPS_PERIOD=EPS_PERIOD,
        EPS_DECAY=EPS_DECAY,
        EPS_RESET_PERIOD=EPS_RESET_PERIOD,
        LR_C=LR_C,
        LR_C_END=LR_C_END,
        LR_C_PERIOD=LR_C_PERIOD,
        LR_C_DECAY=LR_C_DECAY,
        GAMMA=GAMMA,
        GAMMA_END=GAMMA_END,
        GAMMA_PERIOD=GAMMA_PERIOD,
        GAMMA_DECAY=GAMMA_DECAY,
        MEMORY_CAPACITY=MEMORY_CAPACITY,
        BATCH_SIZE=BATCH_SIZE,
        RENDER=RENDER,
        MAX_MODEL=MAX_MODEL,
        ARCHITECTURE=ARCHITECTURE,
        ACTIVATION=ACTIVATION,
        SKIP=SKIP,
        REWARD=REWARD,
        PENALTY=PENALTY,
    )
    self.DOUBLE = DOUBLE
    self.TAU = TAU
    self.HARD_UPDATE = HARD_UPDATE
    self.SOFT_UPDATE = SOFT_UPDATE
