import numpy as np


class ReplayMemory(object):
  """Contains a replay memory (or a memory buffer). """

  def __init__(self, capacity, seed=0):
    """Initializes the memory with the maximum capacity and a random seed."""
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.isfull = False
    self.seed = seed
    np.random.seed(self.seed)

  def reset(self):
    """Clears the memory and reset the position to be zero."""
    self.memory = []
    self.position = 0
    self.isfull = False

  def update(self, transition):
    """Updates the memory given the newcoming transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = transition
    self.position = int((self.position + 1) % self.capacity)
    if len(self.memory) == self.capacity:
      self.isfull = True

  def sample(self, batch_size):
    """Samples batch_size transitions from the memory uniformly at random."""
    length = len(self.memory)
    indices = np.random.randint(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    """Returns the number of transitions in the memory."""
    return len(self.memory)
