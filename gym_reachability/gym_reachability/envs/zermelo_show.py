"""
This module implements an environment considering the 2D point object dynamics.
This environemnt shows comparison between reach-avoid Q-learning and Sum
(Lagrange) Q-learning.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import random
from .env_utils import calculate_margin_rect


class ZermeloShowEnv(gym.Env):

  def __init__(
      self, device, mode='RA', doneType='toEnd', thickness=.1,
      sample_inside_obs=False, envType='show', time_varying=False,
      time_horizon=6.0
  ):
    """Initializes the environment with given arguments."""
    self.envType = envType
    self.time_varying = time_varying
    self.time_horizon = time_horizon

    # State Bounds.
    if envType == 'basic' or envType == 'easy':
      self.bounds = np.array([[-2, 2], [-2, 10]])
    else:
      self.bounds = np.array([[-3., 3.], [0., 6.]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = sample_inside_obs

    # Time-step Parameters.
    self.time_step = 0.05
    self.time = 0.0

    # Control Parameters.
    if envType == 'basic' or envType == 'easy':
      self.upward_speed = 2.
    else:
      self.upward_speed = .6
    self.horizontal_rate = 1.
    self.discrete_controls = np.array([[
        -self.horizontal_rate, self.upward_speed
    ], [0, self.upward_speed], [self.horizontal_rate, self.upward_speed]])

    # Constraint Set Parameters.
    # [X-position, Y-position, width, height]
    if envType == 'basic':
      self.constraint_x_y_w_h = np.array([
          [1.25, 2, 1.5, 1.5],
          [-1.25, 2, 1.5, 1.5],
          [0, 6, 1.5, 1.5],
      ])
      self.constraint_type = ['R', 'L', 'C']
    elif envType == 'easy':
      self.constraint_x_y_w_h = np.array([
          [1.25, 2, 1.5, 1.5],
          [-1.25, 2, 1.5, 1.5],
          [0, 6, 3., thickness],
      ])
      self.constraint_type = ['R', 'L', 'C']
    else:
      self.constraint_x_y_w_h = np.array([
          [0., 1.5, 3., thickness],
          [0., 4., 3., thickness],
      ])
      self.constraint_type = ['C', 'C']
    self.base_constraint_x_y_w_h = self.constraint_x_y_w_h.copy()

    # Target Set Parameters.
    if envType == 'basic' or envType == 'easy':
      self.target_x_y_w_h = np.array([[0., 9.25, 1.5, 1.5]])
    else:
      self.target_x_y_w_h = np.array([[0., 5.5, 1., 1.]])
    self.base_target_x_y_w_h = self.target_x_y_w_h.copy()
    self.obstacle_velocity = np.array([0.8, 0.0])
    self.obstacle_angular_freq = 0.5
    self.target_velocity = np.array([0.25, 0.0])

    # Gym variables.
    self.action_space = gym.spaces.Discrete(3)  # {left, up, right}
    self.midpoint = (self.low + self.high) / 2.0
    self.interval = self.high - self.low
    obs_low = self.midpoint - self.interval / 2
    obs_high = self.midpoint + self.interval / 2
    if self.time_varying:
      obs_low = np.append(obs_low, 0.0)
      obs_high = np.append(obs_high, self.time_horizon)
    self.observation_space = gym.spaces.Box(
        np.float32(obs_low), np.float32(obs_high)
    )
    self.viewer = None

    # Set random seed.
    self.seed_val = 0
    self.set_seed(self.seed_val)

    # Cost Parameters
    self.penalty = 1.
    self.reward = -1.
    self.costType = 'sparse'
    self.scaling = 1.

    # mode: normal or extend (keep track of ell & g)
    self.mode = mode
    if self.time_varying:
      self.state = np.zeros(3)
    elif mode == 'extend':
      self.state = np.zeros(3)
    else:
      self.state = np.zeros(2)
    self.doneType = doneType

    # Visualization Parameters
    self.constraint_set_boundary = self.get_constraint_set_boundary()
    self.target_set_boundary = self.get_target_set_boundary()
    if envType == 'basic' or envType == 'easy':
      self.visual_initial_states = [
          np.array([0, 0]),
          np.array([-1, -2]),
          np.array([1, -2]),
          np.array([-1, 4]),
          np.array([1, 4]),
      ]
    else:
      self.visual_initial_states = [
          np.array([0., 0.]),
          np.array([-1., 0.]),
          np.array([1., 0.]),
          np.array([-2.5, 0.]),
          np.array([2.5, 0.]),
          np.array([-1., 2.5]),
          np.array([1., 2.5])
      ]
    if mode == 'extend':
      self.visual_initial_states = \
          self.extend_state(self.visual_initial_states)
    elif self.time_varying:
      self.visual_initial_states = [
          self.append_time(state, 0.0) for state in self.visual_initial_states
      ]

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}; "
        "time_varying-{}".format(
            self.mode, self.doneType, self.sample_inside_obs,
            self.time_varying
        )
    )

    # for torch
    self.device = device

  def extend_state(self, states):
    """Extends the state to consist of max{ell, g}. Only used for mode='extend'."""
    new_states = []
    for state in states:
      l_x = self.target_margin(state)
      g_x = self.safety_margin(state)
      new_states.append(np.append(state, max(l_x, g_x)))
    return new_states

  def append_time(self, state, t):
    """Returns a time-augmented state [x, y, t]."""
    return np.append(np.asarray(state[:2], dtype=float), float(t))

  def split_state_time(self, state, t=None):
    """Splits either [x, y] or [x, y, t] into xy and scalar time."""
    state = np.asarray(state, dtype=float)
    xy = state[:2]
    if t is not None:
      return xy, float(t)
    if self.time_varying and state.shape[0] >= 3:
      return xy, float(state[-1])
    return xy, float(self.time)

  def get_constraint_sets(self, t=None):
    """Returns obstacle rectangles at time t.

    This simple example shifts obstacles horizontally with a sinusoidal offset.
    The method is isolated so richer time-varying obstacle models can be
    inserted without touching the RL update.
    """

    if not self.time_varying:
        return self.base_constraint_x_y_w_h
    t = self.time if t is None else t
    constraint_sets = self.base_constraint_x_y_w_h.copy()
    phase = np.linspace(0.0, np.pi, constraint_sets.shape[0])
    # Use angular frequency to control speed independently of amplitude
    offset = self.obstacle_velocity[0] * np.sin(self.obstacle_angular_freq * t + phase)
    constraint_sets[:, 0] += offset
    return constraint_sets

  def get_target_sets(self, t=None):
    """Returns target rectangles at time t."""
    if not self.time_varying:
      return self.base_target_x_y_w_h
    t = self.time if t is None else t
    target_sets = self.base_target_x_y_w_h.copy()
    target_sets[:, 0] += self.target_velocity[0] * np.sin(t)
    return target_sets

  def reset(self, start=None):
    """Resets the state of the environment."""
    self.time = 0.0
    if start is None:
      self.state = self.sample_random_state(
          sample_inside_obs=self.sample_inside_obs
      )
    else:
      if self.time_varying:
        self.state = self.append_time(start, start[-1] if len(start) > 2 else 0.0)
        self.time = self.state[-1]
      else:
        self.state = start
    return np.copy(self.state)

  def sample_random_state(self, sample_inside_obs=False):
    """Picks the state uniformly at random."""
    inside_obs = True
    # Repeat sampling until outside obstacle if needed.
    while inside_obs:
      xy_sample = np.random.uniform(low=self.low, high=self.high)
      g_x = self.safety_margin(xy_sample, t=self.time)
      inside_obs = (g_x > 0)
      if sample_inside_obs:
        break

    if self.time_varying:
      return self.append_time(xy_sample, self.time)
    return xy_sample

  # == Dynamics ==
  def step(self, action):
    """Evolves the environment one step forward under given action."""

    u = self.discrete_controls[action]
    current_state = np.copy(self.state)
    current_t = self.time
    l_x = self.target_margin(current_state, t=current_t)
    g_x = self.safety_margin(current_state, t=current_t)
    state, [l_x_next, g_x_next] = self.integrate_forward(self.state, u)
    self.state = state

    fail = g_x_next > 0
    success = l_x_next <= 0

    # = `cost` signal
    if self.mode == 'RA':
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.
    else:
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        if self.costType == 'dense_ell':
          cost = l_x_next
        elif self.costType == 'dense':
          cost = l_x_next + g_x_next
        elif self.costType == 'sparse':
          cost = 0. * self.scaling
        elif self.costType == 'max_ell_g':
          cost = max(l_x_next, g_x_next)
        else:
          raise ValueError("invalid cost type!")

    # = `done` signal
    horizon_done = self.time_varying and self.time >= self.time_horizon
    if self.doneType == 'toEnd':
      done = self.check_within_env(self.state)
    elif self.doneType == 'fail':
      done = fail
    elif self.doneType == 'TF':
      done = fail or success
    else:
      raise ValueError("invalid done type!")
    done = done or horizon_done

    # = `info`
    if done and self.doneType == 'fail':
      info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    info.update({
        "t": current_t,
        "t_next": self.time,
        "done": done,
        "g_x_next": g_x_next,
        "l_x_next": l_x_next,
    })
    return np.copy(self.state), cost, done, info

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step."""
    if self.time_varying:
      x, y, t = state
    elif self.mode == 'extend':
      x, y, z = state
    else:
      x, y = state

    # one step forward
    x = x + self.time_step * u[0]
    y = y + self.time_step * u[1]
    t_next = (t + self.time_step) if self.time_varying else self.time
    l_x = self.target_margin(np.array([x, y]), t=t_next)
    g_x = self.safety_margin(np.array([x, y]), t=t_next)

    if self.time_varying:
      self.time = t_next
      state = np.array([x, y, t_next])
    elif self.mode == 'extend':
      z = min(z, max(l_x, g_x))
      state = np.array([x, y, z])
    else:
      state = np.array([x, y])

    info = np.array([l_x, g_x])

    return state, info

  # == Setting Hyper-Parameters ==
  def set_costParam(
      self, penalty=1., reward=-1., costType='sparse', scaling=1.
  ):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.
    """
    self.penalty = penalty
    self.reward = reward
    self.costType = costType
    self.scaling = scaling

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages."""
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    torch.cuda.manual_seed_all(self.seed_val)  # if using multi-GPU.
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  def set_bounds(self, bounds):
    """Sets the boundary and the observation_space of the environment."""
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

    # Double the range in each state dimension for Gym interface.
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    obs_low = midpoint - interval/2
    obs_high = midpoint + interval/2
    if self.time_varying:
      obs_low = np.append(obs_low, 0.0)
      obs_high = np.append(obs_high, self.time_horizon)
    self.observation_space = gym.spaces.Box(
        np.float32(obs_low), np.float32(obs_high)
    )

  def set_doneType(self, doneType):
    """Sets the condition to terminate the episode."""
    self.doneType = doneType

  def set_sample_type(self, sample_inside_obs=False, verbose=False):
    """Sets the type of the sampling method."""
    self.sample_inside_obs = sample_inside_obs
    if verbose:
      print("sample_inside_obs-{}".format(self.sample_inside_obs))

  # == Getting Margin ==
  def safety_margin(self, s, t=None):
    """Computes g(s,t), positive when the state is unsafe."""
    xy, t = self.split_state_time(s, t)
    g_x_list = []

    # constraint_set_safety_margin
    for _, constraint_set in enumerate(self.get_constraint_sets(t)):
      g_x = calculate_margin_rect(xy, constraint_set, negativeInside=False)
      g_x_list.append(g_x)

    # enclosure_safety_margin
    boundary_x_y_w_h = np.append(self.midpoint, self.interval)
    g_x = calculate_margin_rect(xy, boundary_x_y_w_h, negativeInside=True)
    g_x_list.append(g_x)

    safety_margin = np.max(np.array(g_x_list))

    return self.scaling * safety_margin

  def target_margin(self, s, t=None):
    """Computes l(s,t), negative when the target is reached."""
    xy, t = self.split_state_time(s, t)
    l_x_list = []

    # target_set_safety_margin
    for _, target_set in enumerate(self.get_target_sets(t)):
      l_x = calculate_margin_rect(xy, target_set, negativeInside=True)
      l_x_list.append(l_x)

    target_margin = np.min(np.array(l_x_list))

    return self.scaling * target_margin

  # == Getting Information ==
  def check_within_env(self, state):
    """Checks if the robot is still in the environment."""
    outsideTop = (state[1] >= self.bounds[1, 1])
    outsideLeft = (state[0] <= self.bounds[0, 0])
    outsideRight = (state[0] >= self.bounds[0, 1])
    return outsideTop or outsideLeft or outsideRight

  def get_constraint_set_boundary(self, t=None):
    """Gets the constarint set boundary."""
    constraint_sets = self.get_constraint_sets(t)
    num_constarint_set = constraint_sets.shape[0]
    constraint_set_boundary = np.zeros((num_constarint_set, 5, 2))

    for idx, constraint_set in enumerate(constraint_sets):
      x, y, w, h = constraint_set
      x_l = x - w/2.0
      x_h = x + w/2.0
      y_l = y - h/2.0
      y_h = y + h/2.0
      constraint_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
      constraint_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

    return constraint_set_boundary

  def get_target_set_boundary(self, t=None):
    """Gets the target set boundary."""
    target_sets = self.get_target_sets(t)
    num_target_set = target_sets.shape[0]
    target_set_boundary = np.zeros((num_target_set, 5, 2))

    for idx, target_set in enumerate(target_sets):
      x, y, w, h = target_set
      x_l = x - w/2.0
      x_h = x + w/2.0
      y_l = y - h/2.0
      y_h = y + h/2.0
      target_set_boundary[idx, :, 0] = [x_l, x_l, x_h, x_h, x_l]
      target_set_boundary[idx, :, 1] = [y_l, y_h, y_h, y_l, y_l]

    return target_set_boundary

  def get_warmup_examples(self, num_warmup_samples=100):
    """Gets the warmup samples."""
    x_min, x_max = self.bounds[0, :]
    y_min, y_max = self.bounds[1, :]

    xs = np.random.uniform(x_min, x_max, num_warmup_samples)
    ys = np.random.uniform(y_min, y_max, num_warmup_samples)
    ts = np.random.uniform(0.0, self.time_horizon, num_warmup_samples)
    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      x, y = xs[i], ys[i]
      t = ts[i] if self.time_varying else None
      state = self.append_time([x, y], t) if self.time_varying else np.array([x, y])
      l_x = self.target_margin(state, t=t)
      g_x = self.safety_margin(state, t=t)
      heuristic_v[i, :] = np.maximum(l_x, g_x)
      states[i, :] = state

    return states, heuristic_v

  def get_axes(self):
    """Gets the axes bounds and aspect_ratio."""
    x_span = self.bounds[0, 1] - self.bounds[0, 0]
    y_span = self.bounds[1, 1] - self.bounds[1, 0]
    aspect_ratio = x_span / y_span
    axes = np.array([
        self.bounds[0, 0] - .05, self.bounds[0, 1] + .05,
        self.bounds[1, 0] - .05, self.bounds[1, 1] + .05
    ])
    return [axes, aspect_ratio]

  def get_value(self, q_func, nx=41, ny=121, addBias=False, plot_time=0.0):
    """Gets the state values given the Q-network."""
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=['multi_index'])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index

      x = xs[idx[0]]
      y = ys[idx[1]]
      base_state = np.array([x, y])
      l_x = self.target_margin(base_state, t=plot_time)
      g_x = self.safety_margin(base_state, t=plot_time)

      if self.time_varying:
        state_np = self.append_time(base_state, plot_time)
        state = torch.FloatTensor(state_np).to(self.device).unsqueeze(0)
      elif self.mode == 'normal' or self.mode == 'RA':
        state = torch.FloatTensor([x, y]).to(self.device).unsqueeze(0)
      else:
        z = max([l_x, g_x])
        state = torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)

      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return xs, ys, v

  # == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=250, state=None, keepOutOf=False, toEnd=False
  ):
    """Simulates the trajectory given the state or randomly initialized."""
    if state is None:
      state = self.sample_random_state(sample_inside_obs=not keepOutOf)
    elif self.time_varying and len(state) == 2:
      state = self.append_time(state, 0.0)
    if self.time_varying:
      self.time = state[-1]
    x, y = state[:2]
    traj_x = [x]
    traj_y = [y]
    result = 0  # not finished

    for _ in range(T):
      if toEnd:
        outsideTop = (state[1] > self.bounds[1, 1])
        outsideLeft = (state[0] < self.bounds[0, 0])
        outsideRight = (state[0] > self.bounds[0, 1])
        done = outsideTop or outsideLeft or outsideRight
        if done:
          result = 1
          break
      else:
        if self.safety_margin(state) > 0:
          result = -1  # failed
          break
        elif self.target_margin(state) <= 0:
          result = 1  # succeeded
          break
        elif self.time_varying and state[-1] >= self.time_horizon:
          result = 0
          break

      state_tensor = torch.FloatTensor(state)
      state_tensor = state_tensor.to(self.device).unsqueeze(0)
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      u = self.discrete_controls[action_index]

      state, _ = self.integrate_forward(state, u)
      traj_x.append(state[0])
      traj_y.append(state[1])

    return traj_x, traj_y, result

  def simulate_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.
    """

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      if self.envType == 'basic' or self.envType == 'easy':
        nx = 21
        ny = 61
      else:
        nx = 41
        ny = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      it = np.nditer(results, flags=['multi_index'])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]
        state = np.array([x, y])
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result
        it.iternext()
      results = results.reshape(-1)
    else:
      results = np.empty(shape=(len(states),), dtype=int)
      for idx, state in enumerate(states):
        traj_x, traj_y, result = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj_x, traj_y))
        results[idx] = result

    return trajectories, results

  # == Visualizing ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=201, ny=201, labels=None,
      boolPlot=False, addBias=False, cmap='seismic', plot_time=0.0
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cbarPlot = True

    # == Plot failure / target set ==
    self.plot_target_failure_set(ax, t=plot_time)

    # == Plot reach-avoid set ==
    if not self.time_varying:
      self.plot_reach_avoid_set(ax)

    # == Plot V ==
    self.plot_v_values(
        q_func, ax=ax, fig=fig, vmin=vmin, vmax=vmax, nx=nx, ny=ny, cmap=cmap,
        boolPlot=boolPlot, cbarPlot=cbarPlot, addBias=addBias,
        plot_time=plot_time
    )

    # == Plot Trajectories ==
    self.plot_trajectories(
        q_func, states=self.visual_initial_states, toEnd=False, ax=ax
    )

    # == Formatting ==
    self.plot_formatting(ax=ax, labels=labels)
    fig.tight_layout()

  def plot_v_values(
      self, q_func, ax=None, fig=None, vmin=-1, vmax=1, nx=201, ny=201,
      cmap='seismic', alpha=0.8, boolPlot=False, cbarPlot=True, addBias=False,
      plot_time=0.0
  ):
    """Plots state values."""
    axStyle = self.get_axes()

    # == Plot V ==
    _, _, v = self.get_value(q_func, nx, ny, addBias=addBias, plot_time=plot_time)
    vmax = np.ceil(max(np.max(v), np.max(-v)))
    vmin = -vmax

    if boolPlot:
      im = ax.imshow(
          v.T > 0., interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, alpha=alpha
      )
    else:
      im = ax.imshow(
          v.T, interpolation='none', extent=axStyle[0], origin="lower",
          cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
            ticks=[vmin, 0, vmax]
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  def plot_trajectories(
      self, q_func, T=250, num_rnd_traj=None, states=None, toEnd=False,
      ax=None, c='k', lw=2, zorder=2
  ):
    """Plots trajectories given the agent's Q-network."""

    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))

    trajectories, results = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
    )

    for traj in trajectories:
      traj_x, traj_y = traj
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

    return results

  def plot_target_failure_set(
      self, ax=None, c_c='m', c_t='y', lw=1.5, zorder=1, t=None
  ):
    """Plots the target and the failure set."""
    # Plot bounadries of constraint set.
    for one_boundary in self.get_constraint_set_boundary(t):
      ax.plot(
          one_boundary[:, 0], one_boundary[:, 1], color=c_c, lw=lw,
          zorder=zorder
      )

    # Plot boundaries of target set.
    for one_boundary in self.get_target_set_boundary(t):
      ax.plot(
          one_boundary[:, 0], one_boundary[:, 1], color=c_t, lw=lw,
          zorder=zorder
      )

  def plot_reach_avoid_set(self, ax=None, c='g', lw=3, zorder=1):
    """Plots the analytic reach-avoid set."""
    slope = self.upward_speed / self.horizontal_rate

    def get_line(slope, end_point, x_limit, ns=100):
      x_end, y_end = end_point
      b = y_end - slope*x_end

      xs = np.linspace(x_limit, x_end, ns)
      ys = xs*slope + b
      return xs, ys

    # unsafe set
    for cons, cType in zip(self.constraint_x_y_w_h, self.constraint_type):
      x, y, w, h = cons
      x1 = x - w/2.0
      x2 = x + w/2.0
      y_min = y - h/2.0
      if cType == 'C':
        xs, ys = get_line(-slope, end_point=[x1, y_min], x_limit=x)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
        xs, ys = get_line(slope, end_point=[x2, y_min], x_limit=x)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
      elif cType == 'L':
        x_limit = self.bounds[0, 0]
        xs, ys = get_line(slope, end_point=[x2, y_min], x_limit=x_limit)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
      elif cType == 'R':
        x_limit = self.bounds[0, 1]
        xs, ys = get_line(-slope, end_point=[x1, y_min], x_limit=x_limit)
        ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

    # border unsafe set
    x, y, w, h = self.target_x_y_w_h[0]
    x1 = x - w/2.0
    x2 = x + w/2.0
    y_max = y + h/2.0
    xs, ys = get_line(slope, end_point=[x1, y_max], x_limit=self.low[0])
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)
    xs, ys = get_line(-slope, end_point=[x2, y_max], x_limit=self.high[0])
    ax.plot(xs, ys, color=c, linewidth=lw, zorder=zorder)

  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization."""
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis='both', which='both', bottom=False, top=False, left=False,
        right=False
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
