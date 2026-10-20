"""
Implementing
    - a' = argmin_a' Q_network(s', a')
    - V(s') = Q_target(s', a')
    - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } }
             + (1-gamma) max{ g(s), l(s) }
    - loss = E[ ( V(f(s,a)) - Q_network(s,a) )^2 ]
"""

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import Model
from .DDQN import DDQN, Transition


def format_eta(seconds):
  """Formats seconds as a compact approximate ETA string."""
  if seconds is None or not np.isfinite(seconds) or seconds < 0:
    return "estimating"
  seconds = int(seconds)
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  if hours > 0:
    return "{:d}h {:02d}m".format(hours, minutes)
  return "{:d}m".format(minutes)


class DDQNSingle(DDQN):
  """
  Implements the double deep Q-network algorithm. Supports minimizing the
  reach-avoid cost or the standard sum of discounted costs.
  """

  def __init__(
      self, CONFIG, numAction, actionList, dimList, mode="RA",
      terminalType="g", verbose=True
  ):

    super(DDQNSingle, self).__init__(CONFIG)

    self.mode = mode  # 'normal' or 'RA'
    self.terminalType = terminalType

    # == ENV PARAM ==
    self.numAction = numAction
    self.actionList = actionList

    # == Build neural network for (D)DQN ==
    self.dimList = dimList
    self.actType = CONFIG.ACTIVATION
    self.build_network(dimList, self.actType, verbose)
    print(
        "DDQN: mode-{}; terminalType-{}".format(self.mode, self.terminalType)
    )

  def build_network(self, dimList, actType="Tanh", verbose=True):
    """Builds a neural network for the Q-network."""

    self.Q_network = Model(dimList, actType, verbose=verbose)
    self.target_network = Model(dimList, actType)

    if self.device == torch.device("cuda"):
      self.Q_network.cuda()
      self.target_network.cuda()

    self.build_optimizer()

  def update(self, addBias=False):
    """Updates the Q-network using a batch of sampled replay transitions."""

    if len(self.memory) < self.BATCH_SIZE * 20:
      return

    # == EXPERIENCE REPLAY ==
    transitions = self.memory.sample(self.BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    (non_final_mask, non_final_state_nxt, state, action, reward, g_x,
     l_x) = self.unpack_batch(batch)

    self.Q_network.train()
    state_action_values = (
        self.Q_network(state).gather(dim=1, index=action).view(-1)
    )

    with torch.no_grad():
      self.Q_network.eval()
      if non_final_state_nxt.shape[0] > 0:
        # Reach-avoid values are costs/margins in this implementation:
        # negative is good/reached, positive is unsafe. The controller
        # therefore minimizes over actions, matching the original RARL code.
        action_nxt = (
            self.Q_network(non_final_state_nxt).min(1, keepdim=True)[1]
        )
      else:
        action_nxt = torch.empty((0, 1), dtype=torch.long).to(self.device)

    # == get expected value ==
    state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

    with torch.no_grad():  # V(s') = Q_target(s', a'), a' is from Q_network
      if non_final_state_nxt.shape[0] > 0:
        if self.double_network:
          self.target_network.eval()
          Q_expect = self.target_network(non_final_state_nxt)
        else:
          self.Q_network.eval()
          Q_expect = self.Q_network(non_final_state_nxt)
        state_value_nxt[non_final_mask] = \
            Q_expect.gather(dim=1, index=action_nxt).view(-1)

    # == Discounted Reach-Avoid Bellman Equation (DRABE) ==
    if self.mode == "RA":
      y = torch.zeros(self.BATCH_SIZE).float().to(self.device)
      final_mask = torch.logical_not(non_final_mask)
      if addBias:  # Bias version:
        # V(s) = gamma ( max{ g(s), min{ l(s), V_diff(s') } }
        #        - max{ g(s), l(s) } ),
        # where V_diff(s') = V(s') + max{ g(s'), l(s') }
        min_term = torch.min(l_x, state_value_nxt + torch.max(l_x, g_x))
        terminal = torch.max(l_x, g_x)
        non_terminal = torch.max(min_term, g_x) - terminal
        y[non_final_mask] = self.GAMMA * non_terminal[non_final_mask]
        y[final_mask] = terminal[final_mask]
      else:
        # Time-varying DRABE with time included in state:
        # Q(s,t,a) = (1-gamma) max{l(s,t), g(s,t)}
        #          + gamma max{g(s,t), min{l(s,t), min_a' Q(s',t+dt,a')}}.
        non_terminal = torch.max(
            g_x[non_final_mask],
            torch.min(l_x[non_final_mask], state_value_nxt[non_final_mask]),
        )
        terminal = torch.max(l_x, g_x)

        # normal state
        y[non_final_mask] = non_terminal * self.GAMMA + terminal[
            non_final_mask] * (1 - self.GAMMA)

        # FIX 3: terminal state target must be max{l(s), g(s)} for both
        # terminalType branches. Using only g_x when the episode ends on
        # success (l_x <= 0) pulls the target toward a large positive number
        # instead of the correct negative value, biasing the value boundary
        # inward and inflating the false-success rate.
        if self.terminalType == "g":
          y[final_mask] = terminal[final_mask]   # was: g_x[final_mask]
        elif self.terminalType == "max":
          y[final_mask] = terminal[final_mask]

    else:  # V(s) = c(s, a) + gamma * V(s')
      y = state_value_nxt * self.GAMMA + reward

    # == regression: Q(s, a) <- V(s) ==
    loss = smooth_l1_loss(
        input=state_action_values,
        target=y.detach(),
    )

    # == backpropagation ==
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.update_target_network()

    return loss.item()

  def initBuffer(self, env):
    """Adds some transitions to the replay memory (buffer) randomly."""

    cnt = 0
    # FIX 1: s was used before being assigned, causing a NameError crash.
    # Initialize s before the while loop begins.
    s = env.reset()
    while len(self.memory) < self.memory.capacity:
      cnt += 1
      print("\rWarmup Buffer [{:d}]".format(cnt), end="")

      a, a_idx = self.select_action(s, explore=True)
      s_, r, done, info = env.step(a_idx)
      s_ = None if done else s_

      # Store the transition in memory
      self.store_transition(s, a_idx, r, s_, info)
      if done:
        s = env.reset()
      else:
        s = s_
    print(" --- Warmup Buffer Ends")

  def initQ(
      self, env, warmupIter, outFolder, num_warmup_samples=200, vmin=-1,
      vmax=1, plotFigure=True, storeFigure=True
  ):
    """
    Initalizes the Q-network given that the environment can provide warmup
    examples with heuristic values.
    """
    lossList = np.empty(warmupIter, dtype=float)
    for ep_tmp in range(warmupIter):
      states, heuristic_v = env.get_warmup_examples(
          num_warmup_samples=num_warmup_samples
      )

      self.Q_network.train()
      heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
      states = torch.from_numpy(states).float().to(self.device)
      v = self.Q_network(states)
      loss = mse_loss(input=v, target=heuristic_v, reduction="sum")

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
      self.optimizer.step()
      lossList[ep_tmp] = loss.detach().cpu().numpy()
      print(
          "\rWarmup Q [{:d}]. MSE = {:f}".format(ep_tmp + 1, loss),
          end="",
      )

    print(" --- Warmup Q Ends")
    if plotFigure or storeFigure:
      env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic")
      if storeFigure:
        figureFolder = os.path.join(outFolder, "figure")
        os.makedirs(figureFolder, exist_ok=True)
        figurePath = os.path.join(figureFolder, "initQ.png")
        plt.savefig(figurePath)
      if plotFigure:
        plt.pause(0.001)
      plt.clf()
      plt.close('all')
    self.target_network.load_state_dict(
        self.Q_network.state_dict()
    )  # hard replace
    self.build_optimizer()

    return lossList

  def learn(
      self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100, warmupBuffer=True,
      warmupQ=False, warmupIter=10000, addBias=False, doneTerminate=True,
      runningCostThr=None, curUpdates=None, checkPeriod=50000, plotFigure=True,
      storeFigure=False, showBool=False, vmin=-1, vmax=1, numRndTraj=200,
      storeModel=True, storeBest=False, outFolder="RA", verbose=True
  ):
    """Learns the Q function given the training hyper-parameters."""

    # == Warmup Buffer ==
    startInitBuffer = time.time()
    if warmupBuffer:
      self.initBuffer(env)
    endInitBuffer = time.time()

    # == Warmup Q ==
    startInitQ = time.time()
    if warmupQ:
      self.initQ(
          env, warmupIter=warmupIter, outFolder=outFolder,
          plotFigure=plotFigure, storeFigure=storeFigure, vmin=vmin, vmax=vmax
      )
    endInitQ = time.time()

    # == Main Training ==
    startLearning = time.time()
    lastEtaPrint = startLearning
    trainingRecords = []
    runningCost = 0.0
    trainProgress = []
    checkPointSucc = 0.0
    ep = 0

    if curUpdates is not None:
      self.cntUpdate = curUpdates
      print("starting from {:d} updates".format(self.cntUpdate))

    if storeModel:
      modelFolder = os.path.join(outFolder, "model")
      os.makedirs(modelFolder, exist_ok=True)
    if storeFigure:
      figureFolder = os.path.join(outFolder, "figure")
      os.makedirs(figureFolder, exist_ok=True)

    while self.cntUpdate <= MAX_UPDATES:
      s = env.reset()
      epCost = 0.0
      ep += 1
      # Rollout
      for step_num in range(MAX_EP_STEPS):
        # Select action
        a, a_idx = self.select_action(s, explore=True)

        # Interact with env
        s_, r, done, info = env.step(a_idx)
        epCost += r

        # Store the transition in memory. Pass s_=None when the episode ends
        # so that unpack_batch correctly builds non_final_mask.
        self.store_transition(s, a_idx, r, None if done else s_, info)

        # FIX 2: always keep s as a valid numpy state. The original code did
        # `s = s_` after setting `s_ = None` on done, which passed None into
        # select_action on the next step (crash or silent wrong behaviour).
        # Reset the environment immediately when the episode is done instead.
        if done:
          s = env.reset()
        else:
          s = s_

        # Check after fixed number of gradient updates
        if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
          results = env.simulate_trajectories(
              self.Q_network, T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
              toEnd=False
          )[1]
          success = np.sum(results == 1) / results.shape[0]
          failure = np.sum(results == -1) / results.shape[0]
          unfinish = np.sum(results == 0) / results.shape[0]
          trainProgress.append([success, failure, unfinish])
          if verbose:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            print("\nAfter [{:d}] updates:".format(self.cntUpdate))
            print(
                "  - eps={:.2f}, gamma={:.6f}, lr={:.1e}.".format(
                    self.EPSILON, self.GAMMA, lr
                )
            )
            print("  - success/failure/unfinished ratio:", end=" ")
            with np.printoptions(formatter={"float": "{: .3f}".format}):
              print(np.array([success, failure, unfinish]))

          if storeModel:
            if storeBest:
              if success > checkPointSucc:
                checkPointSucc = success
                self.save(self.cntUpdate, modelFolder)
            else:
              self.save(self.cntUpdate, modelFolder)

          if plotFigure or storeFigure:
            if showBool:
              env.visualize(
                  self.Q_network, vmin=0, boolPlot=True, addBias=addBias
              )
            else:
              env.visualize(
                  self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic",
                  addBias=addBias
              )
            if storeFigure:
              figurePath = os.path.join(
                  figureFolder, "{:d}.png".format(self.cntUpdate)
              )
              plt.savefig(figurePath)
            if plotFigure:
              plt.pause(0.001)
            plt.clf()
            plt.close('all')

        # Perform one step of the optimization (on the target network)
        lossC = self.update(addBias=addBias)
        trainingRecords.append(lossC)
        self.cntUpdate += 1
        self.updateHyperParam()

        # Terminate early
        if done and doneTerminate:
          break

      # Rollout report
      runningCost = runningCost*0.9 + epCost*0.1
      if verbose:
        now = time.time()
        etaText = ""
        if now - lastEtaPrint >= 10 or self.cntUpdate >= MAX_UPDATES:
          elapsed = now - startLearning
          updatesDone = max(self.cntUpdate - (curUpdates or 0), 1)
          updatesLeft = max(MAX_UPDATES - self.cntUpdate, 0)
          secPerUpdate = elapsed / updatesDone
          etaText = " ETA ~ {}".format(format_eta(secPerUpdate * updatesLeft))
          lastEtaPrint = now
        print(
            "\r[{:d}-{:d}]: ".format(ep, self.cntUpdate)
            + "This episode gets running/episode cost = "
            + "({:3.2f}/{:.2f}) after {:d} steps.".
            format(runningCost, epCost, step_num + 1)
            + etaText,
            end="",
        )

      # Check stopping criteria
      if runningCostThr is not None:
        if runningCost <= runningCostThr:
          print(
              "\n At Updates[{:3.0f}] Solved!".format(self.cntUpdate)
              + " Running cost is now {:3.2f}!".format(runningCost)
          )
          env.close()
          break
    endLearning = time.time()
    timeInitBuffer = endInitBuffer - startInitBuffer
    timeInitQ = endInitQ - startInitQ
    timeLearning = endLearning - startLearning
    self.save(self.cntUpdate, modelFolder)
    print(
        "\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}".format(
            timeInitBuffer, timeInitQ, timeLearning
        )
    )
    trainingRecords = np.array(trainingRecords)
    trainProgress = np.array(trainProgress)
    return trainingRecords, trainProgress

  def select_action(self, state, explore=False):
    """Selects the action given the state and conditioned on `explore` flag."""
    self.Q_network.eval()
    # tensor.min() returns (value, indices), which are in the tensor form.
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    if (np.random.rand() < self.EPSILON) and explore:
      action_index = np.random.randint(0, self.numAction)
    else:
      action_index = self.Q_network(state).min(dim=1)[1].item()
    return self.actionList[action_index], action_index