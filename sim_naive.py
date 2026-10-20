"""
To Run
        python sim_naive.py

Default run:
        time-varying reach-avoid, warmup, gamma annealing, and saved figures.
"""

import os
import glob
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj, load_obj
from gym_reachability import gym_reachability  # Custom Gym env.

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-r", "--reward", help="when entering target set", default=-1, type=float
)
parser.add_argument(
    "-p", "--penalty", help="when entering failure set", default=1, type=float
)
parser.add_argument(
    "-s", "--scaling", help="scaling of ell/g", default=4, type=float
)
parser.add_argument(
    "-tv", "--timeVarying", help="use moving target and obstacle sets",
    action="store_true", default=True
)
parser.add_argument(
    "--staticZones", help="use the original time-invariant target/obstacles",
    action="store_false", dest="timeVarying"
)
parser.add_argument(
    "-th", "--timeHorizon", help="episode time horizon for time-varying RA",
    default=12.0, type=float
)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "--noWarmup", help="skip Q-network warmup", action="store_false",
    dest="warmup"
)
parser.set_defaults(warmup=True)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=2000, type=int
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=5000000,
    type=int
)
parser.add_argument(
    "-ut", "--updateTimes", help="#hyper-param. steps", default=20, type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=10000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=400000, type=int
)

# NN hyper-parameters
parser.add_argument(
    "-a", "--annealing", help="gamma annealing", action="store_true",
    default=True
)
parser.add_argument(
    "--noAnnealing", help="disable gamma annealing", action="store_false",
    dest="annealing"
)
parser.add_argument(
    "-arc", "--architecture", help="NN architecture", default=[256, 128],
    nargs="*", type=int
)
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9999, type=float
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='Tanh', type=str
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='RA', type=str)
parser.add_argument(
    "-tt", "--terminalType", help="terminal value", default='g', type=str
)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='tv', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file", default='experiments', type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true",
    default=True
)
parser.add_argument(
    "--noStoreFigure", help="do not save figures", action="store_false",
    dest="storeFigure"
)
parser.add_argument(
    "-eo", "--evalOnly", help="load a trained model and make figures only",
    action="store_true"
)
parser.add_argument(
    "-ls", "--loadStep", help="checkpoint update step to load; -1 loads latest",
    default=-1, type=int
)
parser.add_argument(
    "-et", "--evalTraj", help="#trajectories for evaluation summary",
    default=10000, type=int
)
args = parser.parse_args()
print(args)


def get_model_steps(modelFolder):
  """Returns available checkpoint update steps from Q-*.pth files."""
  steps = []
  for modelPath in glob.glob(os.path.join(modelFolder, "Q-*.pth")):
    modelName = os.path.basename(modelPath)
    try:
      steps.append(int(modelName[2:-4]))
    except ValueError:
      pass
  return sorted(steps)


def pick_model_step(modelFolder, requestedStep):
  """Picks the requested checkpoint, or latest when requestedStep is -1."""
  steps = get_model_steps(modelFolder)
  if len(steps) == 0:
    raise FileNotFoundError("No Q-*.pth checkpoints found in {}".format(
        modelFolder
    ))
  if requestedStep == -1:
    return steps[-1]
  if requestedStep not in steps:
    raise FileNotFoundError(
        "Checkpoint Q-{}.pth not found. Available steps: {}".format(
            requestedStep, steps
        )
    )
  return requestedStep


def load_training_history(outFolder):
  """Loads train.pkl if it exists."""
  trainPath = os.path.join(outFolder, "train")
  if os.path.exists(trainPath + ".pkl"):
    return load_obj(trainPath)
  return None


def print_eval_summary(results, prefix="Evaluation"):
  """Prints success/failure/unfinished ratios from rollout results."""
  success = np.sum(results == 1) / results.shape[0]
  failure = np.sum(results == -1) / results.shape[0]
  unfinish = np.sum(results == 0) / results.shape[0]
  print("{} success/failure/unfinished: {:.3f}/{:.3f}/{:.3f}".format(
      prefix, success, failure, unfinish
  ))
  return success, failure, unfinish


def compute_rollout_metrics(results, steps=None):
  """Computes safety/liveness metrics from rollout outcomes."""
  results = np.asarray(results).reshape(-1)
  total = max(results.shape[0], 1)
  successMask = results == 1
  failureMask = results == -1
  unfinishedMask = results == 0
  safeCount = np.sum(~failureMask)
  metrics = {
      "num_rollouts": int(total),
      "success_rate": float(np.sum(successMask) / total),
      "failure_rate": float(np.sum(failureMask) / total),
      "unfinished_rate": float(np.sum(unfinishedMask) / total),
      "safety_rate": float(1.0 - np.sum(failureMask) / total),
      "liveness_given_safe_rate": float(
          np.sum(successMask) / max(safeCount, 1)
      ),
  }
  if steps is not None:
    steps = np.asarray(steps)
    metrics["mean_steps_to_success"] = float(
        np.mean(steps[successMask])
    ) if np.any(successMask) else None
    metrics["mean_steps_to_failure"] = float(
        np.mean(steps[failureMask])
    ) if np.any(failureMask) else None
  return metrics


def print_metrics(metrics, prefix="Metrics"):
  """Prints a compact safety/liveness metrics block."""
  print("\n{}:".format(prefix))
  print("  rollouts: {:d}".format(metrics["num_rollouts"]))
  print("  liveness success rate: {:.3f}".format(metrics["success_rate"]))
  print("  safety rate: {:.3f}".format(metrics["safety_rate"]))
  print("  failure / violation rate: {:.3f}".format(metrics["failure_rate"]))
  print("  unfinished rate: {:.3f}".format(metrics["unfinished_rate"]))
  print("  success among safe rollouts: {:.3f}".format(
      metrics["liveness_given_safe_rate"]
  ))
  if metrics.get("mean_steps_to_success") is not None:
    print("  mean steps to success: {:.1f}".format(
        metrics["mean_steps_to_success"]
    ))
  if metrics.get("mean_steps_to_failure") is not None:
    print("  mean steps to failure: {:.1f}".format(
        metrics["mean_steps_to_failure"]
    ))


def evaluate_policy_detailed(env, q_func, states, T=250):
  """Evaluates a policy and returns outcomes plus timing/margin diagnostics."""
  results = []
  steps = []
  maxSafetyMargins = []
  minTargetMargins = []

  for state in states:
    if env.time_varying and len(state) == 2:
      state = env.append_time(state, 0.0)
    state = np.asarray(state, dtype=float)
    if env.time_varying:
      env.time = state[-1]

    result = 0
    maxG = -np.inf
    minL = np.inf
    stepCount = T

    for stepIdx in range(T + 1):
      g_x = env.safety_margin(state)
      l_x = env.target_margin(state)
      maxG = max(maxG, g_x)
      minL = min(minL, l_x)

      if g_x > 0:
        result = -1
        stepCount = stepIdx
        break
      if l_x <= 0:
        result = 1
        stepCount = stepIdx
        break
      if env.time_varying and state[-1] >= env.time_horizon:
        result = 0
        stepCount = stepIdx
        break
      if stepIdx == T:
        break

      stateTensor = torch.FloatTensor(state).to(env.device).unsqueeze(0)
      actionIndex = q_func(stateTensor).min(dim=1)[1].item()
      state, _ = env.integrate_forward(state, env.discrete_controls[actionIndex])

    results.append(result)
    steps.append(stepCount)
    maxSafetyMargins.append(maxG)
    minTargetMargins.append(minL)

  return {
      "results": np.asarray(results, dtype=int),
      "steps": np.asarray(steps, dtype=int),
      "maxSafetyMargins": np.asarray(maxSafetyMargins, dtype=float),
      "minTargetMargins": np.asarray(minTargetMargins, dtype=float),
  }


def compute_value_classification_metrics(valueMtx, resultMtx):
  """Compares learned V<=0 predictions with rollout outcomes."""
  predictedSafe = valueMtx.reshape(-1) <= 0
  succeeded = resultMtx.reshape(-1) == 1
  failed = resultMtx.reshape(-1) == -1
  total = max(predictedSafe.shape[0], 1)
  predictedCount = max(np.sum(predictedSafe), 1)
  successCount = max(np.sum(succeeded), 1)
  return {
      "predicted_reach_avoid_rate": float(np.sum(predictedSafe) / total),
      "predicted_safe_precision": float(
          np.sum(predictedSafe & succeeded) / predictedCount
      ),
      "predicted_safe_failure_rate": float(
          np.sum(predictedSafe & failed) / predictedCount
      ),
      "success_coverage_by_value": float(
          np.sum(predictedSafe & succeeded) / successCount
      ),
  }


def write_end_report(reportPath, lines):
  """Writes a plain-text experiment report."""
  with open(reportPath, "w") as f:
    f.write("\n".join(lines))
    f.write("\n")
  print("Saved detailed report to {}".format(reportPath))


def save_zone_motion_figure(env, figureFolder, plotFigure=False,
                            storeFigure=True):
  """Saves target/obstacle center motion for time-varying experiments."""
  if not env.time_varying:
    return

  times = np.linspace(0.0, env.time_horizon, 200)
  targetCenters = np.array([env.get_target_sets(t)[0, :2] for t in times])
  obstacleCenters = [
      np.array([env.get_constraint_sets(t)[idx, :2] for t in times])
      for idx in range(env.base_constraint_x_y_w_h.shape[0])
  ]

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  ax = axes[0]
  ax.plot(targetCenters[:, 0], targetCenters[:, 1], 'y-', lw=2,
          label='target')
  for idx, centers in enumerate(obstacleCenters):
    ax.plot(centers[:, 0], centers[:, 1], lw=2,
            label='obstacle {}'.format(idx + 1))
  ax.set_xlabel('x center')
  ax.set_ylabel('y center')
  ax.set_title('Zone Center Paths')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  ax = axes[1]
  ax.plot(times, targetCenters[:, 0], 'y-', lw=2, label='target x')
  for idx, centers in enumerate(obstacleCenters):
    ax.plot(times, centers[:, 0], lw=2, label='obstacle {} x'.format(idx + 1))
  ax.set_xlabel('time')
  ax.set_ylabel('x center')
  ax.set_title('Motion Over Time')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'zone_motion.png')
    fig.savefig(figurePath)
    print("Saved {}".format(figurePath))
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()


def save_metrics_summary_figure(gridMetrics, randomMetrics, valueMetrics,
                                figureFolder, plotFigure=False,
                                storeFigure=True):
  """Saves a compact final safety/liveness metric chart."""
  fig, axes = plt.subplots(1, 3, figsize=(13, 4))

  ax = axes[0]
  names = ['success', 'safety', 'failure', 'unfinished']
  values = [
      gridMetrics['success_rate'], gridMetrics['safety_rate'],
      gridMetrics['failure_rate'], gridMetrics['unfinished_rate']
  ]
  ax.bar(names, values, color=['tab:green', 'tab:blue', 'tab:red', 'tab:gray'])
  ax.set_ylim(0, 1)
  ax.set_title('Grid Rollouts')
  ax.tick_params(axis='x', rotation=25)

  ax = axes[1]
  values = [
      randomMetrics['success_rate'], randomMetrics['safety_rate'],
      randomMetrics['failure_rate'], randomMetrics['unfinished_rate']
  ]
  ax.bar(names, values, color=['tab:green', 'tab:blue', 'tab:red', 'tab:gray'])
  ax.set_ylim(0, 1)
  ax.set_title('Random Rollouts')
  ax.tick_params(axis='x', rotation=25)

  ax = axes[2]
  names = ['safe set\nsize', 'precision', 'unsafe\ninside', 'coverage']
  values = [
      valueMetrics['predicted_reach_avoid_rate'],
      valueMetrics['predicted_safe_precision'],
      valueMetrics['predicted_safe_failure_rate'],
      valueMetrics['success_coverage_by_value'],
  ]
  ax.bar(names, values, color=['tab:purple', 'tab:green', 'tab:red', 'tab:blue'])
  ax.set_ylim(0, 1)
  ax.set_title('Value Set Diagnostics')
  ax.tick_params(axis='x', rotation=25)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'metrics_summary.png')
    fig.savefig(figurePath)
    print("Saved {}".format(figurePath))
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()


def save_training_summary_figure(trainDict, figureFolder, maxUpdates,
                                 plotFigure=False, storeFigure=True):
  """Saves loss and checkpoint success plots from train.pkl."""
  if trainDict is None:
    print("No train.pkl found, so train_loss_success.png was skipped.")
    return

  trainRecords = trainDict.get("trainRecords")
  trainProgress = trainDict.get("trainProgress")
  if trainRecords is None or trainProgress is None or len(trainProgress) == 0:
    print("Training history is incomplete, so train summary plot was skipped.")
    return

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  ax = axes[0]
  ax.plot(trainRecords, 'b:')
  ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
  ax.set_xticks(np.linspace(0, maxUpdates, 5))
  ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
  ax.set_title('loss_critic', fontsize=18)
  ax.set_xlim(left=0, right=maxUpdates)

  ax = axes[1]
  x = np.linspace(maxUpdates / trainProgress.shape[0], maxUpdates,
                  trainProgress.shape[0])
  ax.plot(x, trainProgress[:, 0], 'g-o', label='success')
  ax.plot(x, trainProgress[:, 1], 'r-o', label='failure')
  ax.plot(x, trainProgress[:, 2], 'k-o', label='unfinished')
  ax.set_xlabel('Gradient updates', fontsize=18)
  ax.set_title('Rollout Metrics', fontsize=18)
  ax.set_xlim(left=0, right=maxUpdates)
  ax.set_ylim(bottom=0, top=1)
  ax.legend(fontsize=10)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'train_loss_success.png')
    fig.savefig(figurePath)
    print("Saved {}".format(figurePath))
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()


def save_value_rollout_action_figure(agent, env, figureFolder, args, vmin, vmax,
                                     plotFigure=False, storeFigure=True):
  """Saves value, rollout, and action-distribution figure."""
  nx = 41
  ny = 121
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  resultMtx = np.empty((nx, ny), dtype=int)
  actDistMtx = np.empty((nx, ny), dtype=int)
  it = np.nditer(resultMtx, flags=['multi_index'])

  while not it.finished:
    idx = it.multi_index
    if idx[1] == 0:
      print("Evaluating grid row {}/{}".format(idx[0] + 1, nx), end='\r')
    x = xs[idx[0]]
    y = ys[idx[1]]

    state = env.append_time([x, y], 0.0) if env.time_varying else np.array([x, y])
    stateTensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
    action_index = agent.Q_network(stateTensor).min(dim=1)[1].cpu().item()
    actDistMtx[idx] = action_index

    _, _, result = env.simulate_one_trajectory(
        agent.Q_network, T=250, state=state, toEnd=False
    )
    resultMtx[idx] = result
    it.iternext()

  print_eval_summary(resultMtx.reshape(-1), prefix="Grid rollout")

  fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
  axStyle = env.get_axes()

  ax = axes[2]
  ax.imshow(
      actDistMtx.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap='seismic', vmin=0, vmax=env.action_space.n - 1, zorder=-1
  )
  ax.set_xlabel('Action', fontsize=24)

  ax = axes[1]
  ax.imshow(
      resultMtx.T != 1, interpolation='none', extent=axStyle[0],
      origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1
  )
  env.plot_trajectories(
      agent.Q_network, states=env.visual_initial_states, toEnd=True, ax=ax,
      c='w', lw=1.5
  )
  ax.set_xlabel('Rollout RA', fontsize=24)

  ax = axes[0]
  _, _, v = env.get_value(agent.Q_network, nx, ny, plot_time=0.0)
  ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
  )
  ax.contour(
      xs, ys, v.T, levels=[0], colors='k', linewidths=2, linestyles='dashed'
  )
  ax.set_xlabel('Value', fontsize=24)

  for ax in axes:
    env.plot_target_failure_set(ax=ax)
    if not env.time_varying:
      env.plot_reach_avoid_set(ax=ax)
    env.plot_formatting(ax=ax)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
    fig.savefig(figurePath)
    print("Saved {}".format(figurePath))
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

  return resultMtx, actDistMtx, v


def save_value_time_slices_figure(agent, env, figureFolder, vmin, vmax,
                                  plotFigure=False, storeFigure=True):
  """Saves learned value function slices across time for moving zones."""
  if not env.time_varying:
    return None

  nx = 61
  ny = 61
  timeSlices = np.arange(0.0, env.time_horizon + 1.0, 1.0).tolist()
  nSlices = len(timeSlices)
  fig, axes = plt.subplots(1, nSlices, figsize=(4 * nSlices, 4), sharex=True,
                          sharey=True)
  axStyle = env.get_axes()

  for ax, plotTime in zip(axes, timeSlices):
    xs, ys, valueMtx = env.get_value(
        agent.Q_network, nx=nx, ny=ny, plot_time=plotTime
    )
    im = ax.imshow(
        valueMtx.T, interpolation='none', extent=axStyle[0], origin='lower',
        cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
    )
    ax.contour(
        xs, ys, valueMtx.T, levels=[0], colors='k', linewidths=2,
        linestyles='dashed'
    )
    env.plot_target_failure_set(ax=ax, t=plotTime)
    env.plot_formatting(ax=ax)
    ax.set_title('V(x,t), t = {:.2f}'.format(plotTime), fontsize=16)

  fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01, fraction=0.04)
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'value_time_slices.png')
    fig.savefig(figurePath)
    print("Saved {}".format(figurePath))
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

# == CONFIGURATION ==
env_name = "zermelo_show-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
  print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
else:
  print("Using CPU: CUDA GPU is not available to PyTorch.")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = max(1, int(maxUpdates / updateTimes))
maxSteps = 250
storeFigure = args.storeFigure
plotFigure = args.plotFigure
if args.evalOnly and not storeFigure and not plotFigure:
  storeFigure = True
  print("Eval-only mode: enabling figure saving.")
print(
    "Default experiment setup: timeVarying={}, storeFigure={}, warmup={}, "
    "annealing={}, maxUpdates={}, checkPeriod={}, evalTraj={}".format(
        args.timeVarying, storeFigure, args.warmup, args.annealing,
        maxUpdates, args.checkPeriod, args.evalTraj
    )
)

if args.mode == 'lagrange':
  fn = args.name + '-' + args.doneType + '-' + args.costType
else:
  fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'naive', 'DDQN', args.mode, fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

if args.mode == 'lagrange':
  envMode = 'normal'
  agentMode = 'normal'
  GAMMA_END = args.gamma
  EPS_PERIOD = updatePeriod
  EPS_RESET_PERIOD = maxUpdates
elif args.mode == 'RA':
  envMode = 'RA'
  agentMode = 'RA'
  if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

if args.doneType == 'toEnd':
  sample_inside_obs = True
elif args.doneType == 'TF' or args.doneType == 'fail':
  sample_inside_obs = False

# == Environment ==
print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode=envMode, doneType=args.doneType,
    sample_inside_obs=sample_inside_obs, envType='show',
    time_varying=args.timeVarying, time_horizon=args.timeHorizon
)

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print(
    "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
        stateDim, actionNum
    )
)
print(env.discrete_controls)

env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)

env.set_seed(args.randomSeed)
print(
    "Cost type: {}, Margin scaling: {:.1f}, ".
    format(env.costType, env.scaling)
    + "Reward: {:.1f}, Penalty: {:.1f}".format(env.reward, env.penalty)
)

if plotFigure or storeFigure:
  nx, ny = 101, 101
  vmin = -1 * args.scaling
  vmax = 1 * args.scaling

  v = np.zeros((nx, ny))
  l_x = np.zeros((nx, ny))
  g_x = np.zeros((nx, ny))
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  it = np.nditer(v, flags=['multi_index'])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    state = env.append_time([x, y], 0.0) if env.time_varying else np.array([x, y])
    l_x[idx] = env.target_margin(state, t=0.0)
    g_x[idx] = env.safety_margin(state, t=0.0)

    v[idx] = np.maximum(l_x[idx], g_x[idx])
    it.iternext()

  axStyle = env.get_axes()

  fig, axes = plt.subplots(1, 3, figsize=(12, 6))

  ax = axes[0]
  im = ax.imshow(
      l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$\ell(x)$', fontsize=18)

  ax = axes[1]
  im = ax.imshow(
      g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$g(x)$', fontsize=18)

  ax = axes[2]
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  if not env.time_varying:
    env.plot_reach_avoid_set(ax)
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$v(x)$', fontsize=18)

  for ax in axes:
    env.plot_target_failure_set(ax=ax)
    env.plot_formatting(ax=ax)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

  if env.time_varying:
    timeSlices = np.arange(0.0, env.time_horizon + 1.0, 1.0).tolist()
    nSlices = len(timeSlices)
    fig, axes = plt.subplots(1, nSlices, figsize=(4 * nSlices, 4))
    axStyle = env.get_axes()
    for ax, plotTime in zip(axes, timeSlices):
      env.plot_target_failure_set(ax=ax, t=plotTime)
      env.plot_formatting(ax=ax)
      ax.axis(axStyle[0])
      ax.set_title('t = {:.2f}'.format(plotTime), fontsize=18)
    fig.tight_layout()
    if storeFigure:
      figurePath = os.path.join(figureFolder, 'time_varying_zones.png')
      fig.savefig(figurePath)
    if plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()
    save_zone_motion_figure(
        env, figureFolder, plotFigure=plotFigure, storeFigure=storeFigure
    )

# == Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100
)

# == AGENT ==
dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingle(
    CONFIG, actionNum, action_list, dimList=dimList, mode=agentMode,
    terminalType=args.terminalType
)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

if args.evalOnly:
  print("\n== Evaluation Only ==")
  modelFolder = os.path.join(outFolder, "model")
  loadStep = pick_model_step(modelFolder, args.loadStep)
  availableSteps = get_model_steps(modelFolder)
  print("Available checkpoints: {}".format(availableSteps))
  print("Loading checkpoint step: {}".format(loadStep))
  agent.restore(loadStep, outFolder)

  vmin = -1 * args.scaling
  vmax = 1 * args.scaling

  trainDict = load_training_history(outFolder)
  save_training_summary_figure(
      trainDict, figureFolder, maxUpdates, plotFigure=plotFigure,
      storeFigure=storeFigure
  )

  print("Evaluating {} random rollouts...".format(args.evalTraj))
  evalStates = np.array([
      env.sample_random_state(sample_inside_obs=False)
      for _ in range(args.evalTraj)
  ])
  detailedEval = evaluate_policy_detailed(
      env, agent.Q_network, evalStates, T=maxSteps
  )
  evalResults = detailedEval["results"]
  print_eval_summary(evalResults, prefix="Random rollout")
  randomMetrics = compute_rollout_metrics(evalResults, detailedEval["steps"])
  print_metrics(randomMetrics, prefix="Random safety/liveness metrics")

  resultMtx, actDistMtx, valueMtx = save_value_rollout_action_figure(
      agent, env, figureFolder, args, vmin, vmax, plotFigure=plotFigure,
      storeFigure=storeFigure
  )
  save_value_time_slices_figure(
      agent, env, figureFolder, vmin, vmax, plotFigure=plotFigure,
      storeFigure=storeFigure
  )
  gridMetrics = compute_rollout_metrics(resultMtx.reshape(-1))
  valueMetrics = compute_value_classification_metrics(valueMtx, resultMtx)
  print_metrics(gridMetrics, prefix="Grid safety/liveness metrics")
  print("\nValue classification metrics:")
  for key, value in valueMetrics.items():
    print("  {}: {:.3f}".format(key, value))
  save_metrics_summary_figure(
      gridMetrics, randomMetrics, valueMetrics, figureFolder,
      plotFigure=plotFigure, storeFigure=storeFigure
  )

  evalDict = {
      "loadStep": loadStep,
      "evalResults": evalResults,
      "detailedEval": detailedEval,
      "resultMtx": resultMtx,
      "actDistMtx": actDistMtx,
      "valueMtx": valueMtx,
      "randomMetrics": randomMetrics,
      "gridMetrics": gridMetrics,
      "valueMetrics": valueMetrics,
  }
  reportLines = [
      "Reach-Avoid RL Evaluation Report",
      "================================",
      "time_varying: {}".format(env.time_varying),
      "time_horizon: {:.3f}".format(env.time_horizon),
      "loaded_checkpoint: Q-{}.pth".format(loadStep),
      "",
      "Random rollout metrics:",
  ]
  for key, value in randomMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  reportLines += ["", "Grid rollout metrics:"]
  for key, value in gridMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  reportLines += ["", "Value classification metrics:"]
  for key, value in valueMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  write_end_report(os.path.join(outFolder, "eval_report-{}.txt".format(loadStep)),
                   reportLines)
  save_obj(evalDict, os.path.join(outFolder, "eval-{}".format(loadStep)))
  print("Saved evaluation data to {}".format(
      os.path.join(outFolder, "eval-{}.pkl".format(loadStep))
  ))
  print("Evaluation-only run finished. No training was performed.")
  raise SystemExit

if args.warmup:
  print("\n== Warmup Q ==")
  lossList = agent.initQ(
      env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
      vmax=vmax, plotFigure=plotFigure, storeFigure=storeFigure
  )

  if plotFigure or storeFigure:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    tmp = np.arange(500, args.warmupIter)
    # tmp = np.arange(args.warmupIter)
    ax.plot(tmp, lossList[tmp], 'b-')
    ax.set_xlabel('Iteration', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    plt.tight_layout()

    if storeFigure:
      figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
      fig.savefig(figurePath)
    if plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()

print("\n== Training Information ==")
vmin = -1 * args.scaling
vmax = 1 * args.scaling
checkPeriod = args.checkPeriod
trainRecords, trainProgress = agent.learn(
    env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
    doneTerminate=True, vmin=vmin, vmax=vmax, numRndTraj=args.evalTraj,
    checkPeriod=checkPeriod, outFolder=outFolder, plotFigure=plotFigure,
    storeFigure=storeFigure
)

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(outFolder, 'train')

print("\n== Training Summary ==")
print("Total gradient updates: {:d}".format(agent.cntUpdate))
print("Saved training data to {}.pkl".format(filePath))
if trainProgress.shape[0] > 0:
  bestIdx = np.argmax(trainProgress[:, 0])
  bestStep = (bestIdx + 1) * args.checkPeriod
  bestSuccess, bestFailure, bestUnfinish = trainProgress[bestIdx]
  lastSuccess, lastFailure, lastUnfinish = trainProgress[-1]
  print(
      "Best checkpoint: Q-{}.pth | success/failure/unfinished = "
      "{:.3f}/{:.3f}/{:.3f}".format(
          bestStep, bestSuccess, bestFailure, bestUnfinish
      )
  )
  print(
      "Last checkpoint metrics: success/failure/unfinished = "
      "{:.3f}/{:.3f}/{:.3f}".format(
          lastSuccess, lastFailure, lastUnfinish
      )
  )
else:
  print("No checkpoint evaluation was recorded. Try a smaller --checkPeriod.")

if plotFigure or storeFigure:
  save_training_summary_figure(
      trainDict, figureFolder, maxUpdates, plotFigure=plotFigure,
      storeFigure=storeFigure
  )

if trainProgress.shape[0] > 0:
  # Pick the checkpoint with the lowest failure rate; success breaks ties.
  bestIdx = np.lexsort((-trainProgress[:, 0], trainProgress[:, 1]))[0]
  bestStep = (bestIdx + 1) * args.checkPeriod
  bestMetrics = trainProgress[bestIdx]
  print(
      "We pick model Q-{} with success/failure/unfinished = "
      "{:.3f}/{:.3f}/{:.3f}".format(
          bestStep, bestMetrics[0], bestMetrics[1], bestMetrics[2]
      )
  )
  agent.restore(bestStep, outFolder)

  resultMtx, actDistMtx, valueMtx = save_value_rollout_action_figure(
      agent, env, figureFolder, args, vmin, vmax, plotFigure=plotFigure,
      storeFigure=storeFigure
  )
  save_value_time_slices_figure(
      agent, env, figureFolder, vmin, vmax, plotFigure=plotFigure,
      storeFigure=storeFigure
  )

  trainDict['resultMtx'] = resultMtx
  trainDict['actDistMtx'] = actDistMtx
  trainDict['valueMtx'] = valueMtx

  gridMetrics = compute_rollout_metrics(resultMtx.reshape(-1))
  valueMetrics = compute_value_classification_metrics(valueMtx, resultMtx)
  print_metrics(gridMetrics, prefix="Grid safety/liveness metrics")
  print("\nValue classification metrics:")
  for key, value in valueMetrics.items():
    print("  {}: {:.3f}".format(key, value))

  evalStates = np.array([
      env.sample_random_state(sample_inside_obs=False)
      for _ in range(args.evalTraj)
  ])
  detailedEval = evaluate_policy_detailed(
      env, agent.Q_network, evalStates, T=maxSteps
  )
  randomMetrics = compute_rollout_metrics(
      detailedEval["results"], detailedEval["steps"]
  )
  print_metrics(randomMetrics, prefix="Random safety/liveness metrics")
  save_metrics_summary_figure(
      gridMetrics, randomMetrics, valueMetrics, figureFolder,
      plotFigure=plotFigure, storeFigure=storeFigure
  )
  trainDict['randomEval'] = detailedEval
  trainDict['gridMetrics'] = gridMetrics
  trainDict['randomMetrics'] = randomMetrics
  trainDict['valueMetrics'] = valueMetrics

  reportLines = [
      "Reach-Avoid RL Experiment Report",
      "================================",
      "time_varying: {}".format(env.time_varying),
      "time_horizon: {:.3f}".format(env.time_horizon),
      "selected_checkpoint: Q-{}.pth".format(bestStep),
      "selected_checkpoint_success_failure_unfinished: {:.3f}/{:.3f}/{:.3f}".
      format(bestMetrics[0], bestMetrics[1], bestMetrics[2]),
      "",
      "Paper/code cross-check:",
      "- State includes time when --timeVarying is enabled: [x, y, t].",
      "- g(s,t) is positive inside unsafe/moving obstacle regions.",
      "- l(s,t) is negative inside the moving target region.",
      "- The reach-avoid Bellman backup uses controller minimization over actions.",
      "- The zero sublevel set V(x,t)<=0 is treated as the learned reach-avoid set.",
      "",
      "Grid rollout metrics:",
  ]
  for key, value in gridMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  reportLines += ["", "Random rollout metrics:"]
  for key, value in randomMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  reportLines += ["", "Value classification metrics:"]
  for key, value in valueMetrics.items():
    reportLines.append("  {}: {}".format(key, value))
  reportLines += [
      "",
      "Suggested success criteria:",
      "- Safety rate high, failure rate preferably below 5% for a demo.",
      "- Liveness success high among non-failed rollouts.",
      "- Predicted-safe failure rate close to 0; this is the important safety check.",
      "- value_time_slices.png should show V(x,t)<=0 moving consistently with T(t), O(t).",
  ]
  write_end_report(os.path.join(outFolder, "end_report.txt"), reportLines)
else:
  print("Skipping final figures/report because no checkpoint metrics exist.")

save_obj(trainDict, filePath)
