# Project overview

## Purpose

This repository trains a discrete-action DDQN to estimate reach-avoid values for a 2D point-mass environment. The time-varying mode augments observations with time, so the policy can condition its actions on the current target and obstacle geometry.

## Runtime flow

1. `sim_naive.py` parses training or evaluation arguments and registers the local Gym environment through `gym_reachability`.
2. `ZermeloShowEnv` produces a state, applies one of three horizontal controls, advances the dynamics by `0.05`, and computes target/safety margins.
3. `DDQNSingle` selects the minimum-Q action with epsilon-greedy exploration, stores transitions, and learns from `ReplayMemory`.
4. The online `Model` selects the next action and the target `Model` estimates its value. The DRABE backup combines the target margin, safety margin, discount factor, and bootstrap value.
5. Training periodically saves `Q-<step>.pth` checkpoints and plots. Evaluation produces rollout metrics, value-set diagnostics, and an `eval_report-<step>.txt`.

## Code-to-thesis alignment

The thesis describes the broader reach-avoid formulation and two experimental map families. The current command-line path is the `show` environment: two thin horizontal obstacle bands, a moving target, and a time-augmented observation when time variation is enabled. `basic` and `easy` map definitions remain in the environment class but are not selectable from `sim_naive.py` as shipped.

## Publication boundary

The public repository should contain source, direct dependencies, documentation, and selected lightweight figures. It should not contain virtual environments, serialized replay/training histories, model checkpoints, local research PDFs, secrets, or unverified proprietary datasets. Those rules are encoded in `.gitignore`.

## Verification status

- Python source: syntax-checked with `python -m compileall`.
- CLI runtime: not claimed as reproduced in the capture environment because `gym` was not installed there.
- Thesis: inspected as background material; its claims and reported metrics are not silently converted into executable-code benchmarks.
