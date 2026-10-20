from gym.envs.registration import register
register(
    id="zermelo_show-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloShowEnv"
)
