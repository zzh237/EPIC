import numpy as np
from envs.new_lunar_lander import NewLunarLander

def make_lunar_env(seed, continuous=False):   
    rng = np.random.default_rng(seed)
    main_engine_power = rng.choice(np.array([11.0, 12.0, 13.0, 14.0, 15.0]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
    side_engine_power = rng.choice(np.array([0.45, 0.55, 0.65, 0.75, 0.85]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
    main_engine_power = main_engine_power + 0.1 * rng.normal()
    side_engine_power = side_engine_power + 0.01 * rng.normal()
    env = NewLunarLander(
        main_engine_power=main_engine_power, side_engine_power=side_engine_power, continuous=continuous
    )
    return env