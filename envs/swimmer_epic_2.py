import gym
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files


import gym.envs.mujoco.assets
import numpy as np
import wandb


def modify_swimmer_xml(viscosity: float = 0.1, density1: float = 0.1, density2: float = 0.1, density3: float = 0.1):
    """
    Modify the XML of the swimmer environment sth the viscosity is as specified, then return
    a string path to a tmpfile containing the modified xml.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmpfile:
        with files(gym.envs.mujoco.assets).joinpath("swimmer.xml").open("r") as infile:
            tree = ET.parse(infile)
            tree.find(".//*[@viscosity]").attrib["viscosity"] = str(viscosity)

            for el, density in zip(tree.findall(".//geom[@type='capsule']"), [density1, density2, density3]):
                el.attrib["density"] = str(density)

            tree.write(tmpfile)

    return tmpfile.name

class ObservationFloatWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32, copy=False)


def make_swimmer(seed: int):
    rng = np.random.default_rng(seed)

    viscosity = rng.uniform(0.08, 0.15)
    density1 = rng.uniform(100, 2000)
    density2 = rng.uniform(100, 2000)
    density3 = rng.uniform(100, 2000)

    modified_xml = modify_swimmer_xml(viscosity=viscosity, density1=density1, density2=density2, density3=density3)

    env = ObservationFloatWrapper(gym.make("Swimmer-v3", xml_file=modified_xml))

    return env
