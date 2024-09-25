import gym
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files


import gym.envs.mujoco.assets
import numpy as np


def modify_swimmer_xml(viscosity: float = 0.1, mass1: float = 0.1, mass2: float = 0.1, mass3: float = 0.1):
    """
    Modify the XML of the swimmer environment sth the viscosity is as specified, then return
    a string path to a tmpfile containing the modified xml.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        with files(gym.envs.mujoco.assets).joinpath("swimmer.xml").open("r") as infile:
            tree = ET.parse(infile)
            tree.find(".//*[@viscosity]").attrib["viscosity"] = str(viscosity)

            for el, mass in zip(tree.findall(".//geom[@type='capsule']"), [mass1, mass2, mass3]):
                el.attrib["mass"] = str(mass)

            tree.write(tmpfile)

    return tmpfile.name


def make_swimmer(seed: int):
    rng = np.random.default_rng(seed)

    viscosity = rng.uniform(0.001, 0.2)
    mass1 = rng.uniform(0.001, 0.3)
    mass2 = rng.uniform(0.001, 0.3)
    mass3 = rng.uniform(0.001, 0.3)

    env = gym.make(
        "Swimmer-v3", xml_file=modify_swimmer_xml(viscosity=viscosity, mass1=mass1, mass2=mass2, mass3=mass3)
    )

    return env
