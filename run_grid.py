"""
Run a grid of epic_mc experiments with warm restarting.
"""
import concurrent.futures
from itertools import product
from pathlib import Path
import subprocess
import concurrent

PARALLELISM = 5
PREFIX = Path("./results/montecarlo")
ENV = "jbw"
DEVICE = "cpu"

GRID = {
    "meta_update_every": [10, 25, 50],
    "steps": [300],
    "m": [1, 5, 15]
}

ALL_PARAMS = [dict(zip(GRID.keys(), point)) for point in product(*GRID.values())]

def calculate_run_prefix(point: dict):
    resdir = PREFIX / ENV / f"n{point['meta_update_every']}_s{point['steps']}_m{point['m']}"
    return resdir

# to support restarting, we calculate the full grid of files that would be produced
# and create a run for each filename. if the file already exists, then we skip performing 
# the run.
def create_subprocess_command(point: dict):
    return [
        "python", 
        "epic_mc.py", 
        f"--env={ENV}",
        f"--device={DEVICE}",
        "--meta_update_every", str(point["meta_update_every"]),
        "--steps", str(point['steps']),
        "--m", str(point['m']),
        "--resdir", calculate_run_prefix(point).resolve()
    ]

def execute_subprocess_command(point: dict):
    run_prefix = calculate_run_prefix(point)
    if run_prefix.exists():
        return
    cmd = create_subprocess_command(point)
    subprocess.check_output(cmd)


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLELISM) as executor:
        futures = executor.map(execute_subprocess_command, ALL_PARAMS)

        concurrent.futures.wait(futures, timeout=None)

        

