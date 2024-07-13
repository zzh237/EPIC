# Code for the Paper
This code requires Python 3.8
- The required codes(.py files) for this paper are:
```
Code
│   README.md
│   maml.py
|   meta.py
│   ...
|   requirements.txt
|   run.sh
|   ...
└───algos
│   │   memory.py
|   └───agents    
|   |   |    core.py
│   │   |    vpg.py
|   |   |    updates.py
|   |   |    ...
|   |   
│   │   
│   └───attackers
│       │   ....
│       │   ....
│   
└───envs
|   |   new_cartpole.py
|   │   new_lunar_lander.py
|   │   swimmer_rand_vel.py
|   |   ...
│   

```

- How to run:
  * A sample bash run script is provided as "run.sh", please check. 
  * Basically, it runs the "meta.py", where it has an entry point main function of our algorithm. 
  * To run a different algorithm, for example, MAML, you can run "maml.py"
  * The results will be saved under the results folder. 
  * Some codes are not used for future research. 
