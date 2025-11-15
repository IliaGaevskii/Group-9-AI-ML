# How to start training?
## Setup environment
Setup virtual environment and install all required libs
```commandline
python -m venv .ml-agents-venv
.ml-agents-venv/Scripts/activate (for windows)
source .ml-agents-venv/bin/activate (for mac)
pip3 install torch~=1.13.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
pip install psutil
```

## Start training
Open terminal and run this command
```commandline
python train_model.py --run-id [run id]
```
Argument
- --config: set yaml file for environment (default: SoccerTwos)
- --run-id: run id
- --command: "resume" or "force" (default: force) 
- --env: "environment" (default: SoccerTwos) 
