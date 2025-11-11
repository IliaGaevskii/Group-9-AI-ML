# How to start training?
## Setup environment
Setup virtual environment and install all required libs
```commandline
python -m venv .ml-agents-venv
.ml-agents-venv/Scripts/activate (for windows)
source venv/bin/activat (for mac)
pip install ml_agents_requirements.txt
```

## Start training
Open terminal and run this command
```commandline
python training/train_model.py --run-id [run id]
```
Argument
- --config: set yaml file for environment (default: SoccerTwos)
- --run-id: run id
- --command: "resume" or "force" (default: force) 
- --env: "environment" (default: SoccerTwos) 