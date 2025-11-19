# How to start training?
## Setup environment
Setup virtual environment and install all required libs
```commandline
python -m venv .ml-agents-venv (I recommend to use version 3.10.1)
```
```commandline
.ml-agents-venv/Scripts/activate (for windows)
source .ml-agents-venv/bin/activate (for mac)
```
```commandline
pip install "torch==2.1.1+cu118" "torchvision==0.16.1+cu118" "torchaudio==2.1.1+cu118" -f https://download.pytorch.org/whl/torch_stable.html (for windows / linux)
pip install torch torchvision torchaudio (for mac)
```
``` commandline
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```

## Start training

### Env: SoccerTwos
Open terminal and run this command
```commandline
python training/train_SCT.py --run-id [run id]
```
Argument
- --run-id: run id
- --command: "resume" or "force" (default: force)
- --base-port: set different port to run multi-processes (default: 5005)

### Env: StrikersVsGoalie
Open terminal and run this command
```commandline
python training/train_SKG.py --run-id [run id]
```
Argument
- --run-id: run id
- --command: "resume" or "force" (default: force)
- --base-port: set different port to run multi-processes (default: 5005)
