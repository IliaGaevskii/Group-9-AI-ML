# Use pgrep -l mlagents - DONE
# Use subprocess
# Use os.kill SIGINT

import subprocess
import os
import signal
import datetime
from git import Repo
import pytz

path_to_repo = os.getcwd()
run_id = 0
path_to_data = 'data/server'
timezone = pytz.timezone('Europe/Amsterdam')
ct = datetime.now(timezone)
repo = Repo(path_to_repo)
path_to_training = 'training/linux_train_model.py'

# Method designed to send ^C SIGINT to mlagents process
def handle_SIGINT():

    result = subprocess.run(['pgrep','-l','mlagents'],stdout = subprocess.PIPE, text=True)

    lines = result.stdout.strip().split('\n')

    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit():
            pid = int(parts[0])
            os.kill(pid,signal.SIGTERM)

def file_counter(run_id=0):
    for _,_,files in os.walk(path_to_data):
        run_id += len(files)
    return run_id

# Method to handle commiting new .json and run_log data
# TODO: Make method count how many runs are in the result folder and increment run counter based on that
def handle_GIT():
    try:
        for root, dirs, files in os.walk(path_to_data):
            for file in files:
                filepath = os.path.relpath(os.path.join(root, file), path_to_repo)
                repo.index.add([filepath])

        repo.index.commit("Server Auto-Commit: Added data from run number #" + str(run_id))
    except Exception as e:
        print(f"Error: {e}")




# Method to handle conda activate start up and ml agents learn
def handle_STARTUP():
    file_counter()
    subprocess.run(['conda','run','-n','mlagents','python',path_to_training,'--run-id',str(run_id+1)],check=True)



def stop_time_check():
    now = datetime.datetime.now()
    return now.hour == 23 and now.minute == 59


def main():

    handle_SIGINT()

    if stop_time_check():
        handle_SIGINT() # Stops ml-agents learn execute
        handle_GIT() # Commits and pushes results to repo
        handle_STARTUP() # Restarts execution

    #print(repo.git.status())


if __name__ == '__main__':
    main()
