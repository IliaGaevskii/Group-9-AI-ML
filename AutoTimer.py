# Use pgrep -l mlagents - DONE
# Use subprocess
# Use os.kill SIGINT

import subprocess
import os
import signal
import datetime
import time

from git import Repo
import pytz
import sys

path_to_repo = '~/Group-9-AI-ML'
run_id = 0
path_to_data = 'data/server'
timezone = pytz.timezone('Europe/Amsterdam')
ct = datetime.datetime.now(timezone)
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

def file_counter(init_run_id=0):
    for _,_,files in os.walk(path_to_data):
        init_run_id += len(files)
    return init_run_id

# Method to handle commiting new .json and run_log data
# TODO: Make method count how many runs are in the result folder and increment run counter based on that
def handle_GIT():
    try:
        for root, dirs, files in os.walk(path_to_data):
            for file in files:
                filepath = os.path.relpath(os.path.join(root, file), path_to_repo)
                repo.index.add([filepath])
        repo.index.commit("Server Auto-Commit: Added data from run number #" + str(run_id))
        origin = repo.remote(name='origin')
        origin.push()
    except Exception as e:
        print(f"Error: {e}")




# Method to handle conda activate start up and ml agents learn
def handle_STARTUP():
    run_id = file_counter() + 1

    try:
        print("Starting run...")
        p = subprocess.Popen(['conda','run','--no-capture-output','-n','mlagents','python','-u',path_to_training,'--run-id',str(run_id)], stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in iter(p.stdout.readline,''):
           print(line,end="")
        p.wait()
        if stop_time_check():
           return
    except KeyboardInterrupt:
        handle_SIGINT()


def stop_time_check():
    now = datetime.datetime.now()
    return now.hour == 23 and now.minute == 59


def main():

    handle_STARTUP()
    try:
        while True:
            if stop_time_check():
                handle_SIGINT()
                handle_GIT()
                handle_STARTUP()

                time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping run...")

if __name__ == '__main__':
    main()
