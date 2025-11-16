# Use pgrep -l mlagents - DONE
# Use subprocess
# Use os.kill SIGINT

import subprocess
import os
import signal
import datetime
import time
import select
from git import Repo
import pytz
import sys
import signal

run_id = 0
path_to_repo = os.path.expanduser('~/Group-9-AI-ML')
path_to_data = os.path.expanduser('~/Group-9-AI-ML/data/server')
timezone = pytz.timezone('Europe/Amsterdam')
ct = datetime.datetime.now(timezone)
repo = Repo(path_to_repo)
path_to_training = 'training/linux_train_model.py'
starttime = None
sigint_input_flag = False

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
    global run_id
    try:
        time.sleep(2)  # Pause 2 seconds to ensure file writes complete

        for root, dirs, files in os.walk(path_to_data):
            for file in files:
                abs_file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(abs_file_path, path_to_repo)
                if os.path.exists(abs_file_path):
                    print(f"Adding file: {rel_file_path}")
                    repo.index.add([rel_file_path])
                else:
                    print(f"Missing file (skipped): {abs_file_path}")

        if repo.is_dirty():
            repo.index.commit(f"Server Auto-Commit: Added data from run number #{run_id}")
            origin = repo.remote(name='origin')
            print("Pushing to branch")
            origin.push()
        else:
            print("No changes detected; no commit made.")
    except Exception as e:
        print(f"Error during git handling: {e}")

def sigint_handler(signum,frame):
    global sigint_input_flag
    print("SIGINT recieved, setting interrupt flag")
    sigint_input_flag = True




# Method to handle conda activate start up and ml agents learn
def handle_STARTUP():
    global run_id
    run_id = file_counter() + 1

    try:
        while True:
            print("Starting run...")
            global starttime
            starttime = datetime.datetime.now()
        
            while True:
                p = subprocess.Popen(['conda','run','--no-capture-output','-n','mlagents','pyhton','-u',path_to_training,'--run-id',str(run_id)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                #for line in iter(p.stdout.readline,''):
                #    print(line,end="")
                #p.wait()
            
                if p.poll() is not None:
                    break

            
                if time_elapsed():
                    handle_SIGINT()
                    time.sleep(5)
                    handle_GIT()
                    print("Restarting due to time elapsed...")
                    break

                if stop_time_check():
                    handle_SIGINT()
                    time.sleep(5)
                    handle_GIT()
                    print("Restarting due to stop time being reached(23:59 UTC+2)")
                    break
            
                time.sleep(1)

            run_id = file_counter() + 1
        


    except KeyboardInterrupt:
        handle_SIGINT()


def stop_time_check():
    now = datetime.datetime.now()
    #print(f"Current time for stop check:{now}")
    return now.hour == 23 and now.minute == 59


def time_elapsed():

    now = datetime.datetime.now()
    elapsed = now - starttime
    #print(f"Checking elapsed time: {elapsed}")
    return elapsed >= datetime.timedelta(hours=1,minutes=30)

def main():
    handle_STARTUP()

if __name__ == '__main__':
    main()
