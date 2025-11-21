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
def handle_GIT():
    global run_id
    try:
        time.sleep(2)  # pause to ensure file writes complete

        files_to_add = []
        for root, dirs, files in os.walk(path_to_data):
            for file in files:
                if f"{run_id}" in file: 
                    abs_file_path = os.path.join(root, file)
                    rel_file_path = os.path.relpath(abs_file_path, path_to_repo)
                    if os.path.exists(abs_file_path):
                        print(f"Adding file: {rel_file_path}")
                        files_to_add.append(rel_file_path)

        if files_to_add:
            repo.index.add(files_to_add)

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
    global run_id,starttime
    run_id = file_counter() + 1
    starttime = datetime.datetime.now()    

    try:
        while True:
            print("Starting run...")
            p = subprocess.Popen(['conda','run','--no-capture-output','-n','mlagents','python','-u',path_to_training,'--run-id',str(run_id)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            while True:
                #reads = [p.stdout.fileno()]
                #ret = select.select(reads,[],[])
                #if p.stdout.fileno() in ret[0]:
                    #line = p.stdout.readline()
                    #if line:
                        #print(line,end='')
                    #else:
                        #break

            
                if p.poll() is not None:
                    break

            
                if time_elapsed():
                    handle_SIGINT()
                    time.sleep(5)
                    handle_GIT()
                    print("Restarting due to time elapsed...")
                    break

                       
                time.sleep(1)

            run_id = file_counter() + 1
        


    except KeyboardInterrupt:
        handle_SIGINT()

def time_elapsed():

    now = datetime.datetime.now()
    elapsed = now - starttime
    print(f"Checking elapsed time: {elapsed}")
    return elapsed >= datetime.timedelta(hours=5)
    #return elapsed >= datetime.timedelta(minutes=5)

def main():
    handle_STARTUP()

if __name__ == '__main__':
    main()
