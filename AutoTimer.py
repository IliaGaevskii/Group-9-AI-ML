# Use pgrep -l mlagents
# Use subprocess
# Use os.kill SIGINT

import subprocess
import os
import signal
import time
import git
from git import Repo, Commit

path_to_repo = os.getcwd()
run_id = 0

# Method designed to send ^C SIGINT to mlagents process
def handle_SIGINT():

    result = subprocess.run(['pgrep','-l','mlagents'],stdout = subprocess.PIPE, text=True)

    lines = result.stdout.strip().split('\n')

    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit():
            pid = int(parts[0])
            os.kill(pid,signal.SIGINT)

# Method to handle commiting new .json and run_log data
#def handle_GIT():




#def handle_STARTUP():
    #subprocess.run(['conda','activate','mlagents'])






def main():
    repo = Repo(path_to_repo)
    handle_SIGINT()

    print(repo.git.status())


if __name__ == '__main__':
    main()
