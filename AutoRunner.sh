#!/bin/bash

control_c(){
    	pkill mlagents-learn
	if [ -n "$PID" ]; then
		kill $PID
	fi
	exit
}

trap control_c SIGINT

while true; do 
	

	run_id=$(($(find data/server -type f | wc -l) + 1))

	timeout --foreground 5h bash -c "
			trap 'exit' SIGINT
			conda run --no-capture-output -n mlagents python -u training/linux_train_model.py --run-id $run_id" 

	pkill mlagents-learn # Killing ml agents process 

	if [[ $? -eq 0 ]] ; then # Checks if process is finished, commit logic
		git add data/server/*.json

		if ! git diff --cached --quiet ; then 
			git commit -m "Server Auto-Commit : Added training results from run #$run_id"
			git pull --rebase
			git push
			echo "Commited results for run_id $run_id after 5 hours"
		else 
			echo "No changes to commit for run_id $run_id"
		fi
	fi
	sleep 10
done
