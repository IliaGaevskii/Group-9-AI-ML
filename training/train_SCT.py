from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import subprocess
import yaml
import glob
import time
import argparse
import os
import json
import socket

CONFIG_FILE = "config/poca/SoccerTwos.yaml"
ENV_FILE = "env/SoccerTwos/UnityEnvironment.exe"
TAGS = {
    "Environment/Group Cumulative Reward": "Mean Group Reward",
    "Environment/Cumulative Reward": "Mean Reward",
    "Environment/Episode Length":"Episode Length",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Self-play/ELO":"ELO"
}

def find_latest_events(path,run_id, recursive = True):
    pattern = '**/events.out.tfevents.*' if recursive else 'events.out.tfevents*'
    search_path = os.path.join(path,run_id,pattern)
    tfevents_files = glob.glob(search_path,recursive=recursive)

    if not tfevents_files:
        print("[ERROR] No tensorboard log files found!")
        return None

    latest_event = max(tfevents_files, key=os.path.getctime)
    return latest_event

def get_training_metrics(path,run_id):
    tfevent = find_latest_events(path,run_id)
    event_acc = EventAccumulator(tfevent)
    event_acc.Reload()
    #print(event_acc.scalars.Keys())
    metrics = {"run id": run_id}
    total_steps = 0
    for tag,label in TAGS.items():
        try:
            event = event_acc.Scalars(tag)
        except KeyError:
            print(f"[ERROR] No tensorboard log file for tag {tag}!")
            metrics[label] = "N/A"
            continue
        if not event:
            metrics[label] = "N/A"
            continue
        steps = np.array([e.step for e in event])
        total_steps = steps[-1]
        values = np.array([e.value for e in event])
        metrics[label] = np.mean(values)
        if tag == "Environment/Group Cumulative Reward":
            metrics["Group Cumulative Reward"] = values[-1]
    return metrics,total_steps

def save_data(run_id,metrics,total_steps,total_time,config_data):
    data = {}
    data['run_id'] = run_id
    data["env_name"] = "SoccerTwos"
    data["algorithm"] = config_data["behaviors"]["SoccerTwos"]["trainer_type"]
    data["total_steps"] = int(total_steps)
    data["learning_rate"] = config_data["behaviors"]["SoccerTwos"]["hyperparameters"]["learning_rate"]
    data["epoch"] = config_data["behaviors"]["SoccerTwos"]["hyperparameters"]["num_epoch"]
    data["bath_size"] = config_data["behaviors"]["SoccerTwos"]["hyperparameters"]["batch_size"]
    data["buffer_size"] = config_data["behaviors"]["SoccerTwos"]["hyperparameters"]["buffer_size"]
    data["gamma"] = config_data["behaviors"]["SoccerTwos"]["reward_signals"]["extrinsic"]["gamma"]
    data["lambda"] = config_data["behaviors"]["SoccerTwos"]["hyperparameters"]["lambd"]
    data["mean_group_reward"] = metrics["Mean Group Reward"]
    data["group_cumulative_reward"] = metrics["Group Cumulative Reward"]
    data["episode_length"] = metrics["Episode Length"]
    data["mean_policy_loss"] = metrics["Mean Policy Loss"]
    data["mean_value_loss"] = metrics["Mean Value Loss"]
    data["mean_entropy"] = metrics["Mean Entropy"]
    data["ELO"] = metrics["ELO"]
    data["training_time"] = total_time
    data["efficiency_score"] = data["mean_group_reward"]/total_time
    create_json_file(data)
    return data

def create_json_file(data):
    hostname = socket.gethostname()
    run_id = data["run_id"]
    file_name = f"{run_id}.json"
    os.makedirs(f"data/{hostname}",exist_ok=True)
    print("[INFO] Saving data...")
    with open(f"data/{hostname}/{file_name}",'w') as f:
        json.dump(data,f,indent=4)
    print("[INFO] Saving success!")

def get_config_datas(config_path):
    config_datas = {}
    try:
        with open(config_path, "r") as f:
            config_datas = yaml.safe_load(f)
    except FileNotFoundError:
        print("[ERROR] Config file not found")
    return config_datas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--command",default="force")
    parser.add_argument("--base-port",default=5005)
    args = parser.parse_args()
    run_id = args.run_id
    command = args.command
    port = args.base_port

    print(f"[INFO] Start training ML-Agents")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Config: {CONFIG_FILE}")

    start_time = time.time()
    process = None

    try:
        cmd = [
            "mlagents-learn ",
            CONFIG_FILE,
            f"--run-id={run_id}",
            # "--torch-device=cpu", # activate this if you want to use cpu instead of gpu
            f"--{command}",
            "--no-graphics",
            f"--env={ENV_FILE}",
            f"--base-port={port}",
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in iter(process.stdout.readline, ''):
            print(line,end="")
        process.wait()

    except KeyboardInterrupt:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    finally:
        print("[INFO] Training Shutdown")

    total_time = time.time() - start_time
    config_datas = get_config_datas(CONFIG_FILE)

    tensor_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","results"))
    metrics,total_steps = get_training_metrics(tensor_data_path,run_id)

    data = save_data(run_id,metrics,total_steps,total_time,config_datas)

if __name__ == "__main__":
    main()
