from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import psutil
import numpy as np
import subprocess
import threading
import yaml
import glob
import time
import argparse
import os
import json
import socket

CONFIG_FILE = "config/poca/SoccerTwos.yaml"
ENV_FILE = "env/SoccerTwos/UnityEnvironments.x86_64"
TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Environment/Episode Length":"Episode Length",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Self-play/ELO":"ELO"
}

def get_hardware_metrics(stop_event, process, interval=1):
    cpu_values = []
    ram_values = []
    start_time = time.time()

    try:
        p = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        print("[WARNING] Process does not exist at the start of metrics collection")
        return None, None, 0

    while not stop_event.is_set():
        try:
            # Check if process is still running
            if process.poll() is not None:  # Process finished
                break

            # Use non-blocking call with small sleep for responsiveness
            cpu_percentage = p.cpu_percent(interval=None)  # non-blocking instant value
            ram_usage = p.memory_info().rss / (1024 * 1024)  # in MB

            cpu_values.append(cpu_percentage)
            ram_values.append(ram_usage)

            time.sleep(interval)  # sleep outside cpu_percent to avoid delayed exit
        except psutil.NoSuchProcess:
            print("[WARNING] Process ended during metric collection")
            break

    time_elapsed = time.time() - start_time

    mean_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
    mean_ram = sum(ram_values) / len(ram_values) if ram_values else 0

    return mean_cpu, mean_ram, time_elapsed


def find_latest_events(path, run_id, recursive=True):
    if run_id:
        base_dir = os.path.join(path, run_id)
    else:
        base_dir = path

    pattern = '**/events.out.tfevents.*' if recursive else 'events.out.tfevents.*'
    search_path = os.path.join(base_dir, pattern)

    tfevents_files = glob.glob(search_path, recursive=recursive)
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
        if tag == "Environment/Cumulative Reward":
            metrics["Cumulative Reward"] = values[-1]
    return metrics,total_steps

def save_data(run_id,metrics,total_steps,cpu_usage,ram_usage,total_time,config_data):
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
    data["mean_reward"] = metrics["Mean Policy Reward"]
    data["cumulative_reward"] = metrics["Cumulative Reward"]
    data["episode_length"] = metrics["Episode Length"]
    data["mean_policy_loss"] = metrics["Mean Policy Loss"]
    data["mean_value_loss"] = metrics["Mean Value Loss"]
    data["mean_entropy"] = metrics["Mean Entropy"]
    data["ELO"] = metrics["ELO"]
    data["training_time"] = total_time
    data["mean_cpu_usage"] = cpu_usage
    data["mean_ram_usage"] = ram_usage
    data["efficiency_score"] = data["mean_reward"]/total_time
    create_json_file(data)
    return data

def create_json_file(data):
    hostname = 'server'
    run_id = data["run_id"]
    file_name = f"{run_id}-{time.time()}.json"
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
    parser.add_argument("--config", default=CONFIG_FILE)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--command",default="force")
    parser.add_argument("--env",default=ENV_FILE)
    args = parser.parse_args()
    run_id = args.run_id
    config_path = args.config
    env_path = args.env
    command = args.command

    print(f"[INFO] Start training ML-Agents")
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Config: {config_path}")

    start_time = time.time()
    stop_event = threading.Event()
    process = None
    hardware_spec_thread = None
    result = {}

    try:
        cmd = [
            "mlagents-learn",
            config_path,
            f"--run-id={run_id}",
            "--torch-device=cpu",
            f"--{command}",
            "--no-graphics",
            f"--env={env_path}"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        def worker():
            result["hardware_metrics"] = get_hardware_metrics(stop_event,process)
        hardware_spec_thread = threading.Thread(target=worker)
        hardware_spec_thread.start()
        for line in iter(process.stdout.readline, ''):
            print(line,end="")
        process.wait()

    except KeyboardInterrupt:
        if process:
            stop_event.set()
            if hardware_spec_thread:
                hardware_spec_thread.join()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[INFO] Training Shutdown")
    finally:

        total_time = time.time() - start_time
        hardware_spec_thread.join()
        try:
            mean_cpu, mean_ram, time_elapsed = result["hardware_metrics"]
        except KeyError:
            mean_cpu, mean_ram, time_elapsed = (None, None, None)
            print("[WARNING] No hardware metrics found")
        config_datas = get_config_datas(config_path)

        tensor_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","results"))
        metrics,total_steps = get_training_metrics(tensor_data_path,run_id)

        data = save_data(run_id,metrics,total_steps,mean_cpu,mean_ram,total_time,config_datas)

if __name__ == "__main__":
    main()
