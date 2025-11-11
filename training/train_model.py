import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
import subprocess
import threading
import yaml
import glob
import time
import argparse
import os

CONFIG_FILE = "config/poca/SoccerTwos.yaml"
ENV_FILE = "env/SoccerTwos/UnityEnvironment.exe"
TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Environment/Episode Length":"Episode Length",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Self-play/ELO":"ELO"
}


def get_hardware_metrics(stop_event,process,interval=1):
    cpu_values = []
    ram_values = []
    start_time = time.time()
    p = psutil.Process(process.pid)
    while not stop_event.is_set():
        cpu_percentage = p.cpu_percent(interval=interval)
        ram_usage = p.memory_info().rss / (1024 * 1024)
        cpu_values.append(cpu_percentage)
        ram_values.append(ram_usage)
    time_elapsed = time.time() - start_time
    mean_cpu = sum(cpu_values) / len(cpu_values)
    mean_ram = sum(ram_values) / len(ram_values)
    return mean_cpu, mean_ram, time_elapsed

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
    return metrics,total_steps

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

    config_datas = {}
    try:
        with open(config_path, "r") as f:
            config_datas = yaml.safe_load(f)
    except FileNotFoundError:
        print("[ERROR] Config file not found")

    start_time = time.time()
    stop_event = threading.Event()
    process = None
    hardware_spec_thread = None
    result = {}

    try:
        cmd = [
            "mlagents-learn ",
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

    total_time = time.time() - start_time
    mean_cpu, mean_ram, time_elapsed = result["hardware_metrics"]

    tensor_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","results"))
    metrics,total_steps = get_training_metrics(tensor_data_path,run_id)
    print(metrics)
    print(total_steps)
    print(f"CPU Usage (%): {mean_cpu}")
    print(f"RAM Usage (MB): {mean_ram}")

if __name__ == "__main__":
    main()
