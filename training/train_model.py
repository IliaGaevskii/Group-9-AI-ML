
import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
import subprocess
import threading
import os
import yaml
import csv
import glob
import time
import argparse

CONFIG_FILE = "config/poca/SoccerTwos.yaml"
TOTAL_STEPS = 500000
TAGS = {
    "Environment/Cumulative Reward": "Mean Policy Reward",
    "Environment/Episode Length":"Episode Length",
    "Losses/Policy Loss": "Mean Policy Loss",
    "Losses/Value Loss": "Mean Value Loss",
    "Policy/Entropy": "Mean Entropy",
    "Self-play":"ELO"
}


def get_hardware_metrics(stop_event,interval=1):
    cpu_values = []
    ram_values = []
    start_time = time.time()
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu_percentage = process.cpu_percent(interval=interval)
        ram_usage = process.memory_info().rss / (1024 * 1024)
        cpu_values.append(cpu_percentage)
        ram_values.append(ram_usage)
    time_elapsed = time.time() - start_time
    mean_cpu = sum(cpu_values) / len(cpu_values)
    mean_ram = sum(ram_values) / len(ram_values)
    print(f"CPU Usage (%): {mean_cpu}")
    print(f"RAM Usage (MB): {mean_ram}")
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
    metrics = {"run id": run_id}
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
        values = np.array([e.value for e in event])
        metrics[label] = np.mean(values)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_FILE)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--num-steps",default=TOTAL_STEPS)
    parser.add_argument("--command",default="force")
    args = parser.parse_args()
    run_id = args.run_id
    num_steps = args.num_steps
    config_path = args.config
    command = "--" + args.command

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
        process = subprocess.Popen(
            ["mlagents-learn",config_path,"--run-id",run_id,"--torch-device","cpu",command,"--no-graphics"],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True
        )

        def worker():
            result["hardware_metrics"] = get_hardware_metrics(stop_event)
        hardware_spec_thread = threading.Thread(target=worker)
        hardware_spec_thread.start()

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
        print("[INFO] Training Shutdown")

    stop_event.set()
    if hardware_spec_thread:
        hardware_spec_thread.join()
    total_time = time.time() - start_time
    mean_cpu, mean_ram, time_elapsed = result["hardware_metrics"]

    tensor_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","results"))
    metrics = get_training_metrics(tensor_data_path,run_id)
    print(metrics)

if __name__ == "__main__":
    main()
