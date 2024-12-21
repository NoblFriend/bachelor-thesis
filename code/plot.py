# ==== Argparse ====
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_names', type=str, nargs='+')
parser.add_argument('--log_dir', type=str, default="../logs")
parser.add_argument('--plot_dir', type=str, default="../plots")
args = parser.parse_args()

# ==== Load data ====
from log.log import DataLogger
loggers = {}

for run_name in args.run_names:
    logger = DataLogger(
        name=run_name,
        hyperparams=None,
        log_dir=args.log_dir,
    )
    logger.load()
    print(logger.hyperparams)
    loggers[run_name] = logger

# ==== Plot metrics ====
import matplotlib.pyplot as plt 
import os

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, plot_type in enumerate(["loss", "train_accuracy", "test_accuracy"]):
    for run_name, logger in loggers.items():
        data = logger.data[plot_type]

        axs[i].plot(list(map(int, data.keys())), list(data.values()), label=run_name)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(plot_type)
        axs[i].grid()
        axs[i].legend()
    
fig.tight_layout()
if not os.path.exists(args.plot_dir):
    os.makedirs(args.plot_dir)

fig.savefig(os.path.join(args.plot_dir, f"{' '.join(args.run_names)}.png"))

# # ==== Plot impacts ====
# import numpy as np
# from tqdm import tqdm

# first_logger = next(iter(loggers.values()))
# num_epochs = len(first_logger.data["loss"])
# keys = first_logger.data["impacts"][0].keys()
# num_plots = len(keys)

# plt.figure(figsize=(num_plots * 4, num_epochs * 4))

# for epoch in tqdm(range(num_epochs)):
#     for key in keys:
#         for run_name, logger in loggers.items():
#             impacts = logger.data["impacts"][epoch]
#             flat_tensor = impacts[key].view(-1).cpu().numpy() * impacts[key].numel()  # Вытягивание тензора в одномерный массив

#             plt.subplot(len(args.run_names), num_plots, (args.run_names.index(run_name) * num_plots) + i)
#             plt.hist(flat_tensor, alpha=0.75, label=run_name)
#             plt.title(f"{run_name} - {key}")
#             plt.xlabel('Value')
#             plt.ylabel('Frequency')

# plt.tight_layout()
# plt.savefig(os.path.join(args.plot_dir, f"{' '.join(args.run_names)}_impacts.png"))

