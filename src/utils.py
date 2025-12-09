import torch
import matplotlib.pyplot as plt
import os
from constants import WINDOW_CACHE_PATH

def check_loss(file_name):
    losses = torch.load(file_name, weights_only=True)
    plt.plot(losses.to("cpu").numpy())
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Iterations")
    # plt.show()
    # os.makedirs("../figures", exist_ok=True)
    plt.savefig("./figures/training_loss_plot.png")


def plot_window(file_name):
    data = torch.load(file_name, weights_only=True)
    plt.plot(data.to("cpu").numpy())
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Data Window Plot")
    plt.savefig("./figures/data_window_plot.png")
    


if __name__ == "__main__":
    # check_loss("checkpoint/run_1_epoch_9_training_losses.pt")
    plot_window(WINDOW_CACHE_PATH / "sub-CC721585" / "1.pt")