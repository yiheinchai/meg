import torch
import matplotlib.pyplot as plt
import os

def check_loss(file_name):
    losses = torch.load(file_name, weights_only=True)
    plt.plot(losses.to("cpu").numpy())
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Iterations")
    # plt.show()
    # os.makedirs("../figures", exist_ok=True)
    plt.savefig("./figures/training_loss_plot.png")
    


if __name__ == "__main__":
    check_loss("checkpoint/run_1_epoch_9_training_losses.pt")