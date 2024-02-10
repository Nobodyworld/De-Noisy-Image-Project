# /utils/plotting.py
import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs, label="Training PSNR")
    plt.plot(val_psnrs, label="Validation PSNR")
    plt.title("Training and Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
