import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv(
    "../../outputs/training_history.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Autoencoder Training History"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "../../outputs/learning_curve.png"
)

plt.show()