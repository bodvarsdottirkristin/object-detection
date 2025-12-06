import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
JOB_NUMBER = "27312386"
CSV_FILE = f"results/loss_curves_{JOB_NUMBER}.csv"
EARLY_STOP_EPOCH = 16

# Seaborn style
sns.set(style="whitegrid", palette="muted")

# Load data
df = pd.read_csv(CSV_FILE)

plt.figure(figsize=(10, 6))

# Plot train and val loss
sns.lineplot(x="epoch", y="train_loss", data=df, label="Train Loss", linestyle="--")
sns.lineplot(x="epoch", y="val_loss", data=df, label="Val Loss")

# Early stopping vertical line
plt.axvline(EARLY_STOP_EPOCH, color="orange", linestyle=":", linewidth=2, label="Early Stopping")

# Annotate early stopping
plt.text(EARLY_STOP_EPOCH + 0.5, plt.ylim()[1]*0.95, "Early Stopping on mAP", color="orange", rotation=90, va="top", ha="left")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/loss_plot_{JOB_NUMBER}.png", dpi=600)