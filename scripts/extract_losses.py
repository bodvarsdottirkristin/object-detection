import re
import csv
from pathlib import Path

# Path to your .out log file (update this as needed)
JOB_NUMBER = "27312386"
LOG_FILE = f"jobs/outputs/pothole_classifier_a100_{JOB_NUMBER}.out"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CSV_FILE = RESULTS_DIR / f"loss_curves_{JOB_NUMBER}.csv"

# Patterns to extract epoch, train loss, val loss
epoch_pat = re.compile(r".*Epoch (\d+) Summary:")
train_pat = re.compile(r".*Train Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)%")
val_pat = re.compile(r".*Val Loss: ([\d\.]+) \| Val Acc: ([\d\.]+)%")

data = []
current = {}

with open(LOG_FILE) as f:
    for line in f:
        # Detect epoch start
        m = epoch_pat.match(line)
        if m:
            if current:
                data.append(current)
            current = {"epoch": int(m.group(1))}
        # Train loss/acc
        m = train_pat.match(line)
        if m and current is not None:
            current["train_loss"] = float(m.group(1))
            current["train_acc"] = float(m.group(2))
        # Val loss/acc
        m = val_pat.match(line)
        if m and current is not None:
            current["val_loss"] = float(m.group(1))
            current["val_acc"] = float(m.group(2))
# Append last epoch
if current:
    data.append(current)

# Write to CSV
with open(CSV_FILE, "w", newline="") as csvfile:
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"Saved loss curves to {CSV_FILE}")