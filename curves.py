import pandas as pd
import matplotlib.pyplot as plt

# Load the training log
train_log = pd.read_excel('Results/train_log.xlsx')

# Shift Epochs by +1 if they start from 0
train_log['Epoch'] = train_log['Epoch'] + 1

# Filter up to best epoch
filtered_log = train_log[train_log['Epoch'] <= 230]

# Plot Training and Validation BCE
plt.figure(figsize=(8, 5))
plt.plot(filtered_log['Epoch'], filtered_log['Train_BCE'], label='Train Loss (BCE)', color='blue')
plt.plot(filtered_log['Epoch'], filtered_log['Val_BCE'], label='Validation Loss (BCE)', color='red')
plt.title('Training and Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.xticks(ticks=range(0, 251, 50))  # Set x-ticks at 0, 50, 100, 150, 200, 250
plt.legend(loc='center right', frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("Results/loss.png")

# Plot Validation AUC
plt.figure(figsize=(8, 5))
plt.plot(filtered_log['Epoch'], filtered_log['Val_AUC'], label='Validation AUC', color='green')
plt.title('Validation AUC vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Validation AUC')
plt.xticks(ticks=range(0, 251, 50))  # Set x-ticks same here too
plt.legend(loc='center right', frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("Results/AUC.png")