import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("scratch/balance_sheet.csv")
history = [1000.0] + df['Balance_INR'].tolist()

plt.figure(figsize=(10, 5))
plt.plot(range(len(history)), history, marker='o', color='green' if history[-1] >= 1000 else 'red')
plt.title("2026 World Cup Arbitrage Bankroll Growth (INR)")
plt.xlabel("Matches Completed")
plt.ylabel("Bankroll (INR)")
plt.grid(True)
plt.axhline(1000, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("scratch/bankroll_chart.png")
print("[OK] Saved Bankroll Chart")
