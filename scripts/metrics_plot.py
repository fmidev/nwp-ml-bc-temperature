import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PATH = Path.home() / "thesis_project" / "metrics"
OUT = Path.home() / "thesis_project" / "figures" /"metrics"
OUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(PATH / "metrics_overall_analysistime_2023.csv")

# Drop non-numeric columns if present
df = df.select_dtypes(include="number")

ax = df.T.plot(kind="bar", legend=False, color="plum")
plt.title("Overall Metrics (analysistime split)")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Annotate bars
for container in ax.containers:              # one container per "series"
    ax.bar_label(container, fmt="%.2f")      # format numbers with 2 decimals

plt.savefig(OUT / "metrics.png")
plt.close()



df = pd.read_csv(PATH / "metrics_overall_skill_analysistime_2023.csv").iloc[0]



# your RMSE values dict
rmse_vals = {
    "Raw": df["rmse_raw"],
    "Corrected (tuned)": df["rmse_corrected_tuned"],
    "Corrected (tuned_weighted)": df["rmse_corrected_tuned_weighted"],
}

fig, ax = plt.subplots()
bars = ax.bar(rmse_vals.keys(), rmse_vals.values(), color="plum")

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # x position = middle of bar
        height,                             # y position = bar height
        f"{height:.2f}",                    # format value
        ha="center", va="bottom"            # center horizontally, just above bar
    )

ax.set_title("RMSE comparison")
ax.set_ylabel("RMSE")
plt.tight_layout()
plt.savefig(OUT / "rmse.png")
plt.close()


skill_vals = {
    "Corrected (tuned)": df["skill_vs_raw_corrected_tuned"] * 100,
    "Corrected (tuned_weighted)": df["skill_vs_raw_corrected_tuned_weighted"] * 100,
}

fig, ax = plt.subplots()
bars = ax.bar(skill_vals.keys(), skill_vals.values(), color="plum")

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.1f}%",   # show one decimal + percent sign
        ha="center", va="bottom"
    )

ax.set_title("Skill vs Raw Forecast")
ax.set_ylabel("Improvement (%)")
plt.tight_layout()
plt.savefig(OUT / "skill.png")
plt.close()
