import numpy as np
import matplotlib.pyplot as plt

visit_counts = np.load("visit_counts.npy")

plt.figure(figsize=(6,6))
plt.imshow(visit_counts, cmap="hot")
plt.colorbar(label="State Visit Frequency")
plt.title("State Visitation Heatmap")
plt.xlabel("Column")
plt.ylabel("Row")
plt.grid(False)

plt.savefig("heatmap.png", dpi=300)
plt.show()
