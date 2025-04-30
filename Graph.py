import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CSV = "data/weekly_dataset1.csv"
K = 8                            # k-NN parameter

# Load station list
df = pd.read_csv(CSV)
stations = df[["STATION", "LATITUDE", "LONGITUDE", "SPLIT"]].drop_duplicates()
lat = stations.LATITUDE.values
lon = stations.LONGITUDE.values
split = stations.SPLIT.values
n = len(stations)

# Build k-NN edges in NumPy (great-circle distance)
coords_rad = np.radians(np.c_[lat, lon])
tree       = BallTree(coords_rad, metric="haversine")
_, idx     = tree.query(coords_rad, k=K + 1)
src = np.repeat(np.arange(n), K)
dst = idx[:, 1:].reshape(-1)           # drop self-loop

# Draw the map
proj = ccrs.PlateCarree()
fig  = plt.figure(figsize=(9, 5))
ax   = fig.add_subplot(1, 1, 1, projection=proj)

# Basemap layers
ax.add_feature(cfeature.LAND,   facecolor="white", edgecolor="0.6", linewidth=0.2)
ax.add_feature(cfeature.LAKES,  facecolor="#AFCBFF", edgecolor="#3366cc", linewidth=0.4)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax.set_extent([-93, -75, 40, 49])      # roughly the Great-Lakes bounding box

# Edges first (light so nodes pop)
for s, d in zip(src, dst):
    ax.plot([lon[s], lon[d]], [lat[s], lat[d]],
            transform=proj, color="lightgray", linewidth=0.5, zorder=1)

# Nodes
train = split == "train"
ax.scatter(lon[train],  lat[train],  c="royalblue",  s=35, label="Train", transform=proj, zorder=3)
ax.scatter(lon[~train], lat[~train], c="darkorange", s=45, marker="s", label="Val",   transform=proj, zorder=3)

ax.legend(loc="lower left")
ax.set_title(f"Spatial 8-NN Graph for Great Lakes Stations")
plt.tight_layout()
plt.savefig("Results/Graph.png")