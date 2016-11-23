import data
import utils
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import os, pickle
import models
import proposal
import sys
import shapefile
import pyproj
import matplotlib.patches as patches
import colorsys

census_data = data.load_census_data()
facility_data = data.load_facility_data()
chain_data = data.load_chain_data(facility_data)
districts = data.load_district_data()

activities = chain_data.get_activities()
facility_indices = facility_data.get_facility_indices()
locations = facility_data.get_locations()

distribution_factory = utils.DistributionFactory(census_data)

coords = census_data.get_activity_locations("shop")
spatial = distribution_factory.get_spatial_distribution("shop")



counts = spatial.counts
maxval = np.max(counts)

plt.figure()
f = counts.flatten()
f.sort()
plt.bar(np.arange(len(f)), f, width = 0.8)
plt.show()
exit()

plt.figure(figsize=(12,8))

for b in spatial.get_bins():
    for path in spatial.get_path(b):
        plt.gca().add_patch(patches.PathPatch(path, facecolor = colorsys.hsv_to_rgb(0.7, (counts[b] / maxval), 1.0), edgecolor="black"))

    centroid = spatial.get_centroid(b)
    #plt.plot(centroid[0], centroid[1], "o", color = "black")

#plt.scatter(locations[:,0], locations[:,1], color = "b", zorder=2, alpha = 0.1)
plt.scatter(coords[:,0], coords[:,1], zorder=2, alpha = 0.01, color = "red")

plt.savefig("shop.png")
#plt.show()
