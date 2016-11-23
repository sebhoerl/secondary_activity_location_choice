import pickle
import numpy as np
import pyproj
import random
import matplotlib.pyplot as plt
import numpy.linalg as la
import data
import os
from itertools import product
import matplotlib.path as mplPath
from tqdm import tqdm

class DistanceDistribution:
    def __init__(self, data = [], maximum = None, prior = 0.0, bins = 40):
        self.prior = prior
        self.bins = bins
        self.maximum = np.max(data) if maximum is None else maximum
        self.dx = self.maximum / self.bins

        self.counts = np.zeros((self.bins), dtype = np.int)

        self.epdf = np.zeros((self.bins))
        self.ecdf = np.zeros((self.bins))

        self.updated = False

        for distance in data: self.add(distance)
        self._update()

    def add(self, distance):
        self.counts[self.get_bin(distance)] += 1
        self.updated = False

    def remove(self, distance):
        self.counts[self.get_bin(distance)] -= 1
        self.updated = False

    def _update(self):
        self.epdf = self.counts + self.prior
        self.epdf /= np.sum(self.epdf)
        self.ecdf = np.cumsum(self.epdf)
        self.updated = True

    def get_bin(self, x):
        return min(int(np.floor(x / self.dx)), self.bins - 1)

    def get_range(self):
        return np.arange(self.bins) * self.dx

    def pdf(self, x):
        if not self.updated: self._update()
        return self.epdf[self.get_bin(x)]

    def cdf(self, x):
        if not self.updated: self._update()
        return self.ecdf[self.get_bin(x)]

    def get_epdf(self):
        if not self.updated: self._update()
        return self.epdf

    def get_ecdf(self):
        if not self.updated: self._update()
        return self.ecdf

class DistrictBasedSpatialDistribution:
    def __init__(self, districts, data = None, minimum = None, maximum = None, prior = 0.0):
        self.prior = prior
        self.districts = districts
        self.bins = len(districts)

        self.epdf = np.zeros((self.bins))
        self.counts = np.zeros((self.bins))

        self.updated = False
        self.centroids = None

        self.add_all(data)
        self._update()

    def add_all(self, coords = None):
        if coords is not None:
            with tqdm(total = coords.shape[0]) as bar:
                for coord in coords:
                    self.add(coord)
                    bar.update(1)

    def add(self, coord):
        self.counts[self.get_bin(coord)] += 1
        self.updated = False

    def remove(self, coord):
        self.counts[self.get_bin(coord)] += 1
        self.updated = False

    def get_bin(self, c):
        distances = la.norm(self.get_centroids() - c, axis = 1)

        for index in np.argsort(distances):
            if self.districts[index].contains(c):
                return index

        return int(np.argmin(distances))

    def _update(self):
        self.epdf = self.counts + self.prior
        self.epdf /= np.sum(self.epdf)
        self.updated = True

    def get_centroids(self):
        if self.centroids is None:
            self.centroids = np.array([ d.center for d in self.districts ])

        return self.centroids

    def pdf(self, c):
        if not self.updated: self._update()
        return self.epdf[self.get_bin(c)]

    def get_epdf(self):
        if not self.updated: self._update()
        return self.epdf

    def get_bins(self):
        return list(range(self.bins))

    def get_path(self, b):
        return self.get_paths()[b]

    def get_centroid(self, b):
        return self.get_centroids()[b]

    def get_epdf_by_bin(self, b):
        return self.get_epdf()[b]

    def get_paths(self):
        return [d.paths for d in self.districts]

class SpatialDistribution:
    def __init__(self, data = None, minimum = None, maximum = None, prior = 0.0, bins = (10, 10)):
        self.prior = prior
        self.bins = bins

        self.minimum = np.min(data, axis = 0) if minimum is None else minimum
        self.maximum = np.max(data, axis = 0) if maximum is None else maximum

        self.dc = (self.maximum - self.minimum) / np.array(self.bins)

        self.epdf = np.zeros((self.bins))
        self.counts = np.zeros((self.bins), dtype = np.int)

        self.updated = False
        self.centroids = None

        self.add_all(data)
        self._update()

    def add_all(self, coords = None):
        if coords is not None:
            with tqdm(total = coords.shape[0]) as bar:
                for coord in coords:
                    self.add(coord)
                    bar.update(1)

    def add(self, coord):
        self.counts[self.get_bin(coord)] += 1
        self.updated = False

    def remove(self, coord):
        self.counts[self.get_bin(coord)] -= 1
        self.updated = False

    def get_bin(self, c):
        ij = np.floor((c - self.minimum) / self.dc)

        ij[0] = max(0, min(self.bins[0] - 1, ij[0]))
        ij[1] = max(0, min(self.bins[1] - 1, ij[1]))

        return (int(ij[0]), int(ij[1]))

    def _update(self):
        self.epdf = self.counts + self.prior
        self.epdf /= np.sum(self.epdf)
        self.updated = True

    def get_lattice(self):
        X = np.zeros((self.bins[0] + 1, self.bins[1] + 1))
        Y = np.zeros((self.bins[0] + 1, self.bins[1] + 1))

        for i in range(self.bins[0] + 1):
            X[i,:] = i

        for j in range(self.bins[1] + 1):
            Y[:,j] = j

        X *= self.dc[0]
        Y *= self.dc[1]

        return X + self.minimum[0], Y + self.minimum[1]

    def get_paths(self):
        return { (i,j) : [mplPath.Path(np.array([
                [i, j],
                [i + 1, j],
                [i + 1, j + 1],
                [i, j + 1],
                [i, j]
            ]) * self.dc + self.minimum)] for i, j in product(range(self.bins[0]), range(self.bins[1])) }

    def get_bins(self):
        return [(i,j) for i,j in product(range(self.bins[0]), range(self.bins[1]))]

    def get_centroids(self):
        return np.array([
            self.minimum + self.dc * np.array(b).astype(np.float) + 0.5 * self.dc for b in self.get_bins()
        ])

    def get_path(self, b):
        return self.get_paths()[b]

    def get_centroid(self, b):
        return self.get_centroids()[self.get_bins().index(b)]

    def get_epdf_by_bin(self, b):
        return self.get_epdf()[b]

    def pdf(self, c):
        if not self.updated: self._update()
        return self.epdf[self.get_bin(c)]

    def get_epdf(self):
        if not self.updated: self._update()
        return self.epdf

class DistributionFactory:
    def __init__(self, census_data, district_data, spatial_model = "grid", distance_bins = 40, spatial_bins = (10, 10)):
        self.census_data = census_data
        self.district_data = district_data
        self.distance_bins = distance_bins
        self.spatial_model = spatial_model
        self.spatial_bins = spatial_bins
        self.distance_distributions = {}
        self.spatial_distributions = {}
        self.districts = district_data

    def get_distance_distribution(self, mode = None, activity_type = None):
        key = (mode, activity_type)

        if key in self.distance_distributions:
            return self.distance_distributions[key]

        print("Loading distance distribution for " + str(mode) + " / " + str(activity_type) + " ...")

        distances = self.census_data.get_distances(mode, activity_type)
        distribution = DistanceDistribution(distances, prior = 0.001, bins = self.distance_bins)

        self.distance_distributions[key] = distribution
        return distribution

    def get_spatial_distribution(self, activity_type = None):
        if activity_type in self.spatial_distributions:
            return self.spatial_distributions[activity_type]

        print("Loading spatial distribution for " + str(activity_type) + " ...")

        coords = self.census_data.get_activity_locations(activity_type)

        if self.spatial_model == "districts":
            distribution = DistrictBasedSpatialDistribution(self.districts, coords, prior = 0.001)
        elif self.spatial_model == "grid":
            distribution = SpatialDistribution(coords, prior = 0.001, bins = self.spatial_bins)
        else:
            raise "Invalid spatial model"

        self.spatial_distributions[activity_type] = distribution
        return distribution

    def create_spatial_distribution(self, reference):
        print("Creating spatial distribution ...")
        if self.spatial_model == "districts":
            return DistrictBasedSpatialDistribution(self.districts, prior = 0.001)
        elif self.spatial_model == "grid":
            return SpatialDistribution(minimum = reference.minimum, maximum = reference.maximum, prior = 0.001, bins = self.spatial_bins)
        else:
            raise "Invalid spatial model"
