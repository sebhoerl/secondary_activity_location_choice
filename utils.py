import pickle
import numpy as np
import pyproj
import random
import matplotlib.pyplot as plt
import numpy.linalg as la
import data
import os

DISTANCE_DISTRIBUTION_BINS = 40
SPATIAL_DISTRIBUTION_BINS = (6, 6)

class DistanceDistribution:
    def __init__(self, data = [], maximum = None, prior = 0.0):
        self.prior = prior
        self.bins = DISTANCE_DISTRIBUTION_BINS
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

class SpatialDistribution:
    def __init__(self, data = [], minimum = None, maximum = None, prior = 0.0):
        self.prior = prior
        self.bins = SPATIAL_DISTRIBUTION_BINS

        self.minimum = np.min(data, axis = 0) if minimum is None else minimum
        self.maximum = np.max(data, axis = 0) if maximum is None else maximum

        self.dc = (self.maximum - self.minimum) / np.array(self.bins)

        self.epdf = np.zeros((self.bins))
        self.counts = np.zeros((self.bins), dtype = np.int)

        self.updated = False

        for coord in data: self.add(coord)
        self._update()

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
        X = np.zeros(self.bins)
        Y = np.zeros(self.bins)

        for i in range(self.bins[0]):
            X[i,:] = i

        for j in range(self.bins[1]):
            Y[:,j] = j

        X *= self.dc[0]
        Y *= self.dc[1]

        return X + self.minimum[0], Y + self.minimum[1]

    def pdf(self, c):
        if not self.updated: self._update()
        return self.epdf[self.get_bin(c)]

    def get_epdf(self):
        if not self.updated: self._update()
        return self.epdf

class DistributionFactory:
    def __init__(self, census_data):
        self.census_data = census_data
        self.distance_distributions = {}
        self.spatial_distributions = {}

    def get_distance_distribution(self, mode = None, activity_type = None):
        key = (mode, activity_type)

        if key in self.distance_distributions:
            return self.distance_distributions[key]

        distances = self.census_data.get_distances(mode, activity_type)
        distribution = DistanceDistribution(distances, prior = 0.001)

        self.distance_distributions[key] = distribution
        return distribution

    def get_spatial_distribution(self, activity_type = None):
        if activity_type in self.spatial_distributions:
            return self.spatial_distributions[activity_type]

        coords = self.census_data.get_activity_locations(activity_type)
        distribution = SpatialDistribution(coords, prior = 0.001)

        self.spatial_distributions[activity_type] = distribution
        return distribution

























class LikelihoodTracker:
    def __init__(self, alpha, beta, activities, facility_locations, distribution_factory):
        self.activities = activities
        self.locations = [None] * len(self.activities)
        self.facility_locations = facility_locations
        self.alpha = alpha
        self.beta = beta

        self.distribution_factory = distribution_factory
        self.L = 0

        self._prepare()

    def compute_left(self, activity_index):
        activity = self.activities[activity_index]
        if activity[0] is None: return 0

        location = self.locations[activity_index]

        left = self.activities[activity[0]]
        left_location = self.locations[activity[0]]

        distribution = self.distribution_factory.get_distance_distribution(activity[3], activity[2])

        return self.alpha * np.log(distribution.pdf(la.norm(left_location - location)))

    def compute_right(self, activity_index):
        activity = self.activities[activity_index]
        if activity[1] is None: return 0

        location = self.locations[activity_index]

        right = self.activities[activity[1]]
        right_location = self.locations[activity[1]]

        distribution = self.distribution_factory.get_distance_distribution(right[3], right[2])
        return self.alpha * np.log(distribution.pdf(la.norm(right_location - location)))

    def compute_center(self, activity_index):
        activity = self.activities[activity_index]
        location = self.locations[activity_index]

        distribution = self.distribution_factory.get_spatial_distribution(activity[2])
        return self.beta * np.log(distribution.pdf(self.locations[activity_index]))

    def compute(self, activity_index):
        return self.compute_left(activity_index) + self.compute_center(activity_index) + self.compute_right(activity_index)

    def _prepare(self):
        for i in range(len(self.activities)):
            self.locations[i] = self.facility_locations[self.activities[i][4]]

        for i in range(len(self.activities)):
            self.L += self.compute(i)

    def replace(self, activity_index, location):
        previous_L = self.L

        current_activity = activity_index
        previous_activity = self.activities[activity_index][0]
        next_activity = self.activities[activity_index][1]

        remove_L = sum([self.compute(activity_index) for a in (current_activity, previous_activity, next_activity) if a is not None])

        self.locations[activity_index] = location

        add_L = sum([self.compute(activity_index) for a in (current_activity, previous_activity, next_activity) if a is not None])

        self.L += add_L - remove_L
        return add_L - remove_L, self.L

class ProgressTracker:
    def __init__(self, activities, locations, distribution_factory, sample = 0):
        self.L = []
        self.dL = []
        self.its = []
        self.activities = activities
        self.locations = locations
        self.distribution_factory = distribution_factory

        #if os.path.isfile("output/progress.pickle"):
        #    with open("output/progress.pickle", "rb") as f:
        #        self.L, self.dL, self.its = pickle.load(f)

        self.sample = sample

    def track(self, difference, absolute):
        self.sample += 1
        swipe = int(self.sample / len(self.activities))
        trigger = max(10**5, 10 ** int(np.log10(self.sample)))

        print("%d samples / %d swipes %% %d" % (self.sample, swipe, trigger))

        if self.sample % trigger == 0:
            self.L.append(absolute)
            self.dL.append(difference)
            self.its.append(self.sample)

            with open("output/progress.pickle", "wb+") as f:
                pickle.dump((self.L, self.dL, self.its), f)

            plt.figure()
            plt.plot(self.its, self.L)
            plt.savefig("output/L.png")
            plt.close()

            plt.figure()
            plt.plot(self.its, self.dL)
            plt.savefig("output/dL.png")
            plt.close()

            indices = [a[4] for a in self.activities]

            #with open("output/locations.pickle", "wb+") as f:
            #    pickle.dump(indices, f)

            #for t in list(data.ACTIVITY_TYPES) + [None]:
            for t in [None]:
                suffix = "_" + t if t is not None else ""

                locations = np.array([self.locations[a[4]] for a in self.activities if a[2] == t or t is None]).astype(np.float)

                reference_distribution = self.distribution_factory.get_spatial_distribution(activity_type = t)
                sampler_distribution = SpatialDistribution(locations, minimum = reference_distribution.minimum, maximum = reference_distribution.maximum)

                epdf = reference_distribution.epdf
                order = np.dstack(np.unravel_index(np.argsort(epdf.ravel()), epdf.shape))[0][::-1]

                reference = [epdf[ind[0], ind[1]] for ind in order]
                sampler = [sampler_distribution.epdf[ind[0], ind[1]] for ind in order]

                plt.figure()
                plt.bar(np.arange(order.shape[0]), reference, 0.3, color = "b")
                plt.bar(np.arange(order.shape[0]) + 0.3, sampler, 0.3, color = "r")
                plt.savefig("output/spatial%s.png" % suffix)
                plt.close()

            #for t in list(data.ACTIVITY_TYPES) + [None]:
            #    for m in list(data.MODES) + [None]:
            for t in [None]:
                for m in [None]:
                    suffix = ""
                    if t is not None: suffix += "_%s" % t
                    if m is not None: suffix += "_%s" % m

                    distances = []

                    for a in self.activities:
                        if a[1] is not None:
                            na = self.activities[a[1]]

                            if (na[2] == t or t is None) and (na[3] == m or m is None):
                                distances.append(la.norm(self.locations[a[4]] - self.locations[na[4]]))

                    reference_distribution = self.distribution_factory.get_distance_distribution(activity_type = t, mode = m)
                    sampler_distribution = DistanceDistribution(distances, maximum = reference_distribution.maximum)

                    lattice = reference_distribution.get_range()
                    width = lattice[1] - lattice[0]

                    plt.figure()
                    plt.bar(lattice, reference_distribution.epdf, width = width * 0.3, color = "b")
                    plt.bar(lattice + 0.3 * width, sampler_distribution.epdf, width = width * 0.3, color = "r")
                    plt.savefig("output/distance%s.png" % suffix)
                    plt.close()
