import numpy as np
import numpy.linalg as la
import utils
from scipy.special import gammaln
loggamma = gammaln

class DistributionModel:
    def __init__(self, reference, distribution):
        self.reference = reference
        self.distribution = distribution

    def get_initial_elements(self, activity_index):
        raise NotImplementedError()

    def get_final_elements(self, activity_index, facility_index):
        raise NotImplementedError()

    def compute_likelihood(self):
        N = np.sum(self.distribution.counts)
        return loggamma(N + 1) - np.sum(loggamma(self.distribution.counts + 1)) + np.sum(self.distribution.counts * np.log(self.reference.get_epdf()))

    def compute_likelihood_difference(self, activity_index, facility_index):
        initial = self.get_initial_elements(activity_index)
        final = self.get_final_elements(activity_index, facility_index)

        beforeL = self.compute_likelihood()

        for e in initial: self.distribution.remove(e)
        for e in final: self.distribution.add(e)

        afterL = self.compute_likelihood()

        for e in initial: self.distribution.add(e)
        for e in final: self.distribution.remove(e)

        return afterL - beforeL

    def replace(self, activity_index, facility_index):
        initial = self.get_initial_elements(activity_index)
        final = self.get_final_elements(activity_index, facility_index)

        for e in initial: self.distribution.remove(e)
        for e in final: self.distribution.add(e)

class SpatialDistributionModel(DistributionModel):
    def __init__(self, activities, locations, distribution_factory, activity_type = None):
        self.activities = activities
        self.locations = locations
        self.activity_type = activity_type

        self.reference = distribution_factory.get_spatial_distribution(activity_type)
        self.distribution = utils.SpatialDistribution(minimum = self.reference.minimum, maximum = self.reference.maximum, prior = 0.001)

        for activity in activities:
            if activity[2] == activity_type or activity_type is None:
                self.distribution.add(locations[activity[4]])

    def get_initial_elements(self, activity_index):
        return [ self.locations[self.activities[activity_index][4]] ]

    def get_final_elements(self, activity_index, facility_index):
        return [ self.locations[facility_index] ]

    def replace(self, activity_index, facility_index):
        if self.activities[activity_index][2] == self.activity_type or self.activity_type is None:
            DistributionModel.replace(self, activity_index, facility_index)

class DistanceDistributionModel(DistributionModel):
    def __init__(self, activities, locations, distribution_factory, mode = None, activity_type = None):
        self.activities = activities
        self.locations = locations

        self.reference = distribution_factory.get_distance_distribution(mode = mode, activity_type = activity_type)
        self.distribution = utils.DistanceDistribution(maximum = self.reference.maximum, prior = 0.001)

        self.mode = mode
        self.activity_type = activity_type

        for activity_index, activity in enumerate(activities):
            if (activity[2] == activity_type or activity_type is None) and (activity[3] == mode or mode is None):
                source = locations[activity[4]]

                if activity[1] is not None:
                    target = locations[activities[activity[1]][4]]
                    self.distribution.add(la.norm(source - target))

    def get_distances(self, activity_index, location):
        activity = self.activities[activity_index]
        distances = []

        if activity[0] is not None:
            left = self.locations[self.activities[activity[0]][4]]
            distances.append(la.norm(left - location))

        if activity[1] is not None:
            right = self.locations[self.activities[activity[1]][4]]
            distances.append(la.norm(right - location))

        return distances

    def get_initial_elements(self, activity_index):
        return self.get_distances(activity_index, self.locations[self.activities[activity_index][4]])

    def get_final_elements(self, activity_index, facility_index):
        return self.get_distances(activity_index, self.locations[facility_index])

    def replace(self, activity_index, facility_index):
        if self.activities[activity_index][2] == self.activity_type or self.activity_type is None:
            if self.activities[activity_index][3] == self.mode or self.mode is None:
                DistributionModel.replace(self, activity_index, facility_index)

class HybridDistributionModel(DistributionModel):
    def __init__(self):
        self.distributions = []
        self.factors = []
        self.names = []

    def add(self, distribution, factor = 1.0, name = None):
        self.distributions.append(distribution)
        self.factors.append(factor)
        self.names.append(name)

    def compute_likelihood(self):
        return np.dot(self.factors, [d.compute_likelihood() for d in self.distributions])

    def compute_likelihood_difference(self, activity_index, facility_index):
        return np.dot(self.factors, [d.compute_likelihood_difference(activity_index, facility_index) for d in self.distributions])

    def replace(self, activity_index, facility_index):
        for d in self.distributions: d.replace(activity_index, facility_index)
