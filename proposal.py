import numpy as np
import numpy.linalg as la

class ProposalDistribution:
    def __init__(self, activities, facility_indices, locations, distribution_factory):
        self.activities = activities
        self.locations = locations
        self.distribution_factory = distribution_factory
        self.facility_indices = facility_indices

    def get_proposal(self, activity_index):
        raise NotImplementedError()

class RandomProposalDistribution(ProposalDistribution):
    def get_proposal(self, activity_index):
        activity = self.activities[activity_index]
        indices = self.facility_indices[activity[2]]
        return indices[np.random.randint(len(indices))], 0.0

class ModelBasedProposalDistribution(ProposalDistribution):
    def __init__(self, alpha, beta, activities, facility_indices, locations, distribution_factory):
        ProposalDistribution.__init__(self, activities, facility_indices, locations, distribution_factory)

        self.alpha = alpha
        self.beta = beta

        self.distance_likelihood_func = np.vectorize(ModelBasedProposalDistribution.distance_likelihood)
        self.spatial_likelihood_func = np.vectorize(ModelBasedProposalDistribution.spatial_likelihood)

    def get_proposal(self, activity_index):
        activity = self.activities[activity_index]
        indices = self.facility_indices[activity[2]]

        previous_activity, current_activity, next_activity = self.activities[activity[0]], activity, self.activities[activity[1]]
        previous_location, current_location, next_location = [self.locations[a[4]] for a in (previous_activity, current_activity, next_activity)]

        activity_type, activity_mode = activity[2], activity[3]
        next_activity_type, next_activity_mode = next_activity[2], next_activity[3]

        previous_distances = la.norm(self.locations[indices] - previous_location, axis = 1)
        next_distances = la.norm(self.locations[indices] - next_location, axis = 1)

        previous_distance_distribution = self.distribution_factory.get_distance_distribution(activity_mode, activity_type)
        next_distance_distribution = self.distribution_factory.get_distance_distribution(next_activity_mode, next_activity_type)
        spatial_distribution = self.distribution_factory.get_spatial_distribution(activity_type)

        pdL = np.log(np.array([previous_distance_distribution.pdf(d) for d in previous_distances]))
        ndL = np.log(np.array([next_distance_distribution.pdf(d) for d in next_distances]))
        alL = np.log(np.array([spatial_distribution.pdf(c) for c in self.locations[indices]]))

        prev_pdL = np.log(previous_distance_distribution.pdf(la.norm(current_location - previous_location)))
        prev_ndL = np.log(next_distance_distribution.pdf(la.norm(current_location - next_location)))
        prev_alL = np.log(spatial_distribution.pdf(current_location - previous_location))

        L = self.alpha * (pdL + ndL) + self.beta * alL
        prev_L = self.alpha * (prev_pdL + prev_ndL) + self.beta * prev_alL

        maximizer = np.argmax(L)
        index = indices[maximizer]

        new_proposal_likelihood = L[maximizer]
        previous_proposal_likelihood = prev_L

        return index, previous_proposal_likelihood - new_proposal_likelihood
