import numpy as np
import numpy.linalg as la
import scipy.misc
import utils

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

        #prev_pdL = np.log(previous_distance_distribution.pdf(la.norm(current_location - previous_location)))
        #prev_ndL = np.log(next_distance_distribution.pdf(la.norm(current_location - next_location)))
        #prev_alL = np.log(spatial_distribution.pdf(current_location - previous_location))

        L = self.beta * (pdL + ndL) + self.alpha * alL
        #prev_L = self.alpha * (prev_pdL + prev_ndL) + self.beta * prev_alL

        P = np.exp(L - scipy.misc.logsumexp(L))
        selection = np.random.choice(np.arange(len(indices)), p = P)

        new_proposal_likelihood = L[selection]
        #previous_proposal_likelihood = prev_L

        return indices[selection], 0 # previous_proposal_likelihood - new_proposal_likelihood

class ApproximateProposalDistribution(ProposalDistribution):
    def __init__(self, bin2indices, alpha, beta, activities, facility_indices, locations, distribution_factory):
        ProposalDistribution.__init__(self, activities, facility_indices, locations, distribution_factory)

        self.alpha = alpha
        self.beta = beta
        self.bin2indices = bin2indices

    def get_proposal(self, activity_index):
        activity = self.activities[activity_index]

        previous_activity, current_activity, next_activity = self.activities[activity[0]], activity, self.activities[activity[1]]
        previous_location, current_location, next_location = [self.locations[a[4]] for a in (previous_activity, current_activity, next_activity)]

        activity_type, activity_mode = activity[2], activity[3]
        next_activity_type, next_activity_mode = next_activity[2], next_activity[3]

        previous_distance_distribution = self.distribution_factory.get_distance_distribution(activity_mode, activity_type)
        next_distance_distribution = self.distribution_factory.get_distance_distribution(next_activity_mode, next_activity_type)
        spatial_distribution = self.distribution_factory.get_spatial_distribution(activity_type)

        centroids = spatial_distribution.get_centroids()
        bins = spatial_distribution.get_bins()

        previous_distances = la.norm(centroids - previous_location, axis = 1)
        next_distances = la.norm(centroids - next_location, axis = 1)

        pdL = np.log(np.array([previous_distance_distribution.pdf(d) for d in previous_distances]))
        ndL = np.log(np.array([next_distance_distribution.pdf(d) for d in next_distances]))
        alL = np.log(np.array([spatial_distribution.pdf(c) for c in centroids]))

        #prev_pdL = np.log(previous_distance_distribution.pdf(la.norm(current_location - previous_location)))
        #prev_ndL = np.log(next_distance_distribution.pdf(la.norm(current_location - next_location)))
        #prev_alL = np.log(spatial_distribution.pdf(current_location - previous_location))
        #prev_L = self.alpha * (prev_pdL + prev_ndL) + self.beta * prev_alL

        L = self.beta * (pdL + ndL) + self.alpha * alL
        P = np.exp(L - scipy.misc.logsumexp(L))
        selection = np.random.choice(np.arange(len(centroids)), p = P)

        indices = self.bin2indices.get_indices(bins[selection], activity[2])
        if indices is None: return None, None

        index = indices[np.random.randint(len(indices))]
        selected_location = self.locations[index]

        pdL = np.log(previous_distance_distribution.pdf(la.norm(selected_location - previous_location)))
        ndL = np.log(next_distance_distribution.pdf(la.norm(selected_location - next_location)))
        alL = np.log(spatial_distribution.pdf(selected_location - previous_location))
        L = self.beta * (pdL + ndL) + self.alpha * alL

        new_proposal_likelihood = L
        #previous_proposal_likelihood = prev_L

        return index, 0# previous_proposal_likelihood - new_proposal_likelihood
