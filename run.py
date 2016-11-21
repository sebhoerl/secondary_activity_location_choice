import data
import utils
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import os, pickle
import models
import proposal
import sys

census_data = data.load_census_data()
facility_data = data.load_facility_data()
chain_data = data.load_chain_data(facility_data)

activities = chain_data.get_activities()
facility_indices = facility_data.get_facility_indices()
locations = facility_data.get_locations()

distribution_factory = utils.DistributionFactory(census_data)

if True: # Make facility distribution singleton for non-fixed activity types
    print("Setting up singleton facility")
    for activity in activities:
        if not activity[2] in data.IGNORED_ACTIVITY_TYPES:
            indices = facility_indices[activity[2]]
            activity[4] = indices[np.random.randint(len(indices))]

activity_indices = np.array([
    index for index, activity in enumerate(activities)
    if (
        activity[2] not in data.IGNORED_ACTIVITY_TYPES and
        activity[0] is not None and
        activity[1] is not None
    )
])

if True: # Full hybrid model
    print("Using full hybrid model")
    model = models.HybridDistributionModel()

    model.add(models.SpatialDistributionModel(activities, locations, distribution_factory), name = "spatial")
    for activity_type in (data.ACTIVITY_TYPES ^ data.IGNORED_ACTIVITY_TYPES):
        model.add(models.SpatialDistributionModel(activities, locations, distribution_factory, activity_type = activity_type), name = "spatial_" + activity_type)

    model.add(models.DistanceDistributionModel(activities, locations, distribution_factory), name="dist")

    for activity_type in (data.ACTIVITY_TYPES ^ data.IGNORED_ACTIVITY_TYPES):
        model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, activity_type = activity_type), name = "dist_" + activity_type)

    for mode in data.MODES:
        model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, mode = mode), name = "dist_" + mode)

    for activity_type in (data.ACTIVITY_TYPES ^ data.IGNORED_ACTIVITY_TYPES):
        for mode in data.MODES:
            model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, mode = mode, activity_type = activity_type), name = "dist_" + mode + "_" + activity_type)

if False: # Aggregate model
    print("Using aggregate hybrid model")
    model = models.HybridDistributionModel()
    model.add(models.SpatialDistributionModel(activities, locations, distribution_factory), name = "spatial")
    model.add(models.DistanceDistributionModel(activities, locations, distribution_factory), name="dist")

if sys.argv[1] == "random":
    print("Using random facility sampling")
    proposal_distribution = proposal.RandomProposalDistribution(activities, facility_indices, locations, distribution_factory)

if sys.argv[1] == "model":
    print("Using model-based facility sampling")
    proposal_distribution = proposal.ModelBasedProposalDistribution(1.0, 1.0, activities, facility_indices, locations, distribution_factory)

likelihood = model.compute_likelihood()

i = 0
K = 1000

for k in range(K):
    np.random.shuffle(activity_indices)

    for activity_index in activity_indices:
        activity = activities[activity_index]
        facility_index, proposal_likelihood_difference = proposal_distribution.get_proposal(activity_index)

        likelihood_difference = model.compute_likelihood_difference(activity_index, facility_index)
        acceptance_likelihood = likelihood_difference + proposal_likelihood_difference

        if acceptance_likelihood > 0.0 or np.log(np.random.random()) <= acceptance_likelihood:
            model.replace(activity_index, facility_index)
            activity[4] = facility_index
            likelihood = model.compute_likelihood()

            i += 1
            #print(i)
            if i % 10000 == 0:
                stats = [str(likelihood)]

                for j in range(len(model.distributions)):
                    lj = model.distributions[j].compute_likelihood()
                    plt.figure()
                    plt.plot(model.distributions[j].reference.get_epdf().flatten(), color = "k")
                    plt.plot(model.distributions[j].distribution.get_epdf().flatten(), color = "r")
                    plt.title("%f / %f" % (lj, likelihood))
                    stats.append("%s %f" % (model.names[j], lj))
                    plt.savefig("output/%d_%s.png" % (i, model.names[j]))
                    plt.close()

                with open("output/%d.txt" % i, "w+") as f:
                    f.write('\n'.join(stats))
