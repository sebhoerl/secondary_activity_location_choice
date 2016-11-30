import data
import utils
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import os, pickle
import models
import proposal
import sys
import argparse
from tqdm import tqdm
import time
import shutil
import datetime

# Input parsing

argument_parser = argparse.ArgumentParser(description="Static Secondary Location Choice")

argument_parser.add_argument("--model", default = "aggregate", choices = ("full", "aggregate"))
argument_parser.add_argument("--data", default = "reset", choices = ("keep", "uniform", "reset"))
argument_parser.add_argument("--proposal", default = "random", choices = ("random", "model", "approx"))

argument_parser.add_argument("--spatial-model", default = "grid", choices = ("grid", "districts"))
argument_parser.add_argument("--distance-bins", type=int, default = 40)
argument_parser.add_argument("--distance-histogram", default = "equi_distance", choices = ("equi_distance", "equi_probability"))
argument_parser.add_argument("--spatial-grid-bins-x", type=int, default = 10)
argument_parser.add_argument("--spatial-grid-bins-y", type=int, default = 10)

argument_parser.add_argument("--output", default = "output")
argument_parser.add_argument("--cache", default = "cache")

argument_parser.add_argument("--census", default = "Sebastian.csv")
argument_parser.add_argument("--facilities", default = "ch_1/facilities.xml.gz")
argument_parser.add_argument("--population", default = "ch_1/population.xml.gz")
argument_parser.add_argument("--districts", default = "shp/g2b16")

argument_parser.add_argument("--alpha", default = 1.0, type=float)
argument_parser.add_argument("--beta", default = 1.0, type=float)

argument_parser.add_argument("--interval", type=int, default = 10)
argument_parser.add_argument("--no-cleanup", default = "no", choices=["no", "yes"])
#argument_parser.add_argument("--activities", default = None)

settings = vars(argument_parser.parse_args())
config_has_changed = settings["no_cleanup"] == "no"

if not os.path.exists(settings['cache']):
    os.mkdir(settings['cache'])

if not os.path.exists(settings['output']):
    os.mkdir(settings['output'])

if os.path.exists(settings['cache'] + "/settings.pickle"):
    with open(settings['cache'] + "/settings.pickle", "rb") as f:
        previous_settings = pickle.load(f)

        if settings == previous_settings:
            config_has_changed = False

# Directory preparation
if config_has_changed:
    if os.path.exists(settings['cache']):
        shutil.rmtree(settings['cache'])
        os.mkdir(settings['cache'])

    if os.path.exists(settings['output']):
        shutil.rmtree(settings['output'])
        os.mkdir(settings['output'])

with open(settings['cache'] + "/settings.pickle", "wb+") as f:
    pickle.dump(settings, f)

# Load all the stuff

census_data = data.load_census_data(settings)
facility_data = data.load_facility_data(settings)
chain_data = data.load_chain_data(settings, facility_data)
district_data = data.load_district_data(settings)

distribution_factory = utils.DistributionFactory(
    census_data, district_data,
    settings['spatial_model'],
    settings['distance_bins'],
    (settings['spatial_grid_bins_x'], settings['spatial_grid_bins_y']),
    settings['distance_histogram'] == "equi_probability"
)
# TODO: Somehow warm this up?

if settings['proposal'] == "approx":
    bin2indices = data.load_bin2indices(settings, distribution_factory, facility_data)

# Load activities
# TODO: Load from advanced file!
activities = chain_data.get_activities()
facility_indices = facility_data.get_facility_indices()
locations = facility_data.get_locations()

# Set up facilities

if settings['data'] == "keep":
    pass
elif settings['data'] == "uniform":
    print("Assigning facilities uniformly ...")
    for activity in tqdm(activities):
        if not activity[2] in data.IGNORED_ACTIVITY_TYPES:
            indices = facility_indices[activity[2]]
            activity[4] = indices[np.random.randint(len(indices))]
elif settings['data'] == "reset":
    print("Resetting facilities to home ...")

    homes = utils.find_homes(activities)
    for index, activity in enumerate(tqdm(activities)):
        if activity[2] in data.IGNORED_ACTIVITY_TYPES: continue
        if index in homes: activity[4] = homes[index]
else:
    raise "Unknown data mode"

# Set up model

if settings['model'] == "full":
    model = models.HybridDistributionModel()
    covered_activity_types = data.ACTIVITY_TYPES - data.IGNORED_ACTIVITY_TYPES

    model.add(models.SpatialDistributionModel(activities, locations, distribution_factory), name = "spatial")
    for activity_type in covered_activity_types:
        model.add(models.SpatialDistributionModel(activities, locations, distribution_factory, activity_type = activity_type), name = "spatial_" + activity_type)

    model.add(models.DistanceDistributionModel(activities, locations, distribution_factory), name="dist")

    for activity_type in covered_activity_types:
        model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, activity_type = activity_type), name = "dist_" + activity_type)

    for mode in data.MODES:
        model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, mode = mode), name = "dist_" + mode)

    #for activity_type in covered_activity_types:
    #    for mode in data.MODES:
    #        model.add(models.DistanceDistributionModel(activities, locations, distribution_factory, mode = mode, activity_type = activity_type), name = "dist_" + mode + "_" + activity_type)
elif settings['model'] == "aggregate":
    model = models.HybridDistributionModel()
    model.add(models.SpatialDistributionModel(activities, locations, distribution_factory), name = "spatial", factor = settings['alpha'])
    model.add(models.DistanceDistributionModel(activities, locations, distribution_factory), name="dist", factor = settings['beta'])
else:
    raise "Unknown model type"

# Proposal distribution

if settings['proposal'] == "random":
    proposal_distribution = proposal.RandomProposalDistribution(activities, facility_indices, locations, distribution_factory)

elif settings['proposal'] == "model":
    proposal_distribution = proposal.ModelBasedProposalDistribution(settings['alpha'], settings['beta'], activities, facility_indices, locations, distribution_factory)

elif settings['proposal'] == "approx":
    proposal_distribution = proposal.ApproximateProposalDistribution(bin2indices, settings['alpha'], settings['beta'], activities, facility_indices, locations, distribution_factory)

else:
    raise "Unknown proposal distribution"

# Start sampling

print("Starting sampling ...")

activity_indices = np.array([
    index for index, activity in enumerate(activities)
    if (
        activity[2] not in data.IGNORED_ACTIVITY_TYPES and
        activity[0] is not None and
        activity[1] is not None
    )
])

likelihood = model.compute_likelihood()
print("Initial log likelihood:", likelihood)

i = 0
k = 0

start_time = time.time()
interval = settings['interval']
next_time = start_time

cmd_interval = 10
next_cmd_time = start_time

overall = []
acceptance_rate = []

accepted_samples = 0
total_samples = 0

while True:
    k += 1
    print("Starting sweep %d ..." % k)

    np.random.shuffle(activity_indices)

    for activity_index in activity_indices:
        activity = activities[activity_index]
        facility_index, proposal_likelihood_difference = None, None

        while facility_index is None:
            facility_index, proposal_likelihood_difference = proposal_distribution.get_proposal(activity_index)
            total_samples += 1

        likelihood_difference = model.compute_likelihood_difference(activity_index, facility_index)
        acceptance_likelihood = likelihood_difference + proposal_likelihood_difference

        if acceptance_likelihood > 0.0 or np.log(np.random.random()) <= acceptance_likelihood:
            accepted_samples += 1
            model.replace(activity_index, facility_index)
            activity[4] = facility_index

            new_likelihood = model.compute_likelihood()
            expected_likelihood = likelihood + likelihood_difference

            assert(abs(new_likelihood - expected_likelihood) < 1e-3)
            likelihood = new_likelihood
        else:
            modified_likelihood = model.compute_likelihood()
            assert(abs(modified_likelihood - likelihood) < 1e-3)

        if time.time() >= next_cmd_time:
            next_cmd_time += cmd_interval

            print()
            print("Runtime: ", str(datetime.timedelta(seconds=next_cmd_time - cmd_interval - start_time)))
            print("Current log likelihood: ", str(likelihood))
            print("Acceptance rate: ", str(accepted_samples / total_samples))
            print("Total samples: ", str(total_samples))
            print()

        if time.time() >= next_time:
            overall.append(likelihood)
            acceptance_rate.append(accepted_samples / total_samples)

            i += 1
            next_time += interval
            stats = [
                "itime " + str(next_time - interval - start_time),
                "time " + str(time.time() - start_time),
                "acceptance " + str(acceptance_rate[-1]),
                "total " + str(total_samples),
                "all " + str(likelihood)
            ]

            if i > 1:
                plt.figure()
                plt.plot(np.arange(i) * interval, overall)
                plt.savefig(settings['output'] + "/performance.png")
                plt.close()

            if i > 1:
                plt.figure()
                plt.plot(np.arange(i) * interval, acceptance_rate)
                plt.savefig(settings['output'] + "/acceptance.png")
                plt.close()

            for j in range(len(model.distributions)):
                lj = model.distributions[j].compute_likelihood()
                distj = model.distributions[j]

                plt.figure()

                if isinstance(model.distributions[j], models.DistanceDistributionModel):
                    plot_range = distj.reference.get_range()
                    plt.plot(np.arange(len(plot_range) - 1), distj.reference.get_epdf() / (plot_range[1:] - plot_range[:-1]), 'k')
                    plt.plot(np.arange(len(plot_range) - 1), distj.distribution.get_epdf() / (plot_range[1:] - plot_range[:-1]), 'r')
                    plt.xticks(np.arange(len(plot_range)), plot_range, rotation='vertical')
                else:
                    plot_ref = model.distributions[j].reference.get_epdf().flatten()
                    plot_dist = model.distributions[j].distribution.get_epdf().flatten()
                    sorter = np.argsort(plot_ref)[::-1]
                    plot_ref = plot_ref[sorter]
                    plot_dist = plot_dist[sorter]
                    plot_range = np.arange(plot_ref.shape[0])
                    plt.plot(plot_range, plot_ref, 'k')
                    plt.plot(plot_range, plot_dist, 'r')

                plt.title("%f / %f" % (lj, likelihood))
                stats.append("%s %f" % (model.names[j], lj))
                plt.savefig(settings['output'] + "/%d_%s.png" % (i, model.names[j]))
                plt.close()

            with open(settings['output'] + "/%d.txt" % i, "w+") as f:
                f.write('\n'.join(stats))

            with open(settings['output'] + "/activities.pickle", "wb+") as f:
                pickle.dump(activities, f)
