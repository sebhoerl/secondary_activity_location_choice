import pickle
import pyproj
import numpy as np
import os
import xml.sax
import gzip
import shapefile
import matplotlib.path as mplPath
from tqdm import tqdm

# TODO: What about remote home?

ACTIVITY_TYPES = set([
    "work",
    "education",
    "shop",
    "leisure",
    "remote_work",
    "escort_kids",
    "escort_other",
    "home"
])

IGNORED_ACTIVITY_TYPES = set([
    "work", "home", "cbWork", "cbHome", "cbShop", "cbLeisure", "remote_home"
])

MODES = set([
    "car",
    "pt",
    "bike",
    "walk"
])

PURPOSE_MAP = {
    '-99' : None,
    '1': '#umsteigen',
    '2': 'work',
    '3': 'education',
    '4': 'shop',
    '5': 'shop',
    '6': 'leisure',
    '7': 'remote_work',
    '8': 'leisure',
    '9': 'escort_kids',
    '10': 'escort_other',
    '11': 'home',
    '12': None
}

MODE_MAP = {
    '-99' : None,
    '1' : None,
    '2' : 'pt', # Bahn
    '3' : 'pt', # Postauto
    '4' : 'pt', # Schiff
    '5' : 'pt', # Tram
    '6' : 'pt', # Bus
    '7' : 'pt', # Sonstig OEV
    '8' : 'car', # Reisecar
    '9' : 'car', # Auto
    '10' : None,
    '11' : None,
    '12' : None,
    '13' : None,
    '14' : "bike", #'bike',
    '15' : "walk", #'walk',
    '16' : None,
    '17' : None
}

CENSUS_SELECTOR = ["wmittel", "w_dist_obj2", "wzweck1", "Z_X", "Z_Y", "S_X", "S_Y", "Z_BFS"]
CENSUS_TYPES = [lambda x: MODE_MAP[x], float, lambda x: PURPOSE_MAP[x], float, float, str, float, float, str]

def extract_line(line):
    return [e.strip().replace('"', '') for e in line.split(';')]

class CensusData:
    def __init__(self, raw):
        self.raw = raw

        keep = self.raw[:,1].astype(np.float) >= 0.0
        self.raw = self.raw[keep, :]

        self.activity_locations = None

        self.modes = []
        self.types = []

        self.mode_masks = {}
        self.type_masks = {}

        self._prepare_masks()
        self._prepare_activity_locations()

    def _prepare_activity_locations(self):
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        lv03p = pyproj.Proj("+init=EPSG:2056")

        x, y = pyproj.transform(wgs84, lv03p, self.raw[:,3].astype(np.float), self.raw[:,4].astype(np.float))

        x, y = np.array(x).reshape((len(x), 1)), np.array(y).reshape((len(y), 1))
        coords = np.hstack((x, y))

        self.activity_locations = coords

    def _prepare_masks(self, mode = None, activity_type = None):
        self.modes = list(np.unique(self.raw[:,0]))
        self.types = list(np.unique(self.raw[:,2]))

        for mode in self.modes:
            self.mode_masks[mode] = self.raw[:,0] == mode

        for type in self.types:
            self.type_masks[type] = self.raw[:,2] == type

    def get_distances(self, mode = None, activity_type = None):
        if activity_type is None or activity_type not in self.types: activity_type = None
        if mode is None or mode not in self.modes: mode = None

        mask = np.array([True] * len(self.raw))
        if mode is not None: mask &= self.mode_masks[mode]
        if activity_type is not None: mask &= self.type_masks[activity_type]

        return self.raw[mask, 1].astype(np.float) * 1000

    def get_activity_locations(self, activity_type = None):
        if activity_type is None or activity_type not in self.types: activity_type = None

        if activity_type is None:
            return self.activity_locations
        else:
            return self.activity_locations[self.type_masks[activity_type]]

def load_census_data(settings):
    print("Loading census data ...")
    if os.path.isfile(settings['cache'] + '/census_data.pickle'):
        with open(settings['cache'] + '/census_data.pickle', 'rb') as f:
            print("from cache!")
            return pickle.load(f)

    data = []
    headers = list()
    hmap = dict()
    first = True

    with open(settings['census']) as f:
        for line in tqdm(f):
            if first:
                headers = extract_line(line)
                hmap = { v : k for k, v in enumerate(headers) }
                first = False
            else:
                line = extract_line(line)
                ldata = [t(line[hmap[s]]) for t, s in zip(CENSUS_TYPES, CENSUS_SELECTOR)]

                if np.all([d is not None for d in ldata]):
                    data.append(ldata)

    data = CensusData(np.array(data))

    with open(settings['cache'] + '/census_data.pickle', 'wb+') as f:
        pickle.dump(data, f)

    return data

class FacilityData:
    def __init__(self, raw, types):
        self.raw = raw
        self.types = types

        self.id2index = None
        self.type2facilities = None

        self._prepare_id2index()
        self._prepare_facilities_by_activity_type()

    def _prepare_id2index(self):
        self.id2index = {
            f[0] : index for index, f in enumerate(self.raw)
        }

    def _prepare_facilities_by_activity_type(self):
        self.type2facilities = {
            type : [] for type in self.types
        }

        for f in self.raw:
            for t in f[3]:
                self.type2facilities[t].append(self.id2index[f[0]])

        self.type2facilities["any"] = np.arange(len(self.id2index))

    def get_facility_indices(self):
        return self.type2facilities

    def get_facility_masks(self):
        masks = {}

        for k, v in self.type2facilities.items():
            mask = np.array([False] * len(self.id2index))
            mask[v] = True
            masks[k] = v

        return masks

    def get_locations(self):
        return np.array(self.raw)[:,1:3].astype(np.float)

    def get_index_map(self):
        return self.id2index

class FacilityReader(xml.sax.ContentHandler):
    def __init__(self):
        self.facilities = []
        self.activity_types = set()
        self.bar = tqdm()

    def startElement(self, name, attributes):
        self.bar.update(1)

        if name == "facility":
            self.facilities.append([
                attributes['id'],
                attributes['x'],
                attributes['y'],
                set()
            ])

        if name == "activity":
            self.facilities[-1][3].add(attributes['type'])
            self.activity_types.add(attributes['type'])

def load_facility_data(settings):
    print("Loading facility data ...")
    if os.path.isfile(settings['cache'] + '/facility_data.pickle'):
        with open(settings['cache'] + '/facility_data.pickle', 'rb') as f:
            print("from cache!")
            return pickle.load(f)

    with gzip.open(settings['facilities']) as f:
        reader = FacilityReader()
        parser = xml.sax.make_parser()
        parser.setContentHandler(reader)
        parser.setFeature(xml.sax.handler.feature_validation, False)
        parser.setFeature(xml.sax.handler.feature_external_ges, False)
        parser.parse(f)

    data = FacilityData(reader.facilities, reader.activity_types)

    with open(settings['cache'] + '/facility_data.pickle', 'wb+') as f:
        pickle.dump(data, f)

    return data

class PopulationReader(xml.sax.ContentHandler):
    def __init__(self, id2index):
        self.id2index = id2index
        self.activities = []

        self.selected_plan = False
        self.plan = []

        self.bar = tqdm()

    def startElement(self, name, attributes):
        self.bar.update(1)

        if name == "plan" and attributes["selected"] == "yes":
            self.plan = []
            self.selected_plan = True

        if name =="act" and self.selected_plan:
            self.plan.append(attributes)

        if name == "leg" and self.selected_plan:
            self.plan.append(attributes)

    def endElement(self, name):
        if name == "plan" and self.selected_plan:
            self._process_plan(self.plan)
            self.plan = []
            self.selected_plan = False

    def _process_plan(self, plan):
        activities = []
        current_mode = None
        # [ previous_activity_index, next_activity_index, activity_type, activity_mode, facility_index  ]

        offset = len(self.activities)

        for i in range(len(plan)):
            if i % 2 == 0: # ACTIVITY
                activity = plan[i]

                activities.append([
                    None if i == 0 else len(activities) + offset - 1,
                    len(activities) + 2 + offset - 1,
                    activity['type'],
                    current_mode,
                    self.id2index[activity['facility']]
                ])
            else: # LEG
                leg = plan[i]
                current_mode = leg['mode']

        activities[-1][1] = None

        self.activities += activities

class ChainData:
    def __init__(self, raw):
        self.raw = raw

    def get_activities(self):
        return self.raw

def load_chain_data(settings, facility_data):
    print("Loading population data ...")
    if os.path.isfile(settings['cache'] + '/chain_data.pickle'):
        with open(settings['cache'] + '/chain_data.pickle', 'rb') as f:
            print("from cache!")
            return pickle.load(f)

    with gzip.open(settings['population']) as f:
        reader = PopulationReader(facility_data.get_index_map())
        parser = xml.sax.make_parser()
        parser.setContentHandler(reader)
        parser.setFeature(xml.sax.handler.feature_validation, False)
        parser.setFeature(xml.sax.handler.feature_external_ges, False)
        parser.parse(f)

    data = ChainData(reader.activities)

    with open(settings['cache'] + '/chain_data.pickle', 'wb+') as f:
        pickle.dump(data, f)

    return data

class District:
    def __init__(self, paths, center):
        self.center = center
        self.paths = paths

    def contains(self, c):
        index = sum([1 for path in self.paths if path.contains_point(c)])
        return index % 2 == 1

def load_district_data(settings):
    print("Loading district data ...")
    if os.path.isfile(settings['cache'] + '/districts.pickle'):
        with open(settings['cache'] + '/districts.pickle', 'rb') as f:
            print("from cache!")
            return pickle.load(f)

    shp = shapefile.Reader(settings['districts'])
    shapes = shp.shapes()
    records = shp.records()

    districts = []

    for shape, record in tqdm(zip(shapes, records), total = len(shapes)):
        points = np.array(shape.points)

        lv03 = pyproj.Proj("+init=EPSG:21781")
        lv03p = pyproj.Proj("+init=EPSG:2056")

        center = record[9:11]
        center[0], center[1] = pyproj.transform(lv03, lv03p, float(center[0]), float(center[1]))

        x, y = pyproj.transform(lv03, lv03p, points[:,0].astype(np.float), points[:,1].astype(np.float))
        x, y = np.array(x).reshape((len(x), 1)), np.array(y).reshape((len(y), 1))
        points = np.hstack((x, y))

        parts = shape.parts
        parts.append(len(points))

        paths = []

        for i in range(len(parts) - 1):
            paths.append(mplPath.Path(points[parts[i]:parts[i+1],:]))

        districts.append(District(paths, np.array(center)))

    with open(settings['cache'] + '/districts.pickle', 'wb+') as f:
        pickle.dump(districts, f)

    return districts

class Bin2Indices:
    def __init__(self, indices):
        self.indices = indices

    def get_indices(self, b, t):
        if not b in self.indices: return None
        indices = self.indices[b][t]
        return indices if len(indices) > 0 else None

def load_bin2indices(settings, distribution_factory, facility_data):
    print("Reading bin2indices...")
    facility_indices = facility_data.get_facility_indices()
    locations = facility_data.get_locations()

    if os.path.isfile(settings['cache'] + '/bin2indices.pickle'):
        with open(settings['cache'] + '/bin2indices.pickle', 'rb') as f:
            print("from cache!")
            return pickle.load(f)

    indices = {}
    spatial = distribution_factory.get_spatial_distribution()

    for index, location in tqdm(enumerate(locations), total = len(locations)):
        b = spatial.get_bin(location)

        if not b in indices:
            indices[b] = { t : [] for t in ACTIVITY_TYPES }

        for t in ACTIVITY_TYPES:
            if index in facility_indices[t]:
                indices[b][t].append(index)

    indices = Bin2Indices(indices)

    with open(settings['cache'] + '/bin2indices.pickle', 'wb+') as f:
        pickle.dump(indices, f)

    return indices





















#
