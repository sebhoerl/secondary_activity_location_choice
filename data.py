import pickle
import pyproj
import numpy as np
import os
import xml.sax
import gzip

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
    "work", "home"
])

MODES = set([
    "car",
    "pt"
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
    '14' : None, #'bike',
    '15' : None, #'walk',
    '16' : None,
    '17' : None
}

CENSUS_SELECTOR = ["wmittel", "w_dist_obj2", "wzweck1", "Z_X", "Z_Y", "S_X", "S_Y"]
CENSUS_TYPES = [lambda x: MODE_MAP[x], float, lambda x: PURPOSE_MAP[x], float, float, str, float, float, str]
CENSUS_SOURCE = "Sebastian.csv"

FACILITIES_SOURCE = "ch_1/facilities.xml.gz"
CHAINS_SOURCE = "ch_1/population.xml.gz"

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

def load_census_data():
    if os.path.isfile('cache/census_data.pickle'):
        with open('cache/census_data.pickle', 'rb') as f:
            return pickle.load(f)

    data = []
    headers = list()
    hmap = dict()
    first = True

    with open(CENSUS_SOURCE) as f:
        for line in f:
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

    with open('cache/census_data.pickle', 'wb+') as f:
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

    def startElement(self, name, attributes):
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

def load_facility_data():
    if os.path.isfile('cache/facility_data.pickle'):
        with open('cache/facility_data.pickle', 'rb') as f:
            return pickle.load(f)

    with gzip.open(FACILITIES_SOURCE) as f:
        reader = FacilityReader()
        parser = xml.sax.make_parser()
        parser.setContentHandler(reader)
        parser.setFeature(xml.sax.handler.feature_validation, False)
        parser.setFeature(xml.sax.handler.feature_external_ges, False)
        parser.parse(f)

    data = FacilityData(reader.facilities, reader.activity_types)

    with open('cache/facility_data.pickle', 'wb+') as f:
        pickle.dump(data, f)

    return data

class PopulationReader(xml.sax.ContentHandler):
    def __init__(self, id2index):
        self.id2index = id2index
        self.activities = []

        self.selected_plan = False
        self.plan = []

    def startElement(self, name, attributes):
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

def load_chain_data(facility_data):
    if os.path.isfile('cache/chain_data.pickle'):
        with open('cache/chain_data.pickle', 'rb') as f:
            return pickle.load(f)

    with gzip.open(CHAINS_SOURCE) as f:
        reader = PopulationReader(facility_data.get_index_map())
        parser = xml.sax.make_parser()
        parser.setContentHandler(reader)
        parser.setFeature(xml.sax.handler.feature_validation, False)
        parser.setFeature(xml.sax.handler.feature_external_ges, False)
        parser.parse(f)

    data = ChainData(reader.activities)

    with open('cache/chain_data.pickle', 'wb+') as f:
        pickle.dump(data, f)

    return data




























#
