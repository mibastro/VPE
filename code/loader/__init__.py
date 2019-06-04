import json

from loader.gtsrb import gtsrbLoader
from loader.gtsrb2TT100K import gtsrb2TT100KLoader
from loader.belga2flickr import belga2flickrLoader
from loader.belga2toplogo import belga2toplogoLoader

def get_loader(name):
    return {
        'gtsrb': gtsrbLoader,
        'gtsrb2TT100K': gtsrb2TT100KLoader,
        'belga2flickr': belga2flickrLoader,
        'belga2toplogo': belga2toplogoLoader,
    }[name]


def get_data_path(name, config_file='config.json'):
    data = json.load(open(config_file))
    return data[name]['data_path']
