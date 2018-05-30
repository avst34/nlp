import os

from datasets.pp_attachement.boknilev.build_dataset import collect_pp_annotations
from datasets.streusle_v4 import StreusleLoader

BASE_PATH = os.path.dirname(__file__)

def match_anns_to_records():
    pass

def build_dataset(boknilev_input_base_path=BASE_PATH + '/streusle'):
    annotations = collect_pp_annotations([boknilev_input_base_path], BASE_PATH + '/annotations.json')
    s_train, s_dev, s_test = StreusleLoader().load()