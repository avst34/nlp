from collections import namedtuple

from models.supersenses import embeddings
from .features import FEATURES, FeatureType


class MountPoint:
    LSTM = 'LSTM'
    MLP = 'MLP'

MountedFeature = namedtuple('InstalledFeature', ['feature', 'mount_point', 'dim'])

def get_feature(feature_name):
    for f in FEATURES:
        if f.name == feature_name:
            return f
    raise Exception("No such feature: %s" % feature_name)

def mount_feature(feature_name, mount_point, dim=None):
    assert hasattr(MountPoint, mount_point)
    assert getattr(MountPoint, mount_point) == mount_point
    feature = get_feature(feature_name)
    if feature.type == FeatureType.REF and mount_point != MountPoint.MLP:
        raise Exception("Feature '%s' of type '%s' can only be mounted to MLP" % (feature.name, feature.type))
    if feature.type != FeatureType.REF:
        assert feature.type == FeatureType.ENUM
        if feature.embedding != embeddings.AUTO:
            embd_dim = len(list(feature.embedding.values())[0])
            assert dim is None or dim == embd_dim
            dim = embd_dim
        else:
            assert dim is not None, "Must provide a vector dimension when embedding is AUTO (feature: '%s')" % feature_name

    return MountedFeature(feature=feature, mount_point=mount_point, dim=dim)
