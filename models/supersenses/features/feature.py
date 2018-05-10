from collections import namedtuple

from models.supersenses import embeddings


class FeatureType:
    STRING = 'STRING'
    ENUM = 'ENUM'
    REF = 'REF'

class MountPoint:
    LSTM = 'LSTM'
    MLP = 'MLP'

class NoneFeatureValue(Exception):
    pass

class Feature(object):

    def __init__(self, name, type, vocab, embedding, extractor, enable, mount_point, embedding_fallback=None, dim=None, update=False, masked_only=True, fall_to_none=False, default_zero_vec=False):
        self.embedding_fallback = embedding_fallback
        self.default_zero_vec = default_zero_vec
        self.name = name
        self.type = type
        self.vocab = vocab
        self.embedding = embedding
        self.extractor = extractor
        self.enable = enable
        self.mount_point = mount_point
        self.dim = dim
        self.update = update
        self.masked_only = masked_only
        self.fall_to_none = fall_to_none

        self._validate_and_init()

    def _validate_and_init(self):
        assert hasattr(MountPoint, self.mount_point)
        assert getattr(MountPoint, self.mount_point) == self.mount_point
        if self.type == FeatureType.REF and self.mount_point != MountPoint.MLP:
            raise Exception("Feature '%s' of type '%s' can only be mounted to MLP" % (self.name, self.type))
        if self.type != FeatureType.REF:
            assert self.type in (FeatureType.ENUM, FeatureType.STRING)
            assert self.vocab
            if self.embedding != embeddings.AUTO:
                embd_dim = len(list(self.embedding.values())[0])
                assert self.dim is None or self.dim == embd_dim
                self.dim = embd_dim
            else:
                assert self.dim is not None, "Must provide a vector dimension when embedding is AUTO (feature: '%s')" % self.name
                if self.update is None:
                    self.update = True
                assert self.update, "AUTO embeddings must be updatable (feature: '%s')" % self.name

    def extract(self, tok, sent):
        try:
            val = self.extractor(tok, sent)
        except NoneFeatureValue as e:
            val = None

        if self.type == FeatureType.ENUM:
            if not self.vocab.has_word(val):
                if not self.fall_to_none:
                    raise Exception("Error in feature '%s': extracted value '%s' is not in the associated vocabulary (%s)" % (self.name, val, self.vocab.name))
                val = None
        elif self.type == FeatureType.REF:
            if val is not None and (type(val) != int or not (0 <= val < len(sent))):
                raise Exception("Error in feature '%s': extracted value '%s' is not a valid ref to the sentence (len: %d)" % (self.name, str(val), len(sent)))
        else:
            assert self.type == 'STRING'

        return val


class Features(object):

    def __init__(self, features):
        self.features = [f for f in features if f.enable]

    def get_feature(self, feature_name):
        for f in self.features:
            if f.name == feature_name:
                return f
        raise Exception("No such feature: %s" % feature_name)

    def list(self):
        return list(self.features)

    def list_lstm_features(self):
        return [f for f in self.features if f.mount_point == MountPoint.LSTM]

    def list_mlp_features(self, include_refs=True):
        return [f for f in self.features if f.mount_point == MountPoint.MLP and include_refs or f.type != FeatureType.REF]

    def list_ref_features(self):
        return [f for f in self.features if f.type == FeatureType.REF]

    def list_enum_features(self):
        return [f for f in self.features if f.type == FeatureType.ENUM]

    def list_string_features(self):
        return [f for f in self.features if f.type == FeatureType.STRING]

    def list_updatable_features(self):
        return [f for f in self.features if f.update]

    def list_features_with_embedding(self, include_auto=True):
        features = [f for f in self.features if f.embedding and (include_auto or f.embedding != embeddings.AUTO)]
        assert all([f.dim is not None for f in features])
        return features

    def list_default_zero_vec_features(self):
        features = [f for f in self.features if f.default_zero_vec]
        assert all([f.type != FeatureType.REF for f in features])
        return features

    def list_features_with_embedding_fallback(self):
        features = [f for f in self.features if f.embedding_fallback]
        assert all([f.type != FeatureType.REF for f in features])
        return features



