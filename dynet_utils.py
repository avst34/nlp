def get_activation_function(name):
    import dynet
    ACTIVATIONS = {
        'tanh': dynet.tanh,
        'cube': dynet.cube,
        'relu': dynet.rectify
    }

    if name not in ACTIVATIONS:
        raise Exception('No such activation:' + name)
    return ACTIVATIONS[name]
