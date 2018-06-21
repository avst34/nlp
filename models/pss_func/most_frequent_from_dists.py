def most_frequent_from_dists(dists):
    return {
        key: max(dist.items(), key=lambda x: x[1])[0]
        for key, dist in dists.items()
    }
