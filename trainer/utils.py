def dictzip(d1, d2):
    for k, v in d1.items():
        yield k, (v, d2[k])
