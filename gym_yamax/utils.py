from itertools import tee

# https://docs.python.jp/3/library/itertools.html
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def dictzip(d1, d2):
    for key in d1.keys():
        yield key, (d1[key], d2[key])

