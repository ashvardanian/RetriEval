import numpy as np


def recall(ids, groundtruth, ats=(1, 5, 10)):
    recalls = []
    for at in ats:
        matches = (
            (ids[:, :at] == groundtruth[:, np.newaxis]).astype(np.float32).sum(axis=1)
        )
        recalls.append(matches.mean())

    return recalls
