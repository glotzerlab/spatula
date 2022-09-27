import itertools

import numpy as np


def permutations(seq, parity="both"):
    for permutation in _permutation_helper(seq):
        cnt = 0
        copy = np.copy(seq)
        for i, j in enumerate(permutation):
            if i != j:
                cnt += 1
                tmp = copy[i]
                copy[i] = copy[j]
                copy[j] = tmp
        if (
            parity == "both"
            or (parity == "even" and cnt % 2 == 0)
            or (parity == "odd" and cnt % 2 == 1)
        ):
            yield copy


def _permutation_helper(seq):
    permutations = [np.arange(len(seq))]
    for i in range(len(seq)):
        new_permutations = []
        for permutation in permutations:
            for j in range(i + 1, len(seq)):
                new_permutation = np.copy(permutation)
                new_permutation[i] = j
                new_permutations.append(new_permutation)
        permutations.extend(new_permutations)
    return permutations


def negations(seq):
    negs = [np.copy(seq)]
    for i in range(len(seq)):
        new_negations = []
        for negation in negs:
            if negation[i] != 0:
                new_negation = np.copy(negation)
                new_negation[i] *= -1
                new_negations.append(new_negation)
        negs.extend(new_negations)
    return negs


def iter_sph_indices(max_l):
    for l in range(max_l + 1):
        ms = range(-l, l + 1)
        for mprime, m in itertools.product(ms, repeat=2):
            yield l, mprime, m
