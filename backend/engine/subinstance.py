"""Subinstâncias pequenas de uma instância real, para comparar a ALNS com o MILP (referência exata)."""

import numpy as np

from .instance import Instance


def subinstance(inst: Instance, n: int, seed: int = 0) -> Instance:
    """
    UBS + n candidatas sorteadas (sem reposição) da instância, preservando a ordem original (urgentes primeiro).
    Um subconjunto das urgentes continua cabendo na jornada, pois o conjunto inteiro cabia (pré-processamento).
    """
    cands = list(inst.candidates)
    if n >= len(cands):
        chosen = cands
    else:
        mandatory = list(inst.fixed_prefix)
        remaining = [i for i in cands if i not in mandatory]
        sampled = np.random.default_rng(seed).choice(
            remaining, size=max(0, n - len(mandatory)), replace=False
        ).tolist()
        chosen = sorted([*mandatory, *sampled])
    idx = [0, *chosen]
    return Instance(visits=[inst.visits[i] for i in idx], t=inst.t[np.ix_(idx, idx)], T=inst.T, Tmax=inst.Tmax,
                    K=inst.K, elapsed_min=inst.elapsed_min, return_node=inst.return_node,
                    fixed_prefix=[1 + chosen.index(i) for i in inst.fixed_prefix])
