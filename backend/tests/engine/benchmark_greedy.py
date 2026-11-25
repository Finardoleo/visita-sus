"""Compara a construção gulosa anterior com a busca local em instâncias sintéticas pequenas.

Execute de backend/: python -m tests.engine.benchmark_greedy
"""

import statistics
from time import perf_counter

from engine import greedy, milp
from engine.evaluation import timing
from engine.instance import Instance, Params
from tests.engine.test_greedy import random_instance


def greedy_before_local_search(inst: Instance, params: Params) -> list[int]:
    """Reproduz a construção gulosa anterior à etapa de busca local."""
    route = greedy.urgent_route(inst)
    first_pos = len(route) - 1
    limit = greedy.graves_limit(inst, route)
    graves = sum(inst.visits[i].grave for i in route)
    H, walk, _ = timing(inst, route)
    pending = sorted(i for i in inst.candidates if not inst.visits[i].urgent)

    while pending:
        best = None
        for candidate in pending:
            visit = inst.visits[candidate]
            if visit.grave and graves >= limit:
                continue
            insertion = greedy.best_insertion(inst, params, route, candidate, first_pos)
            if insertion is None:
                continue
            position, next_H, next_walk = insertion
            gain = params.beta * visit.penalty
            overtime_delta = max(0.0, next_H - inst.T) - max(0.0, H - inst.T)
            delta = next_walk - walk + params.gamma * overtime_delta - gain
            if delta >= 0:
                continue
            ratio = gain / max(next_H - H, 1e-9)
            if best is None or ratio > best[0]:
                best = (ratio, candidate, position, next_H, next_walk)
        if best is None:
            break
        _, candidate, position, H, walk = best
        route.insert(position, candidate)
        pending.remove(candidate)
        graves += inst.visits[candidate].grave
    return route


def main() -> None:
    params = Params(beta=30, gamma=2)
    results: list[tuple[int, float, float, float]] = []
    started = perf_counter()
    for seed in range(1000, 1080):
        n = 6 + seed % 4
        inst = random_instance(seed, n=n, Tmax=110 + 10 * (seed % 3), K=1)
        baseline = greedy.build_solution(
            inst, params, greedy_before_local_search(inst, params), "guloso_base"
        )
        improved = greedy.solve(inst, params)
        optimum = milp.solve(inst, params, time_limit_s=30)
        if not baseline.feasible or not improved.feasible:
            raise RuntimeError(f"solução inviável na semente {seed}")
        if optimum.gap is None or optimum.gap >= 1e-6:
            raise RuntimeError(f"MILP não provou o ótimo na semente {seed}: gap={optimum.gap}")
        if improved.objective > baseline.objective + 1e-9:
            raise RuntimeError(f"busca local piorou a solução na semente {seed}")
        results.append((n, baseline.objective, improved.objective, optimum.objective))

    baseline_optima = sum(abs(old - optimal) < 1e-6 for _, old, _, optimal in results)
    improved_optima = sum(abs(new - optimal) < 1e-6 for _, _, new, optimal in results)
    improvements = sum(new < old - 1e-6 for _, old, new, _ in results)
    def mean_gap(index: int) -> float:
        return statistics.fmean(
            (row[index] - row[3]) / abs(row[3]) * 100 for row in results
        )

    reduction = statistics.fmean(
        (old - new) / abs(old) * 100 for _, old, new, _ in results
    )

    print(f"Casos: {len(results)}; MILP ótimo provado: {len(results)}/{len(results)}")
    print(f"Ótimos: guloso base {baseline_optima}/{len(results)}; com busca local {improved_optima}/{len(results)}")
    print(f"Melhorou {improvements}; manteve {len(results) - improvements}; piorou 0")
    print(f"Gap médio para MILP: base {mean_gap(1):.4f}%; busca local {mean_gap(2):.4f}%")
    print(f"Redução média do objetivo vs. base: {reduction:.4f}%")
    print(f"Tempo total do benchmark: {perf_counter() - started:.2f} s")
    for n in sorted({row[0] for row in results}):
        subset = [row for row in results if row[0] == n]
        old_opt = sum(abs(old - optimal) < 1e-6 for _, old, _, optimal in subset)
        new_opt = sum(abs(new - optimal) < 1e-6 for _, _, new, optimal in subset)
        gap = statistics.fmean(
            (new - optimal) / abs(optimal) * 100 for _, _, new, optimal in subset
        )
        print(f"n={n}: ótimos base {old_opt}/{len(subset)}; busca local {new_opt}/{len(subset)}; "
              f"gap médio novo {gap:.3f}%")


if __name__ == "__main__":
    main()
