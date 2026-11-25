"""
Ponte entre o banco e o engine: monta a Instance do dia de um ACS, roda o solver e grava o Plano.

Grafo e matriz de tempos vêm dos arquivos da instância (Microarea.path), com cache em memória. A matriz
foi calculada na velocidade do gerador (meta.acs.velocidade_m_min); se o ACS tiver outra velocidade, os
tempos são reescalados.

Com PLANEJAMENTO_PARALELO, os solvers rodam num pool de processos compartilhado pelo servidor: os ACS de uma
equipe rodam em paralelo, e requisições simultâneas de ACS diferentes (o painel planeja um ACS por requisição,
para mostrar o progresso de cada um) também, sem disputar o GIL.
"""

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from shapely.geometry import LineString, Point, shape

from engine import export
from engine.instance import Instance, Params, Solution, Visit
from engine.matrix import build_matrix, load_graph, load_matrix, save_matrix
from engine.methods import METHODS, run
from engine.preprocess import Agent, Household, Preprocessed, preprocess

from .models import Agente, Domicilio, Equipe, ItemRoteiro, Microarea, NaoAtendida, Plano

METODOS = METHODS
VELOCIDADE_PADRAO = 75.0

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def pool() -> ProcessPoolExecutor:
    """Pool de processos (spawn) criado na primeira vez e reaproveitado pelas requisições seguintes."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(os.cpu_count() or 1, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _descartar_pool() -> None:
    global _pool
    with _pool_lock:
        _pool = None


def resolver(metodos: list[str], insts: list, params: Params, paralelo: bool | None = None) -> list[Solution]:
    """Roda os solvers (um por instância), no pool se `paralelo` (padrão: settings.PLANEJAMENTO_PARALELO)."""
    paralelo = settings.PLANEJAMENTO_PARALELO if paralelo is None else paralelo
    if paralelo:
        try:
            return list(pool().map(run, metodos, insts, [params] * len(insts)))
        except BrokenProcessPool:  # um processo do pool morreu: recria na próxima vez e roda aqui mesmo
            _descartar_pool()
    return [run(m, inst, params) for m, inst in zip(metodos, insts)]


@lru_cache(maxsize=16)
def _graph(path: str, mtime: float):
    return load_graph(path)


@lru_cache(maxsize=16)
def _matrix(path: str, mtime: float):
    return load_matrix(path)


def grafo(microarea: Microarea):
    path = microarea.path / "graph.graphml"
    return _graph(str(path), path.stat().st_mtime)


def matriz(microarea: Microarea) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de tempos (na velocidade do gerador) entre a UBS e os domicílios; calculada se faltar."""
    path = microarea.path / "matrix.npz"
    required = {microarea.ubs_node, *microarea.domicilios.values_list("node", flat=True)}
    if path.exists():
        nodes, _ = load_matrix(path)
    else:
        nodes, times = np.array([], dtype=np.int64), np.empty((0, 0))
    if not path.exists() or not required.issubset(set(map(int, nodes))):
        nodes = [microarea.ubs_node, *microarea.domicilios.values_list("node", flat=True)]
        save_matrix(path, *build_matrix(grafo(microarea), nodes))
    return _matrix(str(path), path.stat().st_mtime)


def dias_sem_visita(domicilio: Domicilio, data: date) -> int:
    return max(0, (data - domicilio.ultima_visita).days)


def penalidade(domicilio: Domicilio, data: date) -> float:
    return domicilio.peso * (dias_sem_visita(domicilio, data) + 1) / domicilio.intervalo_max_dias


def urgente_na_data(domicilio: Domicilio, data: date) -> bool:
    return domicilio.urgente and (domicilio.urgencia_data is None or domicilio.urgencia_data == data)


def households(microarea: Microarea, data: date) -> list[Household]:
    return [Household(id=d.pk, node=d.node, s=d.duracao_min, w=d.peso, P=d.intervalo_max_dias,
                      ultima_visita=d.ultima_visita, tw=(d.tw_inicio, d.tw_fim),
                      urgent=urgente_na_data(d, data), grave=d.grave)
            for d in microarea.domicilios.all()]


def agent(agente: Agente) -> Agent:
    return Agent(ubs_node=agente.microarea.ubs_node, T=agente.jornada_min, Tmax=agente.jornada_max_min,
                 K=agente.teto_graves)


def montar_instancia(agente: Agente, data: date, params: Params) -> Preprocessed:
    microarea = agente.microarea
    nodes, t = matriz(microarea)
    base = microarea.meta.get("acs", {}).get("velocidade_m_min", VELOCIDADE_PADRAO)
    if agente.velocidade_m_min != base:
        t = t * (base / agente.velocidade_m_min)
    return preprocess(households(microarea, data), agent(agente), nodes, t, data, params)


def planejar(agente: Agente, data: date, metodo: str = "alns", params: Params | None = None) -> Plano:
    params = params or Params()
    pre = montar_instancia(agente, data, params)
    return gravar(agente, data, metodo, params, pre, resolver([metodo], [pre.instance], params)[0])


def _instancia_replanejamento(agente: Agente, origem_node: int, domicilios: list[Domicilio],
                              data: date, elapsed_min: float, params: Params,
                              urgencia_id: int | None) -> Instance:
    microarea = agente.microarea
    ubs_node = microarea.ubs_node
    graph = grafo(microarea)
    households_visits = [
        Visit(
            id=d.pk, node=d.node, s=d.duracao_min, w=d.peso, P=d.intervalo_max_dias,
            d=dias_sem_visita(d, data), tw=(d.tw_inicio, d.tw_fim),
            urgent=urgente_na_data(d, data), grave=d.grave,
        )
        for d in domicilios
    ]
    nodes, matrix = build_matrix(graph, [origem_node, *(v.node for v in households_visits), ubs_node])
    pos = {int(node): i for i, node in enumerate(nodes)}
    ordered_nodes = [origem_node, *(v.node for v in households_visits)]
    n = len(ordered_nodes)
    t = np.empty((n, n), dtype=float)
    for i, source in enumerate(ordered_nodes):
        for j in range(n):
            target = ubs_node if j == 0 else ordered_nodes[j]
            t[i, j] = matrix[pos[int(source)], pos[int(target)]]

    # O nó 0 é a origem atual; a coluna 0 representa a chegada final à UBS.
    t[0, 0] = matrix[pos[int(origem_node)], pos[int(ubs_node)]]
    base_speed = microarea.meta.get("acs", {}).get("velocidade_m_min", VELOCIDADE_PADRAO)
    if agente.velocidade_m_min != base_speed:
        t *= base_speed / agente.velocidade_m_min

    depot = Visit(id=-1, node=origem_node, s=0, w=0, P=0, d=0, tw=(0, agente.jornada_max_min))
    fixed_prefix = [1] if urgencia_id is not None else []
    return Instance(
        visits=[depot, *households_visits], t=t,
        T=min(agente.jornada_min, 360.0, agente.jornada_max_min, 420.0),
        Tmax=min(agente.jornada_max_min, 420.0),
        K=agente.teto_graves, elapsed_min=elapsed_min, return_node=ubs_node, fixed_prefix=fixed_prefix,
    )


@transaction.atomic
def replanejar_urgencia(plano: Plano, dados: dict) -> Plano:
    """Registra um novo domicílio urgente e replaneja apenas o trecho ainda não concluído do dia."""
    from engine import alns, greedy
    from engine.evaluation import timing

    agente = Agente.objects.select_related("microarea__equipe__ubs").get(pk=plano.agente_id)
    microarea = Microarea.objects.select_for_update().get(pk=agente.microarea_id)
    agente.microarea = microarea
    now = timezone.localtime()
    if plano.data != now.date():
        raise ValueError("o replanejamento só pode ser feito no dia do plano")
    shift_start = timezone.make_aware(datetime.combine(plano.data, plano.hora_inicio))
    elapsed_min = (now - shift_start).total_seconds() / 60
    limit = min(agente.jornada_max_min, 420.0)
    if elapsed_min < 0:
        raise ValueError("o turno ainda não começou")
    if elapsed_min >= limit:
        raise ValueError("o turno já atingiu o limite máximo de 7 horas")

    point = Point(dados["lon"], dados["lat"])
    if microarea.poligono and not shape(microarea.poligono).covers(point):
        raise ValueError("o endereço informado fica fora da microárea")
    graph = grafo(microarea)
    import osmnx as ox
    origem_node = int(ox.distance.nearest_nodes(graph, X=dados["acs_lon"], Y=dados["acs_lat"]))
    visit_node = int(ox.distance.nearest_nodes(graph, X=dados["lon"], Y=dados["lat"]))

    negative_codes = list(microarea.domicilios.filter(codigo__lt=0).values_list("codigo", flat=True))
    codigo = min(negative_codes, default=0) - 1
    urgencia = Domicilio.objects.create(
        microarea=microarea, codigo=codigo, node=visit_node, lat=dados["lat"], lon=dados["lon"],
        n_moradores=1, condicao=dados["condicao"], peso=dados["peso"],
        intervalo_max_dias=dados["intervalo_max_dias"], duracao_min=dados["duracao_min"],
        ultima_visita=dados["ultima_visita"], tw_inicio=0, tw_fim=limit, urgente=True, grave=dados["grave"],
        urgencia_data=plano.data, manual=True,
    )

    items = list(plano.itens.select_related("domicilio").filter(fim_min__gt=elapsed_min))
    andamento = next((item for item in items if item.inicio_min <= elapsed_min < item.fim_min), None)
    if andamento:
        elapsed_min = andamento.fim_min
        origem_node = andamento.domicilio.node
        items = [item for item in items if item.pk != andamento.pk]
    restantes = list({item.domicilio_id: item.domicilio for item in items}.values())

    params = params_do_plano(plano)
    tmax = min(agente.jornada_max_min, 420.0)
    # Se a urgência não cabe sozinha no restante do turno, registra-a fora e ainda organiza a rota restante.
    urgencia_isolada = _instancia_replanejamento(
        agente, origem_node, [urgencia], plano.data, elapsed_min, params, urgencia.pk
    )
    h_urg, _, ok_urg = timing(urgencia_isolada, [0, 1, 0])
    descartadas = []
    urgencia_viavel = ok_urg and h_urg <= tmax + 1e-9
    if not urgencia_viavel:
        descartadas.append((urgencia.pk, "urgencia_excedente"))

    candidatas = ([urgencia] if urgencia_viavel else []) + restantes
    fixed_id = urgencia.pk if urgencia_viavel else None
    inst = _instancia_replanejamento(
        agente, origem_node, candidatas, plano.data, elapsed_min, params, fixed_id
    )

    rota_urgente = greedy.urgent_route(inst)
    h_urgentes, _, ok_urgentes = timing(inst, rota_urgente)
    while (not ok_urgentes or h_urgentes > tmax + 1e-9):
        antigas_urgentes = [
            d for d in candidatas if d.pk != fixed_id and urgente_na_data(d, plano.data)
        ]
        if not antigas_urgentes:
            if fixed_id is not None:
                # A urgência nova continua obrigatória; a exceção aponta para urgências antigas incompatíveis.
                raise ValueError("as urgências restantes não cabem no limite de 7 horas")
            break
        retirar = min(antigas_urgentes, key=lambda d: (penalidade(d, plano.data), d.pk))
        candidatas.remove(retirar)
        descartadas.append((retirar.pk, "urgencia_excedente"))
        inst = _instancia_replanejamento(
            agente, origem_node, candidatas, plano.data, elapsed_min, params, fixed_id
        )
        rota_urgente = greedy.urgent_route(inst)
        h_urgentes, _, ok_urgentes = timing(inst, rota_urgente)

    h_retorno, _, ok_retorno = timing(inst, [0, 0])
    if not ok_retorno or h_retorno > tmax + 1e-9:
        raise ValueError("não há tempo para retornar à UBS dentro do limite de 7 horas")

    sol = alns.solve_multi_seed(inst, params)
    pre = Preprocessed(instance=inst, discarded=descartadas)
    novo_plano = gravar(agente, plano.data, "alns", params, pre, sol)
    novo_plano.params = {
        **novo_plano.params,
        "replanejamento": {
            "plano_anterior": plano.pk,
            "tempo_decorrido_min": round(elapsed_min, 2),
            "urgencia_nova": urgencia.pk,
            "urgencia_atendida": novo_plano.itens.filter(domicilio_id=urgencia.pk).exists(),
        },
    }
    novo_plano.save(update_fields=["params"])
    return novo_plano


def planejar_equipe(equipe: Equipe, data: date, metodo: str = "alns", params: Params | None = None,
                    paralelo: bool | None = None) -> list[Plano]:
    """
    Planeja o dia de todos os ACS da equipe. As instâncias são montadas antes (se faltar arquivo de alguma
    microárea, nada é gravado); os solvers rodam em processos separados, um por ACS; os planos são gravados
    juntos, numa transação.
    """
    params = params or Params()
    agentes = list(Agente.objects.filter(microarea__equipe=equipe).select_related("microarea").order_by("microarea__nome"))
    pres = [montar_instancia(a, data, params) for a in agentes]
    sols = resolver([metodo] * len(pres), [p.instance for p in pres], params, paralelo)
    with transaction.atomic():
        return [gravar(a, data, metodo, params, pre, sol) for a, pre, sol in zip(agentes, pres, sols)]


@transaction.atomic
def gravar(agente: Agente, data: date, metodo: str, params: Params, pre: Preprocessed, sol: Solution) -> Plano:
    """Grava o Plano, os itens do roteiro, as não atendidas (com os descartes do pré-processamento) e a rota."""
    inst = pre.instance
    rota = export.route_geojson(grafo(agente.microarea), inst, sol)

    plano = Plano.objects.create(
        agente=agente, data=data, metodo=metodo,
        params={**vars(params), "acs": {"T": inst.T, "Tmax": inst.Tmax, "K": inst.K,
                                        "velocidade_m_min": agente.velocidade_m_min}},
        status=Plano.Status.VIAVEL if sol.feasible else Plano.Status.COM_VIOLACOES,
        violacoes=[{"tipo": v.kind, "domicilio": None if v.visit is None else inst.visits[v.visit].id,
                    "quanto": v.amount} for v in sol.violations],
        hora_inicio=agente.hora_inicio, objetivo=sol.objective, caminhada_min=sol.walk,
        penalidade_residual=sol.residual_penalty, excesso_min=sol.overtime, retorno_min=sol.H,
        n_candidatas=inst.n - 1, n_fora_candidatas=len(pre.not_candidates), runtime_s=sol.runtime_s,
        seed=sol.seed, seeds_stats=sol.seeds_stats, gap=sol.gap, rota_geojson=rota,
    )
    ItemRoteiro.objects.bulk_create([
        ItemRoteiro(plano=plano, ordem=k, domicilio_id=inst.visits[st.visit].id, chegada_min=st.arrival,
                    inicio_min=st.start, fim_min=st.end, caminhada_min=st.walk_from_prev, motivo=st.reason)
        for k, st in enumerate(sol.stops, 1)
    ])
    fora = [(inst.visits[i].id, motivo) for i, motivo in sol.unvisited] + pre.discarded
    doms = Domicilio.objects.in_bulk([pk for pk, _ in fora])
    NaoAtendida.objects.bulk_create([
        NaoAtendida(plano=plano, domicilio_id=pk, motivo=motivo, penalidade=penalidade(doms[pk], data),
                    dias_sem_visita=dias_sem_visita(doms[pk], data),
                    atraso_dias=max(0, dias_sem_visita(doms[pk], data) - doms[pk].intervalo_max_dias))
        for pk, motivo in fora
    ])
    return plano


# ---------- saídas a partir do plano gravado ----------
def inicio(plano: Plano) -> datetime:
    return datetime.combine(plano.data, plano.hora_inicio)


def relogio(plano: Plano, minutos: float) -> str:
    return (inicio(plano) + timedelta(minutes=minutos)).strftime("%H:%M")


def linhas_itinerario(plano: Plano) -> list[export.ItineraryRow]:
    """Itinerário do plano gravado, no formato das exportações do engine (última linha = retorno à UBS)."""
    start = inicio(plano)

    def at(m):
        return start + timedelta(minutes=m)

    rows = []
    for it in plano.itens.select_related("domicilio"):
        d = it.domicilio
        rows.append(export.ItineraryRow(
            ordem=it.ordem, domicilio=d.pk, condicao=d.condicao, motivo=it.motivo, chegada=at(it.chegada_min),
            inicio=at(it.inicio_min), fim=at(it.fim_min), espera_min=it.inicio_min - it.chegada_min,
            caminhada_min=it.caminhada_min, duracao_min=d.duracao_min, w=d.peso, d=dias_sem_visita(d, plano.data),
            P=d.intervalo_max_dias, lat=d.lat, lon=d.lon))
    ubs = plano.agente.microarea.equipe.ubs
    rows.append(export.ItineraryRow(
        ordem=len(rows) + 1, domicilio=export.UBS_ID, condicao="", motivo="retorno", chegada=at(plano.retorno_min),
        inicio=at(plano.retorno_min), fim=at(plano.retorno_min), espera_min=0.0,
        caminhada_min=plano.caminhada_min - sum(r.caminhada_min for r in rows), duracao_min=0.0, w=0, d=0, P=0,
        lat=ubs.lat, lon=ubs.lon))
    return rows


def rota_linha(plano: Plano) -> LineString | None:
    coords = (plano.rota_geojson or {}).get("coordinates") or []
    return LineString(coords) if len(coords) >= 2 else None


def nome_arquivo(plano: Plano, ext: str) -> str:
    return f"itinerario_{plano.agente.microarea.nome}_{plano.data}_{plano.metodo}_{plano.pk}.{ext}"


def instancia_disponivel(microarea: Microarea) -> bool:
    return Path(microarea.path / "graph.graphml").exists()


@lru_cache(maxsize=16)
def _malha(path: str, mtime: float) -> dict:
    import osmnx as ox
    edges = ox.graph_to_gdfs(load_graph(path), nodes=False)
    seen, features = set(), []
    for (u, v, _), e in edges.iterrows():
        key = frozenset((u, v))
        if key in seen:  # mão e contramão têm a mesma geometria
            continue
        seen.add(key)
        features.append({"type": "Feature", "geometry": e.geometry.__geo_interface__,
                         "properties": {"length_m": round(float(e["length"]), 1),
                                        "highway": e["highway"] if isinstance(e["highway"], str) else str(e["highway"])}})
    return {"type": "FeatureCollection", "features": features}


def malha_geojson(microarea: Microarea) -> dict:
    """Ruas da malha a pé (uma feature por trecho, sem duplicar os dois sentidos)."""
    path = microarea.path / "graph.graphml"
    return _malha(str(path), path.stat().st_mtime)


# ---------- comparação de métodos (validação) ----------
def params_do_plano(plano: Plano) -> Params:
    campos = {f for f in Params.__dataclass_fields__}
    return Params(**{k: v for k, v in plano.params.items() if k in campos})


def comparar(plano: Plano, tempo_milp: float = 20.0, n: int | None = None, seed: int = 0) -> dict:
    """
    Guloso × ALNS × MILP na instância do plano (mesmos ACS, data e parâmetros), ou numa subinstância de n
    candidatas. Nada é gravado. O gap de cada método é relativo ao MILP; o MILP traz o gap provado pelo solver.
    """
    from engine import alns, greedy, milp
    from engine.subinstance import subinstance

    params = params_do_plano(plano)
    inst = montar_instancia(plano.agente, plano.data, params).instance
    if n:
        inst = subinstance(inst, n, seed=seed)
    sols = {"guloso": greedy.solve(inst, params), "alns": alns.solve_multi_seed(inst, params)}
    sols["milp"] = milp.solve(inst, params, time_limit_s=tempo_milp)
    ref = sols["milp"].objective
    G = grafo(plano.agente.microarea)
    metodos = []
    for nome, s in sols.items():
        metodos.append({
            "metodo": nome, "objetivo": s.objective, "caminhada_min": s.walk, "penalidade_residual": s.residual_penalty,
            "excesso_min": s.overtime, "retorno": relogio(plano, s.H), "visitas": len(s.stops), "viavel": s.feasible,
            "runtime_s": s.runtime_s, "gap_para_milp": (s.objective - ref) / abs(ref) if ref else 0.0,
            "gap_solver": s.gap, "seeds_stats": s.seeds_stats,
            "ordem": [inst.visits[st.visit].id for st in s.stops],
            "rota": export.route_geojson(G, inst, s),
        })
    return {"plano": plano.pk, "candidatas": inst.n - 1, "subinstancia": n, "seed": seed, "tempo_milp": tempo_milp,
            "metodos": metodos}
