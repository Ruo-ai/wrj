"""Placement utilities from onboard observed free space, never arena truth."""
import heapq
import math

import numpy as np

from swarm_rescue.solutions.my_drone_red_advanced2 import (
    _M16_reachable, _M16_path_to, _M16_PlacementPlan, _M17_transport_costs,
)


def placement_due(initial, inventory, identifier):
    placed = max(0, initial-inventory)
    # Keep the second charge for later exploration, with >= 11% before fallback.
    return (.28 if placed == 0 else .53) + .008*identifier


def walked_costs(graph, step):
    if not graph.points:
        return {}
    costs = {0: 0.}
    queue = [(0., 0)]
    while queue:
        value, node = heapq.heappop(queue)
        if value != costs[node]:
            continue
        for other, length in graph.edges[node].items():
            if graph.blocked.get(tuple(sorted((node, other))), 0) > step:
                continue
            new = value+length
            if new < costs.get(other, math.inf):
                costs[other] = new
                heapq.heappush(queue, (new, other))
    return costs


def difficulty_features(point, transport, root, size, first_visit=0, max_steps=5400):
    """Bounded ranking features; these are not official difficulty scores."""
    if transport is None or root is None or not math.isfinite(transport):
        return dict(depth=0., detour=0., late=0.)
    scale = max(1., min(size))
    direct = float(np.linalg.norm(np.asarray(point)-root))
    return dict(depth=min(1., transport/scale),
                detour=min(1., max(0., transport-direct)/(.5*scale)),
                late=min(1., max(0., first_visit)/max(1., .55*max_steps)))


def rank(gain, steps, features):
    return gain/(1+.018*steps)*(1+.9*features['depth']+.5*features['detour']+.2*features['late'])


def choose_deep_placement(grid, position, field, center, graph, first_visits,
                          step, max_steps, event=None):
    distances, parents = _M16_reachable(grid, position, 40)
    if not distances:
        return None
    # Prefer a semantically observed disposal center. Own walked launch is a
    # fallback depth proxy, not an assertion about blue's spawn or its route.
    root = center if center is not None else (graph.points[0] if graph.points else None)
    transport = _M17_transport_costs(grid, root)
    graph_cost = walked_costs(graph, step) if center is None else {}
    node_cells = {}
    for node, cost in graph_cost.items():
        cell = grid.cell(graph.points[node])
        if grid.line_clear(grid.point(cell), graph.points[node]):
            node_cells[cell] = min(node_cells.get(cell, math.inf), cost)

    def features(point, cell):
        cost = transport.get(cell)
        # No extrapolation through unknown or disconnected cells.
        if cost is None:
            cost = node_cells.get(cell)
        return difficulty_features(point, cost, root, (grid.width, grid.height),
                                   first_visits.get(cell, 0), max_steps)

    coarse = []
    for cell, steps in distances.items():
        point = grid.point(cell)
        if any(np.linalg.norm(point-p) < 85 for p in field.placed):
            continue
        gain = field.sampled_value(point)
        if gain > 0:
            coarse.append((cell, steps, gain, features(point, cell)))
    if not coarse:
        return None
    max_gain = max(c[2] for c in coarse)
    # Do not sacrifice most wall coverage for an empty distant corner.
    coarse = [c for c in coarse if c[2] >= .65*max_gain]
    coarse.sort(key=lambda c: rank(c[2], c[1], c[3]), reverse=True)
    best = None
    selected = []
    for cell, steps, _, f in coarse:
        point = grid.point(cell)
        if any(np.linalg.norm(point-grid.point(c)) < .4*field.radius for c in selected):
            continue
        selected.append(cell)
        for dx in (-8., 0., 8.):
            for dy in (-8., 0., 8.):
                target = point+np.array([dx, dy])
                if (abs(target[0]) >= grid.width/2 or abs(target[1]) >= grid.height/2
                        or not grid.line_clear(point, target)
                        or any(np.linalg.norm(target-p) < 85 for p in field.placed)):
                    continue
                gain = field.value(target)
                if gain < .65*max_gain:
                    continue
                score = rank(gain, steps+np.linalg.norm(target-point)/grid.CELL, f)
                if best is None or score > best[0]:
                    best = (score, cell, target, gain, f)
        if len(selected) >= 12:
            break
    if best is None:
        return None
    _, cell, target, gain, f = best
    if event is not None:
        event('deep_placement_selected', target=target.tolist(), gain=gain,
              best_local_gain=max_gain, route_cells=distances[cell], **f,
              root_kind='observed_disposal' if center is not None else 'own_walked_launch')
    return _M16_PlacementPlan(_M16_path_to(parents, cell), target, gain)
