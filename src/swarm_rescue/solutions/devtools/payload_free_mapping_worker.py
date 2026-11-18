"""Run one unarmed mapping-only trial on an existing native simulation map."""

import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback


def check_empty(world):
    if world.number_bombs or world.bombs:
        raise ValueError("An object was introduced into the empty scene")
    for drone in world.drones:
        if drone.bomb_deployer is not None or drone.carried_bombs_count() or drone.grasped_bombs():
            raise ValueError("A payload or deployment device was present")


def fly(request, run_dir, result):
    import cv2
    import numpy as np

    from swarm_rescue.simulation.drone.ground_truth_guard import run_control
    from swarm_rescue.simulation.elements.sensor_disablers import NoComZone, NoGpsZone, ZoneType
    from swarm_rescue.simulation.reporting.place_score_manager import PlaceScoreManager, walls_only_grid
    from swarm_rescue.solutions.devtools.evaluate_red import configure_ray_backend
    from swarm_rescue.solutions.devtools.payload_free_mapper_trial import PayloadFreeMapperTrial

    cv2.setNumThreads(1)
    seed = request["seed"]
    random.seed(seed)
    np.random.seed(seed)
    configure_ray_backend("cpu")
    map_class = importlib.import_module(request["module"]).MapGenerated
    zones = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE)
    world = map_class(PayloadFreeMapperTrial, zones, number_drones=5)
    try:
        pg = world.playground
        pg.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        step_limit = int(request["limit_steps"])
        wall_limit = float(request["limit_seconds"])
        if not (1 <= step_limit <= world.max_timestep_limit and 0 < wall_limit <= world.max_walltime_limit):
            raise ValueError("Requested limit exceeds the native map budget or is nonpositive")
        for zone_type in (NoComZone, NoGpsZone):
            if not any(isinstance(element, zone_type) for element in pg.elements):
                raise ValueError(f"Missing native zone: {zone_type.__name__}")
        if len(world.drones) != 5 or not all(type(d) is PayloadFreeMapperTrial for d in world.drones):
            raise ValueError("Mapping trial requires five mapping-only controllers")
        for drone in world.drones:
            drone.set_competition_mode(True)
        check_empty(world)
        positions = np.full((step_limit, 5, 2), np.nan, dtype=np.float32)
        result.update(zones=[kind.name for kind in zones], drones=len(world.drones), bombs=0,
                      controller="PayloadFreeMapperTrial", warmup_steps=1,
                      map_max_steps=int(world.max_timestep_limit), map_max_seconds=float(world.max_walltime_limit),
                      limit_steps=step_limit, limit_seconds=wall_limit,
                      step_definition="Physics steps including the first command-free sensor warm-up",
                      positions_definition="Post-physics world coordinates per original slot, NaN after removal",
                      coverage_definition="100 * native ExploredMap.score(), 200px wall-constrained trajectory dilation",
                      prediction_definition="Offline union of the five independently estimated binary wall maps")
        started = time.perf_counter()
        allowed = {"forward", "lateral", "rotation"}
        for frame in range(step_limit):
            elapsed = time.perf_counter() - started
            if frame and elapsed >= wall_limit:
                break
            commands = {}
            if frame:
                for drone in world.drones:
                    if drone.removed:
                        continue
                    drone.elapsed_timestep = frame + 1
                    drone.elapsed_walltime = elapsed
                    command = run_control(drone)
                    if set(command) != allowed or not all(
                            math.isfinite(value) and -1 <= value <= 1 for value in command.values()):
                        raise ValueError(f"Unsupported movement command from slot {drone.identifier}")
                    commands[drone] = command
            pg.step(all_commands=commands)
            check_empty(world)
            # Everything below is measurement by the harness, outside guarded control().
            alive = [d for d in world.drones if not d.removed]
            world.explored_map.update_drones(alive)
            for drone in alive:
                position = drone.true_position()
                if not np.isfinite(position).all():
                    raise ValueError(f"Invalid position from native physics at step {frame+1}")
                positions[frame, int(drone.identifier)] = position
            result["steps"] = frame + 1
            result["wall_seconds"] = time.perf_counter() - started
            if (frame + 1) % 600 == 0:
                print(f"step={frame+1} elapsed={result['wall_seconds']:.2f}s", flush=True)

        result["stop_reason"] = "step_limit" if result["steps"] == step_limit else "wall_limit"
        result.update(mapping_stats=[d.mapping_stats() for d in world.drones],
                      final_health=[int(d.drone_health) for d in world.drones],
                      removed_slots=[int(d.identifier) for d in world.drones if d.removed])
        pred = np.bitwise_or.reduce([d.estimated_walls() for d in world.drones])
        truth = walls_only_grid(pg)
        if truth is None or truth.shape != pred.shape:
            raise ValueError("Native wall geometry and estimated map shapes differ")
        coverage = 100 * world.explored_map.score()
        visible = world.explored_map.get_pretty_map_explo_zones() == 128
        score = PlaceScoreManager().compute_map_score(
            truth, pred, elapsed_walltime=result["wall_seconds"], max_walltime_limit=world.max_walltime_limit)
        if score.error:
            raise ValueError(score.error)
        result.update(coverage_percent=coverage, map_score=score.score,
                      wall_hits=score.breakdown["wall_hits"], charged_pixels=score.breakdown["charged_pixels"],
                      map_breakdown=score.breakdown)
        with (run_dir / "trial.npz").open("xb") as stream:
            np.savez_compressed(stream, truth=truth, pred=pred, visible=visible,
                                positions=positions[:result["steps"]])
        result["status"] = "ok"
    finally:
        world.playground.window.close()


def main():
    if len(sys.argv) != 2 or sys.stdin.readline().strip() != "GO":
        raise SystemExit("Expected a run directory and resource-limiter GO token")
    run_dir = Path(sys.argv[1]).resolve(strict=True)
    if any((run_dir / name).exists() for name in ("result.json", "trial.npz")):
        raise FileExistsError(f"Existing run outputs under {run_dir}")
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8-sig"))
    result = {key: request[key] for key in ("map", "round", "seed")}
    result.update(status="error", steps=0, wall_seconds=0, stop_reason="error")
    try:
        fly(request, run_dir, result)
    except Exception as exc:
        result.update(status="error", error=repr(exc), stop_reason="error")
        traceback.print_exc()
    with (run_dir / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
