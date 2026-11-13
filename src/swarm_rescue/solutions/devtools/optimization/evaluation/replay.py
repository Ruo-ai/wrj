"""Verify a G3 attempt before opening its recorded replay with the generated map."""
import argparse
from pathlib import Path


def validate_replay_sources(sources):
    """Recordings use _ReplayDrone, never the live strategy; pin the engine/maps.

    Controller development must not invalidate historical physical playback.
    Engine, maps, rendering assets and frozen map JSON/PNG must still match.
    This is deliberately separate from strict evaluation-resume validation.
    """
    from .runner import validate_sources
    pinned = {name: sha for name, sha in sources.items()
              if not name.replace('\\', '/').startswith('src/swarm_rescue/solutions/')
              or '/generated_maps/' in name.replace('\\', '/')}
    validate_sources(pinned)


def main():
    from .runner import read,receipt,validate_sources,file_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt',type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    batch = next((p for p in attempt.parents if (p/'plan.json').is_file()),None)
    if batch is None:
        raise ValueError('No parent G3 plan found')
    plan = read(batch/'plan.json')
    validate_replay_sources(plan['source_sha256'])
    if read(attempt/'integrity.json')!=receipt(attempt,file_hash(batch/'plan.json')):
        raise ValueError('Replay artifact integrity mismatch')
    request = read(attempt/'request.json')
    recordings = list((attempt/'artifacts/recordings').glob('*.npz'))
    if len(recordings)!=1:
        raise ValueError('Expected exactly one state recording')
    from ..map_generator.native_map import register
    from ...evaluate_red import configure_ray_backend
    from ...replay_window import configure_replay_window
    from swarm_rescue.simulation.gui_map.replay_view import run_replay
    configure_ray_backend('cpu')
    configure_replay_window()
    from .scenarios import resolve_map,verify_recording
    from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
    verify_recording(recordings[0],request)
    cls = resolve_map(request)
    run_replay(str(recordings[0]),cls,{'zones_config':tuple(ZoneType[n] for n in request['zones'])})


if __name__=='__main__':
    main()
