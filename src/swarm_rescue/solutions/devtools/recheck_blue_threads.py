"""Repeat a single frozen blue request under a single-thread CPU configuration."""
import argparse
import os
from pathlib import Path

from swarm_rescue.solutions.devtools import evaluate_verified_revision as ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    request = ev.read(args.source / "request.json")
    original = ev.read(args.source / "result.json")
    for field, key in (("layout_sha256", "layout"), ("red_result_sha256", "red_result")):
        if original[field] != ev.digest(request[key]):
            raise ValueError(f"Changed input: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"), "1")
    os.environ.update(config)
    ev.atomic_json(args.output / "diagnostic.json", dict(environment=config,
                   original_request=ev.digest(args.source / "request.json")))
    request["run_dir"] = str(args.output.resolve())
    result = ev.execute(request, "blue")
    ev.valid_result(result, request, "blue")


if __name__ == "__main__":
    main()
