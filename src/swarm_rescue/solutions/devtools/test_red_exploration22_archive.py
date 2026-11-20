"""Necessary archive identity and no-workspace-fallback regression checks."""
import hashlib
import importlib
from pathlib import Path
import sys
import unittest

from swarm_rescue.solutions.devtools.red_exploration22_archive import (
    _ArchiveContext, archive_runtime_identity, audit_archive, load_archived_controller,
)


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "team742_preliminary.zip"


class ArchiveIsolationTests(unittest.TestCase):
    def test_real_archive_factory_and_constructor_share_engine_only(self):
        from swarm_rescue.solutions.my_drone_red_repair17 import MyDroneRedRepair17
        from swarm_rescue.simulation.reporting.team_mode import TeamMode
        from swarm_rescue.simulation.utils.misc_data import MiscData
        original_entry = importlib.import_module("swarm_rescue.solutions.my_drone_eval")
        before = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        cls = load_archived_controller(ARCHIVE)
        instance = cls(identifier=0, misc_data=MiscData(
            size_area=(1700, 1100), number_drones=5, max_timestep_limit=5400,
            max_walltime_limit=180, team_mode=TeamMode.PLACE))
        self.assertEqual(cls.__name__, "MyDroneRedRepair17")
        self.assertIsNot(cls, MyDroneRedRepair17)
        self.assertEqual(type(instance._submission_mapper).__name__, "RepairMapper17")
        self.assertIs(sys.modules["swarm_rescue.solutions.my_drone_eval"], original_entry)
        audit = audit_archive(ARCHIVE)
        self.assertEqual(audit["team_info"]["team_number"], 742)
        self.assertEqual(audit["requirements_text"], "")
        self.assertEqual(audit["zip_sha256"], before)
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), before)
        for member, digest in audit["red_dependency_sha256"].items():
            self.assertEqual(archive_runtime_identity(cls)["loaded_member_sha256"][member], digest)

    def test_missing_member_cannot_resolve_to_workspace(self):
        # Break an in-memory view only. No archive is created or modified.
        context = _ArchiveContext(ARCHIVE)
        context.namespace += "_missing_dependency_test"
        context.files.pop("my_drone_red_repair17.py")
        sys.meta_path.insert(0, context)
        try:
            with self.assertRaisesRegex(ModuleNotFoundError, "workspace fallback forbidden"):
                importlib.import_module(context.namespace + ".my_drone_eval")
        finally:
            sys.meta_path.remove(context)
            for name in list(sys.modules):
                if name == context.namespace or name.startswith(context.namespace + "."):
                    del sys.modules[name]

    def test_older_rl11_archive_has_distinct_real_factory(self):
        earlier = ROOT / "output/submissions/rl11_red_20261115_01/team742_preliminary.zip"
        earlier_audit = audit_archive(earlier)
        current = audit_archive(ARCHIVE)
        self.assertEqual(earlier_audit["controller_class"], "MyDroneRedLines11")
        self.assertNotEqual(earlier_audit["zip_sha256"], current["zip_sha256"])
        self.assertNotEqual(earlier_audit["red_dependency_sha256"], current["red_dependency_sha256"])


if __name__ == "__main__":
    unittest.main()
