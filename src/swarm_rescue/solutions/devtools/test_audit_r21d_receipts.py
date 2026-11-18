"""Regression tests for the offline R21d receipt reconciliation."""

import unittest

if __package__:
    from .audit_r21d_receipts import reconcile_tasks
else:
    from audit_r21d_receipts import reconcile_tasks


def example(reason="completed", samples=0, new_cells=0, crosses=None):
    events = [
        {"tick": 1, "reason": "entered"},
        {"tick": 2, "reason": "local_replan"},
        {"tick": 3, "reason": reason},
    ]
    crosses = list(crosses or ())
    task = {
        "owner": 0, "start": 1, "end": 3, "reason": reason,
        "samples": samples, "new_cells": new_cells,
        "cross": sum(c["owner"] == 0 and c["task"] == 1 for c in crosses),
    }
    return {"drones": {"0": {"events": events}}}, {"tasks": [task], "crosses": crosses}


def crossing(tick, task, owner=0, gate="G01"):
    return {"owner": owner, "tick": tick, "gate": gate, "task": task, "side": -1}


class ReconcileTasksTest(unittest.TestCase):
    def test_completed_with_no_crossing_and_no_sample(self):
        task, = reconcile_tasks(*example())
        self.assertEqual((task["owner"], task["start"], task["end"]), (0, 1, 3))
        self.assertEqual(task["reason"], "completed")
        self.assertEqual(task["replan_ticks"], [2])
        self.assertEqual(task["crossings"], 0)
        self.assertIsNone(task["sampled_new_cells"])

    def test_sampled_zero_and_positive_remain_measured_values(self):
        for new_cells in (0, 7):
            with self.subTest(new_cells=new_cells):
                task, = reconcile_tasks(*example(samples=1, new_cells=new_cells))
                self.assertEqual(task["sampled_new_cells"], new_cells)

    def test_counts_each_observed_crossing_only_inside_its_own_task(self):
        crosses = [crossing(0, None), crossing(1, 1), crossing(3, 1, gate="G02"),
                   crossing(4, None), crossing(2, None, owner=1)]
        narrow, review = example(crosses=crosses)
        narrow["drones"]["1"] = {"events": []}
        task, = reconcile_tasks(narrow, review)
        self.assertEqual(task["crossings"], 2)

    def test_abort_reason_survives_reconciliation(self):
        for reason in ("fresh_scan_obstructed", "peer_wait_budget", "mission_preempted",
                       "no_progress", "invalid_odometry", "task_timeout", "inertial_risk"):
            with self.subTest(reason=reason):
                task, = reconcile_tasks(*example(reason=reason))
                self.assertEqual(task["reason"], reason)
                self.assertEqual(task["crossings"], 0)

    def test_duplicate_review_task_is_rejected(self):
        narrow, review = example()
        review["tasks"].append(review["tasks"][0].copy())
        with self.assertRaises(ValueError):
            reconcile_tasks(narrow, review)

    def test_missing_raw_or_review_task_is_rejected(self):
        for missing in ("raw", "review"):
            with self.subTest(missing=missing):
                narrow, review = example()
                if missing == "raw":
                    narrow["drones"]["0"]["events"].clear()
                else:
                    review["tasks"].clear()
                with self.assertRaises(ValueError):
                    reconcile_tasks(narrow, review)

    def test_terminal_step_or_reason_mismatch_is_rejected(self):
        for field, value in (("end", 4), ("reason", "mission_preempted")):
            with self.subTest(field=field):
                narrow, review = example()
                review["tasks"][0][field] = value
                with self.assertRaises(ValueError):
                    reconcile_tasks(narrow, review)

    def test_unclosed_task_is_rejected_even_when_review_omits_it(self):
        narrow, review = example()
        narrow["drones"]["0"]["events"].pop()
        review["tasks"].clear()
        with self.assertRaises(ValueError):
            reconcile_tasks(narrow, review)

    def test_wrong_crossing_attribution_is_rejected(self):
        for event in (crossing(2, None), crossing(2, 7), crossing(4, 1),
                      crossing(2, 1, owner=1)):
            with self.subTest(event=event):
                narrow, review = example(crosses=[event])
                narrow["drones"]["1"] = {"events": []}
                with self.assertRaises(ValueError):
                    reconcile_tasks(narrow, review)

    def test_crossing_count_without_event_is_rejected(self):
        narrow, review = example()
        review["tasks"][0]["cross"] = 1
        with self.assertRaises(ValueError):
            reconcile_tasks(narrow, review)

    def test_growth_without_a_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            reconcile_tasks(*example(samples=0, new_cells=7))


if __name__ == "__main__":
    unittest.main()
