"""Pure release scoring and conservative post-exploration permissions."""
import math
import numpy as np
from swarm_rescue.solutions.my_drone_red_fusion27 import FinalSnapshots27


class MapOnlyFinal45(FinalSnapshots27):
    def delivery(self):
        state = super().delivery()
        missing = {i:r for i,r in state['missing'].items() if r != 'inventory_remaining_at_cutoff'}
        inventory = {str(i):m[4] for i,m in self.manifests.items()}
        state.update(missing=missing,final_complete=not missing,inventory_at_cutoff=inventory,
                     map_delivery_independent_of_inventory=True)
        return state


def exploration_extension_due(progress, identifier):
    # Original second-charge scheduling boundary, independent of actual ammo.
    # Do not enable the extended frontier policy at launch just for saving ammo.
    return progress >= .34+.008*identifier


def release_decision(progress, identifier, placed, gain, depth, detour, recent_best):
    earliest = .22+.12*min(1,placed)+.008*identifier
    if progress < earliest or not math.isfinite(gain) or gain <= 0:
        return False
    score = gain*(1+.8*depth+.4*detour)
    deadline = .42 if placed == 0 else .66
    threshold = .82 if progress < deadline else .35
    return score >= max(8., threshold*recent_best) or progress >= .78


def valid_report(row, count, clock):
    return (isinstance(row,tuple) and len(row)==5 and type(row[0]) is int and 0<=row[0]<count
            and type(row[1]) is int and 0<=row[1]<=clock and type(row[2]) is bool
            and type(row[3]) is int and row[3]>=0 and type(row[4]) is int and row[4]>=0)


def post_exploration_allowed(reports, clock, count, map_state, remaining):
    if remaining < 180 or not map_state.get('final_complete',False): return False
    return all(i in reports and valid_report(reports[i],count,clock)
               and clock-reports[i][1]<=90 and reports[i][2] and reports[i][3]==0
               for i in range(count))


def movement_snapshot(drone, command):
    return (tuple(drone._route),drone._goal_step,drone._d39_mode,drone._gathering,
            drone._explore.extension_enabled,tuple((k,command.get(k)) for k in ('forward','lateral','rotation')))
