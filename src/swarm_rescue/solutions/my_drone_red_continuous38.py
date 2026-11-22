"""Entry37R2 flight policy with shared continuously constrained legal pose."""
from swarm_rescue.solutions.my_drone_red_entry37r2 import MyDroneRedEntry37R2
from swarm_rescue.solutions.my_drone_red_fusion27 import FinalSnapshots27
from swarm_rescue.solutions.red_continuous38 import ContinuousMapper38, SharedPose38


class MyDroneRedContinuous38(MyDroneRedEntry37R2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submission_mapper = ContinuousMapper38(self.size_area)
        self._pose_filter = SharedPose38(self._submission_mapper)
        self._radio_maps = FinalSnapshots27(self._map.shape, self._n,
            self._submission_mapper, lambda: int(self.elapsed_timestep))

    def _update_observation(self):
        super()._update_observation()
        # SegmentMotion retains its measured speed and executor policy, but
        # must not fight a second independent heading estimate.
        if self._position is not None:
            self._segment_motion.pose.position = self._position.copy()
            self._segment_motion.pose.heading = self._heading
