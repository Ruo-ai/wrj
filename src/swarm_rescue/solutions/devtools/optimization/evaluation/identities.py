"""Offline experiment registry. Planned candidates are deliberately not runnable."""
from importlib import import_module
from types import MappingProxyType

RED_CONTROLLERS = MappingProxyType({
    'advanced2': ('my_drone_red_advanced2', 'MyDroneRedAdvanced2'),
    'advanced5_narrow': ('my_drone_red_advanced5_narrow', 'MyDroneRedAdvanced5Narrow'),
    'advanced2_fs1_area': ('my_drone_red_fs1', 'MyDroneRedFS1Area'),
    'advanced2_fs1_gain': ('my_drone_red_fs1', 'MyDroneRedFS1Gain'),
    'advanced2_semifinal': ('my_drone_red_semifinal', 'MyDroneRedSemifinal'),
    'advanced2_mapping': ('my_drone_red_mapping', 'MyDroneRedMapping'),
})
BLUE_CONTROLLERS = MappingProxyType({
    'baseline': ('my_drone_eval_blue', 'MyDroneBlueAdapted'),
    'adapted': ('my_drone_eval_blue', 'MyDroneBlueAdapted'),
    'enhanced': ('my_drone_eval_blue', 'MyDroneBlue'),
    'sweep': ('my_drone_blue_sweep', 'MyDroneBlueSweep'),
    'semifinal': ('my_drone_blue_semifinal', 'MyDroneBlueSemifinal'),
    'semifinal_pose': ('my_drone_blue_semifinal', 'MyDroneBlueSemifinalPose'),
    'transport': ('my_drone_blue_transport', 'MyDroneBlueTransport'),
    'transport_compute': ('my_drone_blue_transport', 'MyDroneBlueTransportCompute'),
    'boundary': ('my_drone_blue_recovery', 'MyDroneBlueBoundary'),
    'contact': ('my_drone_blue_contact', 'MyDroneBlueContact'),
    'semantic_search': ('my_drone_blue_search', 'MyDroneBlueSemanticSearch'),
})


def blue_identity(selector):
    try:
        module, name = BLUE_CONTROLLERS[selector]
    except KeyError as exc:
        raise ValueError(f'Unknown explicit blue controller: {selector!r}') from exc
    return dict(blue_variant=selector, blue_controller_class=name,
                blue_controller_module='swarm_rescue.solutions.' + module)


def load_blue(selector):
    identity = blue_identity(selector)
    cls = getattr(import_module(identity['blue_controller_module']), identity['blue_controller_class'])
    if selector == 'baseline':
        module = import_module(identity['blue_controller_module'])
        if module.REFERENCE_BLUE_CLASS is not cls:
            raise ValueError('Reference blue default changed; review registry before evaluation')
    return cls


def catalog():
    return dict(
        A0=dict(selector='advanced2', state='development_baseline'),
        N0=dict(selector='advanced5_narrow', state='independent_control'),
        B0=dict(selector='adapted', state='development_reference_not_strength_accepted'),
        A1=dict(selector='advanced2_fs1_gain', state='candidate_not_adopted', parent='A0'),
        A1a=dict(selector='advanced2_fs1_area', state='area_ablation_not_adopted', parent='A0'),
        N1=dict(selector=None, state='not_implemented', parent='N0'),
        B1=dict(selector=None, state='not_implemented', parent='B0'),
        A2=dict(selector='advanced2_semifinal', state='candidate_not_adopted', parent='A0'),
        A3=dict(selector='advanced2_mapping', state='mapping_candidate_not_adopted', parent='A0'),
        B2=dict(selector='semifinal', state='candidate_not_adopted', parent='B0'),
        B3=dict(selector='transport_compute', state='compute_candidate_not_strength_accepted', parent='B0'),
        B3_terminal=dict(selector='transport', state='rejected_terminal_experiment', parent='B0'),
        B4_search=dict(selector='semantic_search', state='candidate_health_only_no_disposal_gain', parent='B3'),
        B4_boundary=dict(selector='boundary', state='rejected_regression', parent='B3'),
        B4_contact=dict(selector='contact', state='rejected_health_regression', parent='B3'),
    )
