"""E9 entrance tasks / P9 equivalent cache follow-up, serial and prepare-only default."""
import sys
from swarm_rescue.solutions.devtools import red_r8_batch as wrapper
from swarm_rescue.solutions.devtools import red_rn7_batch as driver


def main():
    # wrapper supplies passive diagnostics. Its original main fixes its module
    # name, so use a process-local shim around the retained RN7 driver's main.
    old=driver.main;old_extra=wrapper.EXTRA
    def registered():
        driver.MODULE='swarm_rescue.solutions.devtools.red_r9_batch'
        return old()
    wrapper.EXTRA={
        'e9_n':('my_drone_red_entrance9','MyDroneRedEntrance9'),
        'p9_m':('red_pixel_faces_incremental','MyDroneRedPixel9')}
    try:
        driver.main=registered
        if '--worker' not in sys.argv and '--controllers' not in sys.argv:sys.argv+=['--controllers','e9_n','p9_m']
        wrapper.main()
    finally:driver.main=old;wrapper.EXTRA=old_extra


if __name__=='__main__':main()
