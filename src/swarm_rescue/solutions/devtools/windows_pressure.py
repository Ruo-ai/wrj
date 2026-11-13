"""Run a local evaluation under a Windows job-wide commit and affinity limit."""
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import runpy
import sys


class Basic(c.Structure):
    _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64), ('flags', w.DWORD),
                ('min_ws', c.c_size_t), ('max_ws', c.c_size_t), ('active', w.DWORD),
                ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]


class Limits(c.Structure):
    _fields_ = [('basic', Basic), ('io', c.c_uint64 * 6), ('process_limit', c.c_size_t),
                ('job_limit', c.c_size_t), ('peak_process', c.c_size_t), ('peak_job', c.c_size_t)]


def main():
    filename, module, *args = sys.argv[1:]
    dest = Path(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', encoding='utf-8') as stream:
        k = c.WinDLL('kernel32', use_last_error=True)
        k.CreateJobObjectW.restype = w.HANDLE
        job = w.HANDLE(k.CreateJobObjectW(None, None))
        process = w.HANDLE(-1)
        allowed, system = c.c_size_t(), c.c_size_t()

        def check(value):
            if not value:
                raise c.WinError(c.get_last_error())

        check(job.value)
        check(k.GetProcessAffinityMask(process, c.byref(allowed), c.byref(system)))
        limits = Limits()
        limits.basic.affinity = allowed.value & -allowed.value
        limits.basic.flags = 0x200 | 0x10  # JOB_MEMORY + AFFINITY
        limits.job_limit = 4_000_000_000
        check(k.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits)))
        check(k.AssignProcessToJobObject(job, process))
        try:
            sys.argv = [module, *args]
            runpy.run_module(module, run_name='__main__')
        finally:
            check(k.QueryInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits), None))
            json.dump(dict(job_commit_limit=limits.job_limit, affinity_mask=limits.basic.affinity,
                           peak_job_commit=limits.peak_job), stream)
            k.CloseHandle(job)


if __name__ == '__main__':
    main()
