"""One worker at a time, with an OS-enforced memory limit and one CPU affinity."""
import os
import subprocess
import time


def linux_limit(memory_bytes):
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(memory_bytes,memory_bytes))
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


def run_limited(command,environment,log,timeout,memory_bytes=4_000_000_000):
    started = time.monotonic()
    process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
        env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    job,k,limits = None,None,None
    peak = None
    status = 'process_error'
    try:
        if os.name=='nt':
            import ctypes as c
            from ctypes import wintypes as w
            from ...windows_pressure import Limits
            k = c.WinDLL('kernel32',use_last_error=True)
            k.CreateJobObjectW.argtypes,k.CreateJobObjectW.restype = [c.c_void_p,w.LPCWSTR],w.HANDLE
            k.SetInformationJobObject.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
            k.AssignProcessToJobObject.argtypes = [w.HANDLE,w.HANDLE]
            k.QueryInformationJobObject.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD,c.c_void_p]
            k.GetProcessAffinityMask.argtypes = [w.HANDLE,c.c_void_p,c.c_void_p]
            k.SetProcessAffinityMask.argtypes = [w.HANDLE,c.c_size_t]
            k.CloseHandle.argtypes = [w.HANDLE]
            def checked(value):
                if not value:
                    raise c.WinError(c.get_last_error())
            job = k.CreateJobObjectW(None,None)
            checked(job)
            allowed,system = c.c_size_t(),c.c_size_t()
            checked(k.GetProcessAffinityMask(w.HANDLE(process._handle),c.byref(allowed),c.byref(system)))
            limits = Limits()
            limits.basic.flags = 0x200|0x2000|0x10  # job memory, kill descendants on close, affinity
            limits.basic.affinity = allowed.value & -allowed.value
            limits.job_limit = memory_bytes
            checked(k.SetInformationJobObject(job,9,c.byref(limits),c.sizeof(limits)))
            checked(k.AssignProcessToJobObject(job,w.HANDLE(process._handle)))
            # Also pin the process explicitly. Nested Windows jobs may retain
            # a broad process mask even after the job affinity limit is set.
            checked(k.SetProcessAffinityMask(w.HANDLE(process._handle),limits.basic.affinity))
            effective,available=c.c_size_t(),c.c_size_t()
            checked(k.GetProcessAffinityMask(w.HANDLE(process._handle),c.byref(effective),c.byref(available)))
            if effective.value.bit_count()!=1:
                raise RuntimeError('One-CPU process affinity could not be verified')
        process.stdin.write(b'GO\n')
        process.stdin.flush()
        process.stdin.close()
        try:
            process.wait(timeout=timeout)
            status = 'exited' if process.returncode==0 else 'process_error'
        except subprocess.TimeoutExpired:
            status = 'process_timeout'
            process.kill()
            process.wait()
        if job:
            if k.QueryInformationJobObject(job,9,c.byref(limits),c.sizeof(limits),None):
                peak = int(limits.peak_job)
        return dict(exit_code=process.returncode,status=status,process_seconds=time.monotonic()-started,
                    peak_job_commit_bytes=peak,memory_limit_bytes=memory_bytes,
                    memory_limit_kind='Windows job commit' if os.name=='nt' else 'POSIX address space',cpu_count=1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if job:
            k.CloseHandle(job)
