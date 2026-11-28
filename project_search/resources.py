"""Process-local CPU limits. Children inherit affinity; no machine settings changed."""
import ctypes
import os

DEFAULT_CORES = 4


def core_masks():
    if os.name != 'nt':
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else range(os.cpu_count() or 1)
        return [1 << i for i in available]
    from ctypes import wintypes as W
    class Info(ctypes.Structure):
        _fields_ = [('mask', ctypes.c_size_t), ('relationship', W.DWORD), ('data', ctypes.c_byte * 16)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    size = W.DWORD()
    kernel.GetLogicalProcessorInformation(None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformation(buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [W.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    cores = []
    for offset in range(0, size.value, ctypes.sizeof(Info)):
        entry = Info.from_buffer_copy(buffer, offset)
        usable = entry.mask & process.value
        if entry.relationship == 0 and usable:
            cores.append(usable)
    return cores


def configure(cores=None):
    count = int(cores or os.environ.get('LOCALDOCS_CPU_CORES', DEFAULT_CORES))
    if not 1 <= count <= 64:
        raise ValueError('LOCALDOCS_CPU_CORES must be between 1 and 64')
    masks = core_masks()
    selected = masks[:count]
    # One logical CPU per physical core; SMT siblings cannot add hidden CPU load.
    mask = sum(m & -m for m in selected)
    if not mask:
        raise RuntimeError('No CPU cores available')
    if os.name == 'nt':
        from ctypes import wintypes as W
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = W.HANDLE
        kernel.SetProcessAffinityMask.argtypes = [W.HANDLE, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {i for i in range(mask.bit_length()) if mask & (1 << i)})
    threads = len(selected)
    os.environ['LOCALDOCS_CPU_CORES'] = str(threads)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _raw_ew = os.environ.get('LOCALDOCS_EMBED_WORKERS', '').strip()
    try:
        embed_workers = max(1, int(_raw_ew)) if _raw_ew else 1
    except ValueError:
        embed_workers = 1
    os.environ['LOCALDOCS_EMBED_WORKERS'] = str(embed_workers)
    os.environ['LOCALDOCS_OCR_WORKERS'] = '1'
    from pathlib import Path
    if (Path(__file__).resolve().parents[1] / 'models').is_dir():
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return {'physical_cores': threads, 'logical_cpus': [i for i in range(mask.bit_length()) if mask & (1 << i)],
            'affinity_mask': mask, 'threads': threads, 'embed_workers': embed_workers, 'ocr_workers': 1}


def thread_count():
    return min(int(os.environ.get('LOCALDOCS_CPU_CORES', DEFAULT_CORES)), len(core_masks()))
