import time
from typing import Callable

def _test_time(in_func: Callable):
    def out_func(*args, **kwargs):
        t1 = time.time()
        out = in_func(*args, **kwargs)
        t2 = time.time()
        print("call function %s: %dms" % (in_func.__name__, int((t2 - t1) * 1000)))
        return out

    return out_func
