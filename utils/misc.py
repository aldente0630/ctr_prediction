import pickle
import time
from contextlib import contextmanager


def dump_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
        
        
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
        

@contextmanager
def get_elapsed_time(format_string='Elapsed time: %d sec', verbose=True):
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time
    if verbose:
        print(format_string % elapsed_time)
