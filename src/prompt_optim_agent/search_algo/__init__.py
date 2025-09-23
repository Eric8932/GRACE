from . import *
from .grace_search import GraceSearch


SEARCH_ALGOS = {'grace': GraceSearch,}

def get_search_algo(algo_name):
    assert algo_name in SEARCH_ALGOS.keys(), f"Search algo {algo_name} is not supported."
    return SEARCH_ALGOS[algo_name]