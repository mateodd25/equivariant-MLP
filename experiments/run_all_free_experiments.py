#!/usr/bin/env python3

from free_diagonal_extraction import run_diagonal_extraction_experiment
from free_O_invariant import run_O_invariant_experiment
from free_singular_vector import run_singular_vector_experiment
from free_trace import run_trace_experiment

if __name__ == "__main__":
    try:
        run_diagonal_extraction_experiment()
    except:
        pass

    try:
        run_trace_experiment()
    except:
        pass

    try:
        run_singular_vector_experiment()
    except:
        pass

    try:
        run_O_invariant_experiment()
    except:
        pass
