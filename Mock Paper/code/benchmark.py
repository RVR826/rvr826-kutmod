import time
import numpy as np
from algos import lu

SIZES = [10, 50, 250, 1000]
REPEAT = 100

def run():
    all_runs = []
    print("Running simulation (PERFORMANCE)...\n")
    # matrix sizes
    for n in SIZES:
        result_times = []
        
        print(f"n = {n}")
        print("progress: ")
        
        # run this many time per matrix
        for i in range(REPEAT):
            A = np.random.rand(n, n)
            start = time.perf_counter()
            P, L, U = lu(A)
            end = time.perf_counter()
            
            print("#" * i, end='\r')
            result_times.append(end - start)
            
        all_runs.append(result_times)
        print()

    return np.array(all_runs)

def errors():
    all_runs = []
    print("Running simulation (RESIDUAL ERROR)...\n")
    # matrix sizes
    for n in SIZES:
        result_errors = []
        
        print(f"n = {n}")
        print("progress: ")
        
        # run this many time per matrix
        for i in range(REPEAT):
            A = np.random.rand(n, n)
            P, L, U = lu(A)
            
            print("#" * i, end='\r')
            result_errors.append(np.linalg.norm(P @ A - L @ U, ord='fro'))
            
        all_runs.append(result_errors)
        print()

    return np.array(all_runs)
