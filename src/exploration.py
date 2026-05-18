import numpy as np

from sat import SATSolution, TrioOrthogonalSAT_Sn, TrioOrthogonalSAT_YoungSym, TrioOrthogonalSAT_Cyclic
from utils import Partial_exp, coeff_dev, is_triorthogonal_rep

from itertools import combinations_with_replacement

print("# Exploring with subclass invariant under $S_n$")

S_n = {}
for N in range(5,24):
    min_dist = 3
    st = 4
    # Getting max distance
    while st == 4:
        trisat = TrioOrthogonalSAT_Sn(N,min_dist=min_dist)
        s = SATSolution(trisat,solver_time=120)
        st = s.solve()
        if st == 4:
            min_dist += 1
            M = trisat.extract_gate_matrix(s.solver)
            assert is_triorthogonal_rep(M)
            tcost = M.shape[1]
    min_dist -= 1
    print(f"{N}:  d={min_dist}")
    # Optimizing T-cost
    st = 4
    while st == 4:
        trisat = TrioOrthogonalSAT_Sn(N,min_dist=min_dist, max_gate=tcost-1)
        s = SATSolution(trisat, solver_time=120)
        st = s.solve()
        if st == 4:
            M = trisat.extract_gate_matrix(s.solver)
            assert is_triorthogonal_rep(M)
            print(f"\t {tcost} -> {M.shape[1]}")
            tcost = M.shape[1]
    np.save(f"../data/S_N__N={N}_{tcost}-to-1_d={min_dist}.npy",M)
    S_n[N] = {"d": min_dist, "n": tcost}


print("# Exploring with subclasss invariant under Young subgroup")

def find_combinations(N, k):
    return [list(c) for c in combinations_with_replacement(range(1, N-k+2), k) if sum(c) == N]

Young = {}
limit = 5 # set the limit to the maximum size of partition
for N in range(6,12):
    limit = N if limit == 0 else limit
    min_dist = S_n[N]['d']
    tcost = S_n[N]['n']+1 #we first check if we can recover the tcost
    if S_n[N-1]['d'] > min_dist:
        min_dist += 1
        tcost = 1
    print(f"N:{N}, d: {min_dist}")
    best_p = []
    possible_partitions = []
    for k in range (2,limit):
        print(f"\t{k}")
        partitions = find_combinations(N,k)
        for p in partitions:
            trisat = TrioOrthogonalSAT_YoungSym(partition=p,min_dist=min_dist,max_gate=tcost-1)
            s = SATSolution(trisat,solver_time=120,verbose=False);
            st = s.solve()
            if st == 4:
                M = trisat.extract_gate_matrix(s.solver)
                assert is_triorthogonal_rep(M)
                tcost = M.shape[1]
                print(f"{N} - {p} : {tcost}")
                best_p = p
                possible_partitions += [p]
            else:
                partitions.remove(p)
    while possible_partitions:
        for p in possible_partitions:
            trisat = TrioOrthogonalSAT_YoungSym(partition=p,min_dist=min_dist,max_gate=tcost-1)
            s = SATSolution(trisat,solver_time=120,verbose=False);
            st = s.solve()
            if st == 4:
                M = trisat.extract_gate_matrix(s.solver)
                assert is_triorthogonal_rep(M)
                tcost = M.shape[1]
                print(f"{N} - {p} : {tcost}")
                best_p = p
            else:
                possible_partitions.remove(p)
    np.save(f"../data/Young__N={N}_{tcost}-to-1_d={min_dist}_p={best_p}.npy",M)
    Young[N] = {'d': min_dist ,'partition': best_p, 'n':tcost}


print("# Exploring with subclasss invariant under cyclic permutations")

Cyclic = {}
limit = 4 # set the limit to the maximum size of partition
for N in range(6,12):
    limit = N if limit == 0 else limit
    min_dist = S_n[N]['d']
    tcost = S_n[N]['n']+1 #we first check if we can recover the tcost
    if S_n[N-1]['d'] > min_dist:
        min_dist += 1
        tcost = 1
    print(f"N:{N}, d: {min_dist}")
    best_p = []
    possible_partitions = []
    for k in range (2,limit):
        print(f"\t{k}")
        partitions = find_combinations(N,k)
        for p in partitions:
            trisat = TrioOrthogonalSAT_Cyclic(partition=p,min_dist=min_dist,max_gate=tcost-1)
            s = SATSolution(trisat,solver_time=120,verbose=False);
            st = s.solve()
            if st == 4:
                M = trisat.extract_gate_matrix(s.solver)
                assert is_triorthogonal_rep(M)
                tcost = M.shape[1]
                print(f"{N} - {p} : {tcost}")
                best_p = p
                possible_partitions += [p]
            else:
                partitions.remove(p)
    while possible_partitions:
        for p in possible_partitions:
            trisat = TrioOrthogonalSAT_Cyclic(partition=p,min_dist=min_dist,max_gate=tcost-1)
            s = SATSolution(trisat,solver_time=120,verbose=False);
            st = s.solve()
            if st == 4:
                M = trisat.extract_gate_matrix(s.solver)
                assert is_triorthogonal_rep(M)
                tcost = M.shape[1]
                print(f"{N} - {p} : {tcost}")
                best_p = p
            else:
                possible_partitions.remove(p)
    np.save(f"../data/Cyclic__N={N}_{tcost}-to-1_d={min_dist}_p={best_p}.npy",M)
    Cyclic[N] = {'d': min_dist ,'partition': best_p, 'n':tcost}


print("# Exploring synthillation protocols")


