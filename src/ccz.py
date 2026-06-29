import os
import pickle
import time
import numpy as np

from itertools import combinations
from ortools.sat.python import cp_model

from utils import sol_to_mat

class TrioOrthogonalSATCCZ:
    def __init__(self,n,min_dist=3,max_gate=0):
        self.n = n+2
        self.min_dist = min_dist
        self.max_gate = max_gate

        # ortools CP model
        self.model = cp_model.CpModel()

        self.Sn = []
        self.v = []
        self.hweights = []
        self.zw = []

        # builds Pauli product rotation as binary vector
        self._build_columns()
        # create SAT bool vars
        self._create_variables()
        # trio constraints 
        self._triorthogonal_constraints()
        # dist constraints 
        self._distance_constraints()

    def _build_columns(self):
        n = self.n
        Sn = np.array([[(k//(2**i)) % 2 for i in range(n)] for k in range(1<<n)])[1:]
        if self.min_dist > 1:
            # remove the all the gates column  that would lead to a logical error
            err = np.array([[j,k]+[i]*(n-2) for i in range(2) for j in range(2) for k in range(2)])
            mask = [True if ~np.any(np.all(err == S,axis=1)) else False for S in Sn]
            Sn = Sn[mask]
        # weight sorted
        perm = sorted(range(len(Sn)), key=lambda k: sum(Sn[k]))
        Sn = Sn[perm]
        self.Sn = Sn

    def _create_variables(self):
        self.v = [self.model.NewBoolVar(f"v_{k}") for k in range(len(self.Sn))]

    def _triorthogonal_constraints(self):
        Sn = self.Sn
        v = self.v
        n = self.n
        model = self.model
        lSn = len(Sn)

        n_last = n
        ## Lines (qubit) add up to 0 mod 2
        for i in range(n_last):
            line = [v[k] for k in range(lSn) if Sn[k][i] == 1]
            model.AddBoolXOr(line+[1])
        ## Pairs (qubit) add up to 0 mod 2
        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                pair = [v[k] for k in range(lSn)
                    if Sn[k][i1] == 1 and Sn[k][i2] == 1]
                model.AddBoolXOr(pair+[1])
        ## Triplets (qubit) : 1 if i,j=0,1 else 0
        for i1 in range(n - 2):
            for i2 in range(i1 + 1, n - 1):
                for i3 in range(i2 + 1, n_last):
                    triplet = [v[k] for k in range(lSn)
                        if Sn[k][i1] == 1 and Sn[k][i2] == 1 and Sn[k][i3] == 1]
                    if i1 == 0 and i2 == 1:
                        model.AddBoolXOr(triplet)
                    else:
                        model.AddBoolXOr(triplet+[1])

    def _distance_constraints(self):
        Sn = self.Sn
        v = self.v
        N = self.n
        model = self.model
        lSn = len(Sn)

        clauses = []
        index_map = {tuple(Sn[k].tolist()): k for k in range(lSn)}
        err = np.array([[j,k]+[i]*(N-2) for i in range(2) for j in range(2) for k in range(2)])
        # we already excluded dist 1 from Sn at creation
        # we now add constraints for dist >= 2
        # for each distance d
        for d in range(2, self.min_dist):
            # for every combination of d-1 gates
            for subset in combinations(range(lSn), d-1):
                # check with what other gate the combination+that gate sum up to id
                # i.e. what combination of gate will create a logical error
                comb = np.mod(sum(Sn[i] for i in subset), 2)
                for e in err:
                    comp = np.mod(comb+e,2)
                    key = tuple(comp.tolist())
                    if key in index_map:
                        k_comp = index_map[key]
                        if k_comp > subset[-1]:
                            # we exclude that combination using OR
                            clauses.append(tuple(list(subset) + [k_comp]))

        for clause in clauses:
            # exclude the combinations of gates producing logical error with dist d
            model.AddBoolOr([v[k].Not() for k in clause])

def is_CCZ_Victor(G):
    """
    Whether a matrix is triorthogonal for the reptition code.
    """
    n,_ = G.shape
    for i in range(n):
        if np.sum(G[i])%2 == 1:
            print(i)
            return False
        for j in range(i + 1, n):
            if np.sum(G[i]*G[j])%2 == 1:
                print(i,j)
                return False
            for k in range(j + 1, n):
                if i == 0 and j == 1:
                    if np.sum(G[i]*G[j]*G[k])% 2 == 0:
                        print(i,j,k)
                        return False
                else:
                    if np.sum(G[i]*G[j]*G[k])% 2 == 1:
                        print(i,j,k)
                        return False
    return True
                

def bruteforce_distance(G,d):
    # distance k = check that no combinatinos of k-1 gates lead to err
    N = G.shape[0]
    err = np.array([[j,k]+[i]*(N-2) for i in range(2) for j in range(2) for k in range(2)])
    for k in range(1,d): 
        for c in combinations(range(G.shape[1]),k):
            comb = np.mod(sum([G[:,i] for i in c]),2)
            if np.any(np.all(err == comb,axis=1)):
                print(c)
                return False
    return True

cczsat = TrioOrthogonalSATCCZ(2,min_dist=2)
solver = cp_model.CpSolver()
st = solver.solve(cczsat.model)
print(st)
if st == 4:
    vs = [solver.Value(e) for e in cczsat.v]
    M = sol_to_mat(vs,cczsat.Sn)
