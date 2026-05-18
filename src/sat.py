import os
import pickle
import time
import numpy as np

from math import comb
from itertools import combinations, combinations_with_replacement, product as iproduct
from collections import defaultdict
from ortools.sat.python import cp_model

from utils import sol_to_mat, coeff_dev, Partial_exp

class TrioOrthogonalSAT:
    def __init__(self,n,min_dist=3,max_gate=0,exact_gate=0,max_Z=0,
                 allowed_rot=None,symmetry="none",
                 row_constraints=False,unital=False,use_cache=False,
                 skip_triorthogonal=False):
        self.n = n
        self.min_dist = min_dist
        self.max_gate = max_gate
        self.exact_gate = exact_gate
        self.max_Z = max_Z
        self.allowed_rot = allowed_rot or []
        self.symmetry = symmetry
        self.row_constraints = row_constraints
        self.unital = unital
        self.use_cache = use_cache

        if self.unital:
            if self.exact_gate > 0:
                assert (self.exact_gate%2 ==1), "unital requires odd columns"

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
        if not skip_triorthogonal:
            self._triorthogonal_constraints()
        else:
            print("Warning: skipping triorthogonal constraints!")
        # dist constraints 
        self._distance_constraints()
        # extra 
        self._gate_constraints()

        if self.symmetry in ("lex"):
            self._add_row_lex_ordering()
        if self.symmetry in ("hamming"):
            self._add_hamming_ordering()
        if self.symmetry in ("weight"):
            self._add_row_weight_ordering()

    def _build_columns(self):
        n = self.n
        Sn = np.array([[(k//(2**i)) % 2 for i in range(n)] for k in range(1<<n)])[1:]
        if self.unital:
            # here we only keep columns so that last row is all 1s
            Sn = Sn[len(Sn) // 2 :]
        if self.min_dist > 1:
            # remove the all 1s column (would lead to a logical error)
            Sn = Sn[:-1]
        if self.max_Z > 0:
            Sn = np.array([s for s in Sn if sum(s) <= self.max_Z])
        if self.allowed_rot:
            Sn = np.array([s for s in Sn if sum(s) in self.allowed_rot])
        # weight sorted
        perm = sorted(range(len(Sn)), key=lambda k: sum(Sn[k]))
        Sn = Sn[perm]
        self.Sn = Sn
        # hamming weights of columns
        self.hweights = np.sum(Sn, axis=1).astype(int)

    def _create_variables(self):
        self.v = [self.model.NewBoolVar(f"v_{k}") for k in range(len(self.Sn))]

    def _triorthogonal_constraints(self):
        Sn = self.Sn
        v = self.v
        n = self.n
        model = self.model
        lSn = len(Sn)
        unital = self.unital

        # When unital we know we have the |last_row| = 1
        # and no need to have trio constraints as redundant with pair-wise
        n_last = n-1 if unital else n
        if unital:
            model.AddBoolXOr(v)
        ## Lines (qubit) add up to 1 mod 2
        for i in range(n_last):
            line = [v[k] for k in range(lSn) if Sn[k][i] == 1]
            model.AddBoolXOr(line)
        ## Pairs (qubit) add up to 1 mod 2
        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                pair = [v[k] for k in range(lSn)
                    if Sn[k][i1] == 1 and Sn[k][i2] == 1]
                model.AddBoolXOr(pair)
        ## Triplets (qubit) same
        for i1 in range(n - 2):
            for i2 in range(i1 + 1, n - 1):
                for i3 in range(i2 + 1, n_last):
                    triplet = [v[k] for k in range(lSn)
                        if Sn[k][i1] == 1 and Sn[k][i2] == 1 and Sn[k][i3] == 1]
                    model.AddBoolXOr(triplet)


    def _distance_constraints(self):
        Sn = self.Sn
        v = self.v
        n = self.n
        model = self.model
        lSn = len(Sn)

        os.makedirs("../cache", exist_ok=True)
        cache_file = f"../cache/dist_constraints_n{n}_d{self.min_dist}_{int(self.unital)*'unital'}.pkl"

        # either load constraints from cache files
        if os.path.exists(cache_file) and self.use_cache:
            with open(cache_file, "rb") as f:
                clauses = pickle.load(f)
        else:
            #or generate them on the go, and store them in a file
            clauses = []
            index_map = {tuple(Sn[k].tolist()): k for k in range(lSn)}
            # we already excluded dist 1 from Sn at creation
            # we now add constraints for dist >= 2
            # for each distance d
            for d in range(2, self.min_dist):
                # for every combination of d-1 gates
                for subset in combinations(range(lSn), d - 1):
                    # check with what other gate the combination+that gate sum up to id
                    # i.e. what combination of gate will create a logical error
                    comp = np.mod(sum(Sn[i] for i in subset) + 1, 2)
                    key = tuple(comp.tolist())
                    if key in index_map:
                        k_comp = index_map[key]
                        if k_comp > subset[-1]:
                            # we exclude that combination using OR
                            clauses.append(tuple(list(subset) + [k_comp]))

            with open(cache_file, "wb") as f:
                pickle.dump(clauses, f)

        for clause in clauses:
            # exclude the combinations of gates producing logical error with dist d
            model.AddBoolOr([v[k].Not() for k in clause])

    def _gate_constraints(self):
        v = self.v
        model = self.model
        n = self.n
        # max amount of gates is 2^(n-1)
        max_gate = (1<<(n-1)) if self.max_gate==0 else self.max_gate
        # 49 gate is required for d>3
        if self.min_dist > 3:
            assert max_gate > 48, "No code with distance >3 exist with less than 49 gates"
            model.Add(sum(v) > 48)
        if self.exact_gate > 0:
            model.Add(sum(v) == self.exact_gate)
        else:
            model.Add(sum(v) <= max_gate)

    def _add_row_weight_ordering(self):
        n = self.n
        Sn = self.Sn
        lSn = len(Sn)
        v = self.v
        row_w = []
        for i in range(n):
            # compute the weight sum of each row so kinda \sum_j S_i[j]v_i for all i
            row_w.append(sum(v[k] for k in range(lSn) if Sn[k][i] == 1))
        # Rows should be of increasing weights 
        for i in range(n-2):
            for j in range(i+1,n-1):
                self.model.Add(row_w[j] >= row_w[i])
        # last row should have the max weight if unital (no two unital rows)
        if self.unital:
            self.model.Add(row_w[n-1] > row_w[n-2])
        else:
            self.model.Add(row_w[n-1] >= row_w[n-2])

    def _add_row_lex_ordering(self):
        rows = []
        lSn = len(self.Sn)
        # we compute the lex order of each row
        for i in range(self.n):
            row_vec = []
            for k in range(lSn):
                if self.Sn[k][i] == 1:
                    row_vec.append(self.v[k])
                else:
                    row_vec.append(self.model.NewConstant(0))
            rows.append(row_vec)
        # qdd lex ordering constraints
        for i in range(self.n - 1):
            self._add_lex_leq(rows[i], rows[i + 1], f"rowlex_{i}")

    def _add_lex_leq(self, A, B, name):
        model = self.model
        m = len(A)
        # one new bool var per row
        eq = [model.NewBoolVar(f"{name}_eq_{i}") for i in range(m + 1)]
        model.Add(eq[0] == 1)
        for i in range(m):
            # for every column, we check wether bit in A < B
            model.AddImplication(eq[i], B[i].Not()).OnlyEnforceIf(A[i])
            model.Add(A[i] == B[i]).OnlyEnforceIf(eq[i + 1])

    def _add_hamming_ordering(self):
        # For each Hamming-weight class, enforce an ordering: 
        # if columns of same weight have indices i<j,
        # then v[j] => v[i]  (i.e., v[i] >= v[j])
        # This prevents permutations among identical-weight columns being explored repeatedly.
        # UNSAFE
        Sn = self.Sn
        lSn = len(Sn)
        model = self.model
        v = self.v
        hweights = np.sum(Sn, axis=1).astype(int)
        classes = defaultdict(list)
        for k in range(lSn):
            classes[hweights[k]].append(k)
        for _, members in classes.items():
            # sort members by their index to have deterministic ordering
            members_sorted = sorted(members)
            # enforce v[i] >= v[j] for i earlier than j
            for idx in range(len(members_sorted)-1):
                i = members_sorted[idx]
                for j in members_sorted[idx+1:]:
                    # v[i] >= v[j]  <=>  v[j] -> v[i]
                    model.AddImplication(v[j], v[i])

    def model_max_gate(self,k):
        """ Return SAT model with max_gate"""
        assert self.exact_gate == 0, "Constraint on max_gate already exist"
        model = self.model.Clone()
        model.Add(sum(self.v) == k)
        return model

    def _build_logicals_of_weight_w(self,w):
        """
        Builds all logical indicators of weight exactly w:
            z_T = AND(v[k] for k in T)
        for all subsets T with:
            XOR(Sn[k] for k in T) = all-ones vector
        """
        if len(self.zw) == 0:
            model = self.model
            Sn = self.Sn
            v = self.v
            n = Sn.shape[1]
            lSn = len(Sn)
            z_w = self.zw
            ones = np.ones(n, dtype=int)
            for T in combinations(range(lSn), w):
                # for each subset T that contains w gates
                vec = np.zeros(n, dtype=int)
                for k in T:
                    vec ^= Sn[k]
                if not np.array_equal(vec, ones):
                    # if the sum of the gates is not the one vector -> not a logical error
                    continue
                # create an indicator bool variable
                z = model.NewBoolVar(f"z_w{w}_" + "_".join(map(str, T)))
                # we have to hack some linearization of a AND
                # upperbound: z <= v_k (z=1 only if all the v[k] are 1, i.e. forces 0 if any v[k] is 0)
                for k in T:
                    model.Add(z <= v[k])
                # lowerbound: z >= sum(v_k)-(w-1) (forces z=1 as soon as all the v[k] are 1)
                model.Add(z >= sum(v[k] for k in T) - (w - 1))
                z_w.append(z)
            if not z_w:
                raise ValueError(f"No algebraic logicals exist at weight {w}")

    def exclude_solution(self,vs):
        self.model.AddBoolOr(self.v[i].Not() for i in np.nonzero(vs)[0])


class SATSolution:
    def __init__(self,trisat : TrioOrthogonalSAT, solver_time = 60,n_workers=None, verbose=False):
        self.trisat = trisat
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = solver_time
        # Use all available cores (CP-SAT is embarrassingly parallel for UNSAT)
        import os
        if n_workers is None:
            n_workers = os.cpu_count() or 8
        self.solver.parameters.num_search_workers = n_workers

        # Increase clause database size limits
        self.solver.parameters.max_memory_in_mb = 8192

        # Prefer clause-learning sub-solvers over LP relaxation ones
        # (UNSAT problems benefit more from CDCL than from LP cuts)
        self.solver.parameters.linearization_level = 0   # disable LP linearisation

        # Allow more restarts
        self.solver.parameters.clause_cleanup_period = 10000

        # Log search progress every 10 s
        self.solver.parameters.log_search_progress = verbose

    def solve(self,print_sol=False):
        st = time.time()
        status = self.solver.Solve(self.trisat.model)
        if print_sol:
            print(f"Solved in {time.time()-st}s")
            self.print_solution(status)
        return status

    def optimize_prefactor(self,w=None,print_sol=False):
        w = self.trisat.min_dist if w is None else w
        if len(self.trisat.zw) == 0:
            self.trisat._build_logicals_of_weight_w(w=w)
            self.trisat.model.Minimize(sum(self.trisat.zw))
        status = self.solve(print_sol=print_sol)
        if status == cp_model.FEASIBLE and print_sol:
            print("prefactor may not be minimal")
        return status

    def solve_range_k(self, range_k=None, print_sol=False):
        results = {'n': self.trisat.n, 'd': self.trisat.min_dist, 
                   'infeasible_k': [], 'feasible_k': [], 'unkown_k': []}
        if range_k == None:
            min_k = 1
            if self.trisat.min_dist == 3:
                min_k = 15
            elif self.trisat.min_dist >= 4:
                min_k = 49
            range_k = range(min_k,1<<(self.trisat.n-1))
        if self.trisat.unital:
            print("Unital exploration only, exploring k odd only")
        for k in range_k:
            if self.trisat.unital and k%2==0:
                continue
            model = self.trisat.model_max_gate(k)
            status = self.solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                results['feasible_k'] += [k]
            elif status == cp_model.UNKNOWN:
                results['unkown_k'] += [k]
            elif status == cp_model.INFEASIBLE:
                results['infeasible_k'] += [k]
            if print_sol:
                print(f"{k} :",end=" ")
                self.print_solution(status)
        return results

    def optimize_prefactor_range_k(self, w=None, range_k=None, print_sol=False):
        results = {'n': self.trisat.n, 'd': self.trisat.min_dist, 
                   'infeasible_k': [], 'feasible_k': [], 'unkown_k': [], 'optimal_k': []}
        if range_k == None:
            min_k = 1
            if self.trisat.min_dist == 3:
                min_k = 15
            elif self.trisat.min_dist >= 4:
                min_k = 49
            range_k = range(min_k,1<<(self.trisat.n-1))
        if self.trisat.unital:
            print("Unital exploration only, exploring k odd only")
        w = self.trisat.min_dist if w is None else w
        self.trisat._build_logicals_of_weight_w(w)
        for k in range_k:
            if self.trisat.unital and k%2==0:
                continue
            model = self.trisat.model_max_gate(k)
            status = self.solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                model.Minimize(sum(self.trisat.zw))
                status = self.solver.Solve(model)
                if status == cp_model.OPTIMAL:
                    results['optimal_k'] += [[k,sum(self.solver.Value(z) for z in self.trisat.zw)]]
                elif status == cp_model.FEASIBLE:
                    results['feasible_k'] += [[k,sum(self.solver.Value(z) for z in self.trisat.zw)]]
                else:
                    results['feasible_k'] += [[k,None]]
            elif status == cp_model.UNKNOWN:
                results['unkown_k'] += [k]
            elif status == cp_model.INFEASIBLE:
                results['infeasible_k'] += [k]
            if print_sol:
                print(f"{k} :",end=" ")
                self.print_solution(status)
        return results

    def solve_incrementally(self,batch_size=10,debug=True):
        model = self.trisat.model
        Sn = self.trisat.Sn
        v = self.trisat.v
        n = self.trisat.n
        lSn = len(Sn)
        unital = self.trisat.unital
        # When unital we know we have the |last_row| = 1
        # and no need to have trio constraints as redundant with pair-wise
        n_last = n-1 if unital else n
        if unital:
            model.AddBoolXOr(v)
        ## Lines (qubit) add up to 1 mod 2
        if debug:
            print("Adding Rows and Pairs...")
        for i in range(n_last):
            line = [v[k] for k in range(lSn) if Sn[k][i] == 1]
            model.AddBoolXOr(line)
        ## Pairs (qubit) add up to 1 mod 2
        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                pair = [v[k] for k in range(lSn)
                    if Sn[k][i1] == 1 and Sn[k][i2] == 1]
                model.AddBoolXOr(pair)
        if debug:
            print("Solving...", end="")
        st = time.time()
        status = self.solve()
        if debug:
            print(f"done in {time.time()-st}s")
        if status == cp_model.INFEASIBLE:
            print("UNSAT for row/pairs")
            return status
        elif status == cp_model.UNKNOWN:
            print("timeout, continue...")
        ## Triplets (qubit) same
        triplets = list(combinations(range(n), 3))
        count_t = 0
        if debug:
            print(f"Adding {len(triplets)} triplets in batches of {batch_size}...")
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i:i+batch_size]
            count_t += len(batch)
            print(f"{count_t}/{len(triplets)} triplets")
            for t in batch:
                i1, i2, i3 = t
                lits = [v[k] for k in range(lSn) 
                        if Sn[k][i1] == 1 and Sn[k][i2] == 1 and Sn[k][i3] == 1]
                model.AddBoolXOr(lits)
            status = self.solve()
            if status == cp_model.INFEASIBLE:
                print(f"UNSAT")
                return status
            elif status == cp_model.UNKNOWN:
                print("timeout, continue...")
        return status
 
    def print_solution(self,status):
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"sat: {self.get_poly()}")
        elif status == cp_model.INFEASIBLE:
            print("infeasible.")
        elif status == cp_model.MODEL_INVALID:
            print("model invalid.")
        elif status == cp_model.UNKNOWN:
            print("unkown, try increasing `solver_time`.")

    def get_sol(self):
        vs = [self.solver.Value(e) for e in self.trisat.v]
        return vs

    def get_M(self):
        vs = self.get_sol()
        M = sol_to_mat(vs,self.trisat.Sn)
        return M

    def get_poly(self):
        M = self.get_M()
        rm = coeff_dev(M)
        p = Partial_exp(rm,self.trisat.min_dist+2)
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Partition-symmetry reduced solver
# ─────────────────────────────────────────────────────────────────────────────

def compute_partition_orbits(n, partition):
    """
    Partition {1, …, 2^n − 2} into orbits under cyclic permutations within
    each block of *partition*.

    The symmetry group is the product of cyclic groups C_{u_0} × … × C_{u_{l-1}}
    (one cyclic shift per block), matching the group used in SolveVector_Cyclic.

    Parameters
    ----------
    n         : int   – total number of qubits (= sum(partition))
    partition : list  – block sizes, e.g. [2, 3] → qubits 0-1 / 2-4

    Returns
    -------
    orbits_list : list[list[int]]
        Each element is one orbit, stored as a list of integer gate-indices
        (the integer k whose binary expansion is the gate vector).
    Ind : list[int], length 2^n
        Ind[k] = orbit index for k ∈ {1, …, 2^n − 2};
        Ind[0] = Ind[2^n − 1] = -1  (excluded gates).
    """
    assert sum(partition) == n, "partition must sum to n"
    N = 1 << n
    block_starts = [sum(partition[:b]) for b in range(len(partition))]

    def get_orbit(k):
        """All indices reachable from k by cyclic-shifting each block."""
        s = [(k >> i) & 1 for i in range(n)]
        orb = set()
        for shifts in iproduct(*[range(sz) for sz in partition]):
            idx = 0
            for b, (start, sz) in enumerate(zip(block_starts, partition)):
                shift = shifts[b]
                block = s[start:start + sz]
                for j in range(sz):
                    if block[(j - shift) % sz]:
                        idx |= 1 << (start + j)
            orb.add(idx)
        return list(orb)

    Ind = [-1] * N
    remaining = set(range(1, N - 1))
    orbits_list = []

    # Iterate in sorted order for determinism
    while remaining:
        k = min(remaining)
        orb = [x for x in get_orbit(k) if 0 < x < N - 1]
        orbit_idx = len(orbits_list)
        for idx in orb:
            remaining.discard(idx)
            Ind[idx] = orbit_idx
        orbits_list.append(orb)

    return orbits_list, Ind


class TrioOrthogonalSAT_Cyclic(TrioOrthogonalSAT):
    """
    Partition-symmetry reduced version of TrioOrthogonalSAT.

    Instead of one boolean variable per gate, this solver uses one variable
    per *orbit* under cyclic permutations within each partition block.  It
    searches only for triorthogonal codes whose column multiset is invariant
    under those permutations, reducing the search space substantially when a
    natural partition structure is expected.

    Parameters
    ----------
    partition : list[int]
        Block sizes, e.g. [2, 3] for a 5-qubit code whose first two qubits
        form one symmetry class and the last three another.
    min_dist, max_gate, exact_gate, use_cache : same as TrioOrthogonalSAT.

    Notes
    -----
    * ``unital``, ``allowed_rot``, ``max_Z``, and symmetry-breaking orderings
      are not supported (they are features of the full solver).
    * ``optimize_prefactor`` is not yet supported; call it on the full solver
      after lifting the reduced solution.
    """

    def __init__(self, partition, min_dist=3, max_gate=0, exact_gate=0,
                 use_cache=False):
        self.partition = list(partition)
        # Parent __init__ will dispatch to our overridden methods below.
        super().__init__(
            n=sum(partition),
            min_dist=min_dist,
            max_gate=max_gate,
            exact_gate=exact_gate,
            use_cache=use_cache,
            symmetry="none",
            row_constraints=False,
            unital=False,
            skip_triorthogonal=False,
        )

    # ------------------------------------------------------------------ setup

    def _build_columns(self):
        n = self.n
        N = 1 << n
        # Full binary gate table (all 2^n rows, including 0 and all-ones)
        self.Sn_full = np.array([[(k >> i) & 1 for i in range(n)]
                                  for k in range(N)], dtype=int)
        # Orbit decomposition
        self.orbits_list, self.Ind = compute_partition_orbits(n, self.partition)
        # Orbit sizes (total gates = weighted sum over selected orbits)
        self.orbit_sizes = [len(orb) for orb in self.orbits_list]
        # For parent compatibility: store Sn as representatives (one row per orbit)
        reps = np.array([self.Sn_full[orb[0]] for orb in self.orbits_list],
                        dtype=int)
        self.Sn = reps
        self.hweights = np.sum(reps, axis=1).astype(int)

    def _create_variables(self):
        r = len(self.orbits_list)
        self.v = [self.model.NewBoolVar(f"v_{k}") for k in range(r)]

    # ------------------------------------------------ triorthogonal constraints

    def _triorthogonal_constraints(self):
        n = self.n
        model = self.model
        v = self.v
        orbits_list = self.orbits_list
        r = len(orbits_list)

        def orbit_parity(orb, bits):
            """
            Count elements of *orb* that have ALL bits in *bits* set, mod 2.

            This is the parity contribution of this orbit to the triorthogonality
            XOR for the row / pair / triplet indexed by *bits*.
            """
            return sum(
                all((elem >> b) & 1 for b in bits)
                for elem in orb
            ) % 2

        # Pre-compute block start indices for row representative selection.
        block_starts = [sum(self.partition[:b]) for b in range(len(self.partition))]

        # ── row constraints: one per block, using its first qubit as representative.
        #    All qubits within a block share the same orbit parity (cyclic shifts
        #    permute them), so one constraint per block suffices — matching main.py.
        for i in block_starts:
            lits = [v[k] for k in range(r)
                    if orbit_parity(orbits_list[k], [i]) == 1]
            if not lits:
                raise ValueError(
                    f"No orbit covers qubit {i}; problem is infeasible by construction.")
            model.AddBoolXOr(lits)

        # ── pair constraints: one per orbit of qubit-pairs under G.
        #    A weight-2 orbit representative identifies a unique pair orbit; its two
        #    set bits give the representative pair.  Iterating all C(n,2) pairs would
        #    post duplicates for pairs that are G-images of each other — main.py
        #    avoids this by collecting weight-2 orbit representatives directly.
        pair_reps = []
        for orb in orbits_list:
            bits_set = [i for i in range(n) if (orb[0] >> i) & 1]
            if len(bits_set) == 2:
                pair_reps.append(bits_set)

        for bits in pair_reps:
            lits = [v[k] for k in range(r)
                    if orbit_parity(orbits_list[k], bits) == 1]
            if lits:
                model.AddBoolXOr(lits)

        # ── triplet constraints: one per orbit of qubit-triplets under G.
        #    Same logic as pairs, using weight-3 orbit representatives.
        triplet_reps = []
        for orb in orbits_list:
            bits_set = [i for i in range(n) if (orb[0] >> i) & 1]
            if len(bits_set) == 3:
                triplet_reps.append(bits_set)

        for bits in triplet_reps:
            lits = [v[k] for k in range(r)
                    if orbit_parity(orbits_list[k], bits) == 1]
            if lits:
                model.AddBoolXOr(lits)

    # ---------------------------------------------------- distance constraints

    def _distance_constraints(self):
        """
        For each distance level d ∈ {2, …, min_dist − 1}, forbid any orbit
        selection that contains a logical error of weight d.

        Strategy (mirrors SolveVector_Cyclic):
          • Fix the *representative* of the first orbit in each clause
            (justified by the partition symmetry: any error can be conjugated
            to use that representative).
          • Iterate over **all** elements of every other orbit in the clause.
          • Use combinations_with_replacement so the same orbit can appear
            multiple times (e.g. two elements from one orbit XOR to a third).
          • Deduplicate clauses via a set before posting to the model.
        """
        n = self.n
        N = 1 << n
        orbits_list = self.orbits_list
        Ind = self.Ind
        v = self.v
        model = self.model
        r = len(orbits_list)

        os.makedirs("../cache/", exist_ok=True)
        part_str = "_".join(map(str, self.partition))
        cache_file = (f"../cache/dist_constraints_reduced_part{part_str}"
                      f"_d{self.min_dist}.pkl")

        if os.path.exists(cache_file) and self.use_cache:
            with open(cache_file, "rb") as f:
                clauses = pickle.load(f)
        else:
            clauses_set = set()

            for d in range(2, self.min_dist):
                # orbit_tuple has d-1 entries (the "choosers"); the d-th orbit
                # is determined by the XOR complement.
                # combinations_with_replacement allows the same orbit twice.
                for orbit_tuple in combinations_with_replacement(range(r), d - 1):
                    k1 = orbit_tuple[0]
                    rep_k1 = orbits_list[k1][0]   # canonical representative
                    other_orbits = orbit_tuple[1:]

                    # Subsumption filter (mirrors main.py):
                    # if k2 is the complement orbit of k1, the 2-element clause
                    # {k1, comp_k1} is already posted and subsumes any 3-element
                    # clause {k1, comp_k1, k3}.  Skip to avoid redundant constraints.
                    comp_k1 = Ind[rep_k1 ^ (N - 1)]
                    if len(other_orbits) >= 1 and other_orbits[0] == comp_k1:
                        continue

                    # Iterate over all element combinations of the non-k1 orbits
                    for elem_combo in iproduct(
                            *[orbits_list[oi] for oi in other_orbits]):
                        xor_val = rep_k1
                        for e in elem_combo:
                            xor_val ^= e
                        # The complement gate that would complete the logical error
                        comp = xor_val ^ (N - 1)
                        if comp == 0 or comp == N - 1:
                            continue            # complement is excluded
                        k_comp = Ind[comp]
                        if k_comp == -1:
                            continue
                        # Canonical ordering (mirrors main.py's `m3 >= k2`):
                        # the complement orbit must be >= the last chooser orbit.
                        # This ensures each clause triple is generated exactly once,
                        # and avoids (a,b,b) type clauses from orbit_tuple=(b,b)
                        # with k_comp=a<b being re-sorted into a spurious (a,b,b).
                        if k_comp < orbit_tuple[-1]:
                            continue
                        # Deduplicate orbit indices within the clause: repeated
                        # indices collapse (v_k AND v_k = v_k), reducing the clause
                        # to its shortest equivalent form before deduplication.
                        clause = tuple(sorted(set(list(orbit_tuple) + [k_comp])))
                        clauses_set.add(clause)

            clauses = list(clauses_set)
            with open(cache_file, "wb") as f:
                pickle.dump(clauses, f)

        for clause in clauses:
            model.AddBoolOr([v[k].Not() for k in clause])

    # --------------------------------------------------------- gate constraint

    def _gate_constraints(self):
        """
        Constrain the *total gate count* (= weighted sum of orbit sizes),
        not the number of selected orbits.
        """
        v = self.v
        model = self.model
        orbit_sizes = self.orbit_sizes

        # Total gates = Σ |orbit_k| · v[k]
        total_gates = cp_model.LinearExpr.WeightedSum(v, orbit_sizes)

        max_gate = (1 << (self.n - 1)) if self.max_gate == 0 else self.max_gate
        if self.min_dist > 3:
            assert max_gate > 48, "No code with distance > 3 exists with < 49 gates"
            model.Add(total_gates > 48)

        if self.exact_gate > 0:
            model.Add(total_gates == self.exact_gate)
        else:
            model.Add(total_gates <= max_gate)

    # ---------------------------------------------- unsupported parent methods

    def _add_row_lex_ordering(self):
        raise NotImplementedError("Lex ordering not supported for reduced solver.")

    def _add_hamming_ordering(self):
        raise NotImplementedError("Hamming ordering not supported for reduced solver.")

    def _add_row_weight_ordering(self):
        raise NotImplementedError("Weight ordering not supported for reduced solver.")

    def _build_logicals_of_weight_w(self, w):
        raise NotImplementedError(
            "Prefactor optimisation is not yet supported for the reduced solver. "
            "Extract the solution matrix with get_gate_matrix() and run "
            "optimize_prefactor on a full TrioOrthogonalSAT instance instead."
        )

    # --------------------------------------------------------- solution export

    def extract_gate_matrix(self, solver):
        """
        Expand selected orbits back into individual gate columns and return the
        (n × h) binary matrix M, where each column is one physical T-gate.

        Parameters
        ----------
        solver : cp_model.CpSolver  – already called .Solve(...)

        Returns
        -------
        M : np.ndarray, shape (n, h)
        """
        n = self.n
        gates = []
        for k, orb in enumerate(self.orbits_list):
            if solver.Value(self.v[k]):
                for elem in orb:
                    gates.append([(elem >> i) & 1 for i in range(n)])
        if not gates:
            return np.zeros((n, 0), dtype=int)
        return np.array(gates, dtype=int).T

# ─────────────────────────────────────────────────────────────────────────────
# Full S_n-symmetry reduced solver
# ─────────────────────────────────────────────────────────────────────────────

class TrioOrthogonalSAT_Sn:
    """
    Triorthogonal SAT solver reduced under the full qubit-permutation group S_n.

    Under S_n the gate space {1,…,2^n−2} splits into exactly n−1 orbits,
    one per Hamming weight w ∈ {1,…,n−1}, each of size C(n,w).  A solution
    is therefore a subset W ⊆ {1,…,n−1} of active weight classes; every gate
    column of each active weight is included.

    Variables
    ---------
    v[k]  –  BoolVar, True iff all C(n, k+1) columns of weight k+1 are included
             (k = 0,…,n−2, so weight = k+1).

    Triorthogonality (3 XOR constraints)
    -------------------------------------
    Row parity   (one rep, qubit 0):   XOR_{w: C(n-1,w-1) odd}  v[w-1]  = 1
    Pair parity  (one rep, {0,1}):     XOR_{w: C(n-2,w-2) odd}  v[w-1]  = 1
    Triplet parity (one rep, {0,1,2}): XOR_{w: C(n-3,w-3) odd}  v[w-1]  = 1

    Distance constraints
    --------------------
    A logical error of physical weight r exists iff there is a multiset
    {w_1,…,w_r} of weights (each in 1..n-1) satisfying
        sum >= n,  sum ≡ n (mod 2),  max <= sum − max.
    We forbid every such multiset for r < min_dist by posting
        OR(NOT v[w_j-1] for distinct w_j in multiset).

    Gate-count constraint
    ---------------------
    Total gates = ∑_{k: v[k]=1} C(n, k+1).  Controlled by exact_gate / max_gate.

    Parameters
    ----------
    n         : int  – number of qubits
    min_dist  : int  – minimum code distance (default 3)
    max_gate  : int  – upper bound on total gate count (0 = 2^(n-1))
    exact_gate: int  – fix total gate count exactly (0 = unconstrained)
    """

    def __init__(self, n, min_dist=3, max_gate=0, exact_gate=0):
        self.n = n
        self.min_dist = min_dist
        self.max_gate = max_gate
        self.exact_gate = exact_gate

        self.model = cp_model.CpModel()

        # One variable per weight class w = k+1, index k = 0..n-2
        self.v = [self.model.NewBoolVar(f"v_w{w}") for w in range(1, n)]

        self._triorthogonal_constraints()
        self._distance_constraints()
        self._gate_constraints()

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _comb_parity(n, k):
        """C(n,k) mod 2 via Lucas' theorem (returns 0 or 1)."""
        if k < 0 or k > n:
            return 0
        # Lucas: C(n,k) is odd iff every bit of k is <= corresponding bit of n
        return int((n & k) == k)

    @staticmethod
    def _xor_to_ones_feasible(weights, n):
        """
        Can 'len(weights)' binary vectors of the given Hamming weights XOR to 1^n?
        Necessary and sufficient conditions (weights in 1..n-1):
            1. sum(weights) >= n
            2. sum(weights) ≡ n (mod 2)
            3. max(weights) <= sum(weights) − max(weights)   [no single dominator]
        """
        s = sum(weights)
        m = max(weights)
        return s >= n and (s % 2) == (n % 2) and m <= s - m

    # ----------------------------------------- triorthogonality constraints

    def _triorthogonal_constraints(self):
        """
        Three XOR constraints, one each for the unique S_n orbit of
        qubit-singles, qubit-pairs, and qubit-triplets.

        For representative subset B ⊆ {0,…,n-1} of size |B| = t, the
        parity contribution of weight-class w is C(n-t, w-t) mod 2:
        the number of weight-w vectors supported on {0,…,n-1} that have
        all bits of B set equals C(n-t, w-t).
        """
        n = self.n
        v = self.v
        model = self.model

        for t in (1, 2, 3):           # row / pair / triplet
            lits = [
                v[w - 1]
                for w in range(1, n)
                if self._comb_parity(n - t, w - t) == 1
            ]
            if not lits:
                raise ValueError(
                    f"No weight class satisfies the t={t} triorthogonality "
                    f"constraint for n={n}; problem is infeasible."
                )
            model.AddBoolXOr(lits)

    # ------------------------------------------------- distance constraints

    def _distance_constraints(self):
        """
        For every r in {2,…,min_dist-1}, forbid every multiset of r weights
        whose XOR to 1^n is realisable.  Each forbidden multiset becomes one
        AddBoolOr clause on the negated v-variables.  Repeated weights in a
        multiset collapse (v AND v = v), so only distinct weights matter per
        clause.
        """
        from itertools import combinations_with_replacement

        n = self.n
        model = self.model
        v = self.v

        clauses = set()
        for r in range(2, self.min_dist):
            for weights in combinations_with_replacement(range(1, n), r):
                if not self._xor_to_ones_feasible(weights, n):
                    continue
                # Collapse repeated weights: AND-idempotent, only distinct matter
                clause = tuple(sorted(set(w - 1 for w in weights)))
                clauses.add(clause)

        for clause in clauses:
            model.AddBoolOr([v[k].Not() for k in clause])

    # ---------------------------------------------------- gate constraints

    def _gate_constraints(self):
        """
        Total gate count = ∑_{k: v[k]=1} C(n, k+1).
        Constrained by exact_gate (equality) or max_gate (upper bound).
        """
        n = self.n
        v = self.v
        model = self.model

        orbit_sizes = [comb(n, w) for w in range(1, n)]  # C(n,w) for w=1..n-1
        total_gates = cp_model.LinearExpr.WeightedSum(v, orbit_sizes)

        max_gate = (1 << (n - 1)) if self.max_gate == 0 else self.max_gate

        if self.min_dist > 3:
            #assert max_gate > 48, "No code with distance > 3 exists with < 49 gates"
            model.Add(total_gates > 48)

        if self.exact_gate > 0:
            model.Add(total_gates == self.exact_gate)
        else:
            model.Add(total_gates <= max_gate)

    # -------------------------------------------------- solution extraction

    def extract_gate_matrix(self, solver):
        """
        Expand the selected weight classes back into individual gate columns
        and return the (n × h) binary matrix M, where each column is one T-gate.

        For each active weight w, all C(n,w) binary vectors of weight w are
        included as columns (in lexicographic order of their integer index).

        Parameters
        ----------
        solver : cp_model.CpSolver  – already called .Solve(...)

        Returns
        -------
        M : np.ndarray, shape (n, h), dtype=int
        """
        n = self.n
        N = 1 << n
        gates = []
        for w in range(1, n):
            if solver.Value(self.v[w - 1]):
                for k in range(1, N - 1):
                    if bin(k).count('1') == w:
                        gates.append([(k >> i) & 1 for i in range(n)])
        if not gates:
            return np.zeros((n, 0), dtype=int)
        return np.array(gates, dtype=int).T

    # -------------------------------------------------------- Sn shim for SATSolution.get_M

    @property
    def Sn(self):
        """
        Weight-class representatives as an (n-1, n) array.
        v[k]=1 selects the representative of weight k+1: the vector with
        bits 0..k set.  Used by SATSolution.get_M -> sol_to_mat as a
        minimal valid gate matrix (one representative column per active class).
        For the full expanded matrix use extract_gate_matrix() directly.
        """
        n = self.n
        return np.array(
            [[(1 if i < w else 0) for i in range(n)] for w in range(1, n)],
            dtype=int,
        )



from itertools import product as iproduct, combinations_with_replacement
from math import comb
from collections import defaultdict
import numpy as np
from ortools.sat.python import cp_model


def compute_young_orbits(partition):
    """
    Enumerate all orbits of {1, …, 2^n − 2} under the Young subgroup
    S_{λ_1} × S_{λ_2} × … × S_{λ_p}  (partition = [λ_1, …, λ_p], sum = n).

    An orbit is identified by its weight-tuple (w_1, …, w_p) where w_i is
    the number of 1-bits in block i.  Orbit size = ∏ C(λ_i, w_i).

    We exclude the all-zeros tuple  (w_i = 0 for all i)
    and the all-full  tuple  (w_i = λ_i for all i).

    Returns
    -------
    orbits       : dict  (w_tuple) -> list[int]   gate integers in that orbit
    orbit_index  : dict  (w_tuple) -> int          index into variable list
    weight_tuples: list[(w_1,…,w_p)]               ordered list of orbit labels
    """
    partition = list(partition)
    p = len(partition)
    n = sum(partition)
    N = 1 << n

    # Precompute block bit-masks and offsets
    offsets = []
    off = 0
    for lam in partition:
        offsets.append(off)
        off += lam
    masks = [(((1 << lam) - 1) << offsets[i]) for i, lam in enumerate(partition)]

    orbits = {}
    all_zeros = tuple(0 for _ in partition)
    all_full  = tuple(partition)

    for k in range(1, N - 1):
        wtup = tuple(bin(k & masks[i]).count('1') for i in range(p))
        if wtup == all_zeros or wtup == all_full:
            continue
        orbits.setdefault(wtup, []).append(k)

    weight_tuples = sorted(orbits.keys())
    orbit_index   = {wt: i for i, wt in enumerate(weight_tuples)}
    return orbits, orbit_index, weight_tuples


class TrioOrthogonalSAT_YoungSym:
    """
    Triorthogonal SAT solver reduced under the Young subgroup
    G_λ = S_{λ_1} × S_{λ_2} × … × S_{λ_p},  with  ∑ λ_i = n.

    One boolean variable per G_λ-orbit of gate columns.  Selecting orbit
    (w_1, …, w_p) includes ALL ∏ C(λ_i, w_i) gate columns whose i-th block
    has Hamming weight exactly w_i.

    Triorthogonality
    ----------------
    We need one XOR-parity constraint per G_λ-orbit of qubit subsets of
    size t ∈ {1, 2, 3}.  An orbit of qubit subsets is characterised by a
    "type vector"  τ = (τ_1, …, τ_p)  with  τ_i ≤ λ_i  and  ∑ τ_i = t.

    For a representative subset of type τ, the parity contribution of orbit
    (w_1, …, w_p) is

        ∏_i  C(λ_i − τ_i,  w_i − τ_i)  mod 2.

    The constraint is:  XOR of v_{w_tuple}  over all orbits with odd
    product = 1.

    Distance constraints
    --------------------
    A logical error of physical weight r corresponds to a multiset of r
    orbit labels {(w^(1)_1, …, w^(1)_p), …, (w^(r)_1, …, w^(r)_p)} such
    that, for every block i, the list (w^(1)_i, …, w^(r)_i) can XOR
    to the all-ones vector of length λ_i — verified independently per block
    with the standard DP reachability check.

    Each such forbidden multiset becomes one AddBoolOr clause on the negated
    v-variables (repeated orbit indices collapsed by AND-idempotence).

    Gate-count constraint
    ---------------------
    Total gates = ∑_{selected orbits} ∏_i C(λ_i, w_i).
    Controlled by exact_gate / max_gate.

    Parameters
    ----------
    partition  : list[int]  – block sizes [λ_1, …, λ_p], must sum to n
    min_dist   : int        – minimum code distance  (default 3)
    max_gate   : int        – gate-count upper bound  (0 → 2^(n−1))
    exact_gate : int        – fix gate count exactly  (0 → unconstrained)
    """

    def __init__(self, partition, min_dist=3, max_gate=0, exact_gate=0):
        partition = list(partition)
        assert all(l >= 1 for l in partition), "All block sizes must be >= 1"
        assert len(partition) >= 1
        self.partition  = partition
        self.n          = sum(partition)
        self.p          = len(partition)
        self.min_dist   = min_dist
        self.max_gate   = max_gate
        self.exact_gate = exact_gate

        self.model = cp_model.CpModel()

        (self.orbits,
         self.orbit_index,
         self.weight_tuples) = compute_young_orbits(partition)

        self.v = [
            self.model.NewBoolVar(f"v_{'_'.join(map(str, wt))}")
            for wt in self.weight_tuples
        ]

        self._triorthogonal_constraints()
        self._distance_constraints()
        self._gate_constraints()

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _comb_parity(n, k):
        """C(n, k) mod 2 via Lucas' theorem."""
        if k < 0 or k > n:
            return 0
        return int((n & k) == k)

    @staticmethod
    def _xor_reachable(weights, target):
        """
        Can binary vectors of the given Hamming weights XOR to 1^{target}?
        Standard DP on reachable XOR-sums in {0, …, target}.

        A weight-w vector can adjust the current sum s to any value in
        {|s − w|, |s − w| + 2, …, min(s + w, target)}  (must stay ≤ target
        and preserve parity of sum−s).
        """
        reachable = {0}
        for w in weights:
            new_reachable = set()
            for s in reachable:
                lo = abs(s - w)
                hi = min(s + w, target)
                for t in range(lo, hi + 1, 2):
                    new_reachable.add(t)
            reachable = new_reachable
        return target in reachable

    def _orbit_parity_for_type(self, tau):
        """
        For a qubit-subset of type τ = (τ_1, …, τ_p), return the set of
        orbit indices whose parity contribution is odd, i.e.

            ∏_i  C(λ_i − τ_i,  w_i − τ_i)  ≡ 1  (mod 2).
        """
        odd_indices = []
        for idx, wt in enumerate(self.weight_tuples):
            prod_parity = 1
            for i, (lam, tau_i, w_i) in enumerate(
                    zip(self.partition, tau, wt)):
                prod_parity *= self._comb_parity(lam - tau_i, w_i - tau_i)
                if prod_parity == 0:
                    break
            if prod_parity % 2 == 1:
                odd_indices.append(idx)
        return odd_indices

    # ----------------------------------------- triorthogonality constraints

    def _triorthogonal_constraints(self):
        """
        For t = 1, 2, 3: enumerate all type vectors τ = (τ_1, …, τ_p) with
        ∑ τ_i = t and τ_i ≤ λ_i, build one XOR constraint per type.
        """
        model = self.model
        v     = self.v

        for t in range(1, 4):
            # generate all type vectors τ with ∑ τ_i = t, 0 ≤ τ_i ≤ λ_i
            type_vectors = self._type_vectors(t)
            for tau in type_vectors:
                odd_idx = self._orbit_parity_for_type(tau)
                if not odd_idx:
                    # no orbit contributes — the constraint is 0 = 1: infeasible
                    raise ValueError(
                        f"Triorthogonality infeasible for partition "
                        f"{self.partition}, type τ={tau}  (no orbit has odd "
                        f"parity contribution)."
                    )
                lits = [v[i] for i in odd_idx]
                model.AddBoolXOr(lits)

    def _type_vectors(self, t):
        """
        All non-negative integer vectors (τ_1, …, τ_p) with
        ∑ τ_i = t  and  τ_i ≤ λ_i.

        Uses a simple recursive generator.
        """
        partition = self.partition
        p         = self.p

        def _gen(block, remaining, current):
            if block == p:
                if remaining == 0:
                    yield tuple(current)
                return
            lam = partition[block]
            for tau_i in range(min(remaining, lam) + 1):
                current.append(tau_i)
                yield from _gen(block + 1, remaining - tau_i, current)
                current.pop()

        return list(_gen(0, t, []))

    # ------------------------------------------------- distance constraints

    def _distance_constraints(self):
        """
        For every r in {2, …, min_dist − 1}: forbid every multiset of r
        orbit labels whose XOR-to-all-ones is realisable simultaneously in
        every block.

        The feasibility check for block i: does _xor_reachable(w_i_list, λ_i)
        return True?  We require this for ALL blocks simultaneously.
        """
        model         = self.model
        v             = self.v
        r_count       = len(self.weight_tuples)
        partition     = self.partition
        weight_tuples = self.weight_tuples

        clauses = set()
        for r in range(2, self.min_dist):
            for orbit_multiset in combinations_with_replacement(
                    range(r_count), r):
                # per-block weight lists
                w_lists = [
                    [weight_tuples[idx][i] for idx in orbit_multiset]
                    for i in range(self.p)
                ]
                # check all blocks simultaneously
                if all(
                    self._xor_reachable(w_lists[i], partition[i])
                    for i in range(self.p)
                ):
                    # collapse repeated orbit indices (AND-idempotent)
                    clause = tuple(sorted(set(orbit_multiset)))
                    clauses.add(clause)

        for clause in clauses:
            model.AddBoolOr([v[k].Not() for k in clause])

    # ---------------------------------------------------- gate constraints

    def _gate_constraints(self):
        """
        Total gate count = ∑_{k: v[k]=1} orbit_size[k]
        where orbit_size[k] = ∏_i C(λ_i, w_i).
        """
        n         = self.n
        v         = self.v
        model     = self.model
        partition = self.partition

        orbit_sizes = [
            int(np.prod([comb(partition[i], wt[i])
                         for i in range(self.p)]))
            for wt in self.weight_tuples
        ]
        total_gates = cp_model.LinearExpr.WeightedSum(v, orbit_sizes)

        max_gate = (1 << (n - 1)) if self.max_gate == 0 else self.max_gate

        if self.min_dist > 3:
            model.Add(total_gates > 48)
        if self.exact_gate > 0:
            model.Add(total_gates == self.exact_gate)
        else:
            model.Add(total_gates <= max_gate)

    # -------------------------------------------------- solution extraction

    def extract_gate_matrix(self, solver):
        """
        Expand the selected orbits into all individual gate columns and
        return the (n × h) binary matrix M.

        For each active orbit (w_1, …, w_p), every gate integer whose i-th
        block has exactly w_i bits set is included as a column of M.

        Parameters
        ----------
        solver : cp_model.CpSolver  – already called .Solve(...)

        Returns
        -------
        M : np.ndarray, shape (n, h), dtype=int
        """
        gates = []
        for idx, wt in enumerate(self.weight_tuples):
            if solver.Value(self.v[idx]):
                for k in self.orbits[wt]:
                    gates.append([(k >> j) & 1 for j in range(self.n)])
        if not gates:
            return np.zeros((self.n, 0), dtype=int)
        return np.array(gates, dtype=int).T

    @property
    def Sn(self):
        """
        One representative column per orbit, shape (num_orbits, n).

        The representative of orbit (w_1, …, w_p) has the first w_i bits
        of block i set to 1 and the rest to 0.

        Used by SATSolution.get_M → sol_to_mat for a minimal valid matrix
        (one column per orbit).  For the full expanded matrix, call
        extract_gate_matrix(solver) after solving.
        """
        rows = []
        off = 0
        block_offsets = []
        for lam in self.partition:
            block_offsets.append(off)
            off += lam

        for wt in self.weight_tuples:
            col = [0] * self.n
            for i, (lam, w_i) in enumerate(zip(self.partition, wt)):
                start = block_offsets[i]
                for j in range(w_i):
                    col[start + j] = 1
            rows.append(col)
        return np.array(rows, dtype=int)

    # --------------------------------------------------------- introspection

    def num_variables(self):
        """Number of SAT boolean variables (= number of orbits)."""
        return len(self.weight_tuples)

    def variable_summary(self):
        """
        Print a table of all orbit variables with their sizes.
        """
        print(f"Partition λ = {self.partition},  n = {self.n}")
        print(f"{'orbit (w_1,…,w_p)':<30} {'orbit size':>12}")
        print("-" * 44)
        for idx, wt in enumerate(self.weight_tuples):
            size = int(np.prod([comb(self.partition[i], wt[i])
                                for i in range(self.p)]))
            print(f"{str(wt):<30} {size:>12}")
        print(f"\nTotal orbits (variables): {self.num_variables()}")
        max_gates = sum(
            int(np.prod([comb(self.partition[i], wt[i])
                         for i in range(self.p)]))
            for wt in self.weight_tuples
        )
        print(f"Max possible gates (all orbits selected): {max_gates}")
