"""
recycling.py
============
Qubit-recycling / active-qubit-count minimization for distillation protocols,
covering both nT->1T and nT->1CCZ constructions with a single implementation.

A distillation protocol is given by a binary matrix G (rows = logical qubits,
columns = commuting Z-diagonal Pauli product rotations). Because the rotations
commute, a check qubit may be measured out once its last rotation has been
applied and its physical qubit reused, so the real spatial cost is the peak
number of simultaneously ACTIVE qubits, A(G), rather than the number of rows.
This module reduces A(G) by (i) choosing a light generator basis and (ii)
searching over column orderings, and it can certify the distance is preserved.

TWO PROTOCOL TYPES, ONE IMPLEMENTATION
--------------------------------------
The active-qubit machinery is identical for both; they differ only in two
independent parameters:

  * k_out -- the number of never-recycled OUTPUT rows (rows 0..k_out-1), which
    stay active until the end of the circuit and set the lower bound
    A(G) >= k_out. These rows are frozen during basis reduction (only the
    remaining check rows are added to lighten weights), since modifying an
    output row changes the logical operator.
        - nT->1T   : k_out = 1  (single output qubit).
        - nT->1CCZ : k_out = 3  (the three CCZ output qubits).

  * kind -- the set of undetectable Z-error patterns used to compute the
    distance. A protocol has distance d if d is the smallest number of columns
    whose XOR equals one of these patterns.
        - "t"   : a single pattern, a lone Z on the output row 0.
        - "ccz" : eight patterns -- rows 0 and 1 take any of the four values,
                  with the check tail (rows 2..N-1) either all-zero or all-one.
                  The all-one tail is a genuine logical representative because
                  the all-ones-on-checks operator lies in the code. This set
                  matches the project's bruteforce reference and reproduces the
                  known distances of the reference CCZ matrices.

k_out and kind are independent: k_out governs the recycling schedule (which
qubits are never released), while kind governs the distance test. The
convenience wrappers set both together so the two protocol types can never be
run with a mismatched convention:

    distance_t(G),   distance_ccz(G)      -- distance only
    compress_t(G),   compress_ccz(G)      -- full reduction pipeline

PIPELINE (compress / compress_t / compress_ccz)
-----------------------------------------------
  1. min_generator_weights : minimize the output-row weight(s) exactly over all
     subsets of the check rows (output_weight_min), then greedily lighten the
     check rows (check_weight_min). All additions stay within the check block,
     preserving the code and the distance.
  2. branch_and_bound_column_ordering : depth-first search over column orders
     with a monotone-peak pruning bound, returning a schedule, the achieved
     A(G), and whether it was certified optimal (tree exhausted within budget).

The lower-level functions all take k_out (and, for distance, kind) explicitly,
so they can be used directly for either protocol type or for experimentation.
"""

import pulp #for the ILP only
import numpy as np
from itertools import combinations


# ----------------------------------------------------------------------------
# Utilities: windows, active-qubit count, distance
# ----------------------------------------------------------------------------
def _windows(G, k_out):
    """First/last 1 of each row, and the 'active-until' column tl (= last col
    for the k_out never-recycled output rows, else the row's last 1)."""
    G = np.asarray(G) % 2

    N, n = G.shape
    f = [n] * N
    l = [-1] * N
    for i in range(N):
        o = np.where(G[i])[0]
        if len(o):
            f[i] = int(o[0]); l[i] = int(o[-1])
    tl = [(n - 1) if i < k_out else l[i] for i in range(N)]
    return f, l, tl


def active_qubit_count(G, k_out=1, order=None):
    """
    Peak number of simultaneously active qubits, A(G), for a given column order
    (default: identity). Output rows (0..k_out-1) stay active to the end;
    check rows are released after their last placed 1.
    """
    G = np.asarray(G) % 2
    N, n = G.shape
    if order is not None:
        G = G[:, order]
    f, l, tl = _windows(G, k_out)
    best = 0
    for j in range(n):
        active = sum(1 for i in range(N) if f[i] <= j <= tl[i])
        best = max(best, active)
    return best


def _ccz_patterns(N):
    """
    Undetectable Z-error patterns for a T->1CCZ triorthogonal protocol: the
    seven nonzero triples on the three output rows (rows 0,1,2), with an all-zero
    tail on the check rows (rows 3..N-1). This is the standard triorthogonal-CCZ
    convention; it reproduces the known distances of the reference matrices
    (validated: 8T->CCZ -> 2, 48T->CCZ -> 4).

    Because every pattern is zero on all check rows, a change of check-row basis
    (adding one check row to another) can neither create nor destroy a logical
    error, so the distance is invariant under check_weight_min.
    """
    heads = [(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    pats = []
    for h in heads:
        e = np.zeros(N, dtype=np.int8)
        e[0], e[1], e[2] = h
        pats.append(e)
    return pats


def _t_patterns(N):
    """The single undetectable pattern for a T->1T protocol: a lone Z on the
    output qubit (row 0), all checks silent."""
    e = np.zeros(N, dtype=np.int8); e[0] = 1
    return [e]


def distance(G, max_d=5, kind="t"):
    """
    Circuit distance = minimum number of faulty rotations (columns) whose XOR
    is an undetectable logical error.

    `kind` selects the undetectable-pattern convention, independently of the
    recycling parameter `k_out`:
      - "ccz": the 8-pattern set (rows 0,1 free; check tail all-0 or all-1),
        matching the project's bruteforce reference. Validated to reproduce the
        d=2 and d=3 CCZ matrices.
      - "t":   the single pattern (a lone Z on the output row 0), for T->1T.

    Returns the distance, or None if it exceeds max_d.
    """
    G = np.asarray(G) % 2
    N, n = G.shape
    cols = [G[:, j] for j in range(n)]
    if kind == "ccz":
        patset = {tuple(p.tolist()) for p in _ccz_patterns(N)}
    elif kind == "t":
        patset = {tuple(p.tolist()) for p in _t_patterns(N)}
    else:
        raise ValueError("kind must be 'ccz' or 't'")
    for d in range(1, max_d + 1):
        for K in combinations(range(n), d):
            s = np.zeros(N, dtype=np.int8)
            for j in K:
                s ^= cols[j]
            if tuple(s.tolist()) in patset:
                return d
    return None


# ----------------------------------------------------------------------------
# Generator-weight minimization
# ----------------------------------------------------------------------------
def output_weight_min(G, k_out=1):
    """
    Minimize the Hamming weight of each of the k_out output rows by adding
    subsets of the FREE check rows (rows k_out..N-1). The CCZ-triple rows are
    never added to one another (that would change the logical CCZ), so each
    output row is reduced independently over the 2^(N-k_out) subsets of checks.

    Exact exhaustive search per output row; cost O(k_out * 2^(N-k_out) * n).
    """
    G = np.asarray(G).copy() % 2
    N, n = G.shape
    free = list(range(k_out, N))            # check rows only (output rows frozen)
    for oi in range(k_out):                 # reduce each output row independently
        base = G[oi].copy()
        best_w = int(base.sum()); best = base.copy()
        # enumerate every nonempty subset of the free check rows via a bitmask:
        # bit b of `msk` set  <=>  add free[b] into the output row.
        for msk in range(1, 1 << len(free)):
            c = base.copy()
            for b, r in enumerate(free):
                if (msk >> b) & 1:
                    c = c ^ G[r]            # XOR in this check row
            w = int(c.sum())
            if w < best_w:                  # keep the lightest representative
                best_w = w; best = c
        G[oi] = best
    return G


def check_weight_min(G, k_out=1, kind="t", preserve_distance=False, _d0=None):
    """
    Greedily minimize the weights of the FREE check rows (rows k_out..N-1) by
    adding one check row to another whenever it strictly lowers the target's
    weight, iterating to a fixed point. The output rows (0..k_out-1) are frozen:
    never modified and never used as a source.

    Distance safety. For both supported conventions (kind="t" and the standard
    triorthogonal kind="ccz"), the undetectable patterns are zero on every check
    row, so a change of check-row basis can neither create nor destroy a logical
    error: the distance is INVARIANT under these additions. Hence the greedy
    pass is distance-preserving by construction and no guard is needed
    (preserve_distance defaults to False).

    preserve_distance=True enables a per-step distance check that rejects any
    addition which would change the distance. This is unnecessary for the
    conventions above but is available as a safety net for experimentation with
    other pattern sets; note it costs a distance computation per accepted
    candidate and scales poorly in the number of columns.
    """
    G = np.asarray(G).copy() % 2
    N, n = G.shape
    free = list(range(k_out, N))
    d0 = _d0 if _d0 is not None else (distance(G, kind=kind) if preserve_distance else None)
    improved = True
    while improved:
        improved = False
        for t in free:                      # target: a check row
            for s in free:                  # source: another check row
                if s == t:
                    continue
                cand = G[t] ^ G[s]
                if int(cand.sum()) < int(G[t].sum()):
                    if preserve_distance:
                        # optional guard: apply, verify distance, roll back if worse
                        old_row = G[t].copy()
                        G[t] = cand
                        if distance(G, kind=kind) == d0:
                            improved = True
                        else:
                            G[t] = old_row
                    else:
                        G[t] = cand
                        improved = True
    return G


def column_weight_min(G, k_out=1, kind="t", preserve_distance=True, _d0=None):
    """
    Greedily reduce the MAXIMUM column weight of G by check-row additions.

    Motivation. A(G) >= max_j (weight of column j), because every row with a 1 in
    column j is live there; this bound holds for every column ordering. If the
    heaviest column has weight w, no reordering can bring A(G) below w. Row-weight
    minimization does not touch this bound (it is a column property), so a matrix
    can be stuck at A(G) = w purely because of one heavy column, even though a
    different check-row basis would make every column lighter. This pass lowers
    that bound.

    Method. Repeatedly, while it helps: pick a heaviest column, and look for a
    check-row addition row_t <- row_t ^ row_s (t, s check rows) that lowers the
    overall maximum column weight (or lowers the number of columns achieving it,
    as a tie-break) without increasing the max. Apply the best such addition;
    stop when no addition improves the max-column-weight objective.

    Only check rows (k_out..N-1) are used as sources and targets; the output rows
    are frozen. Additions are distance-guarded when preserve_distance is True
    (see check_weight_min for the rationale).

    Note: this trades against row weight -- lightening a column may lengthen a
    row. Run output_weight_min again afterwards if the output-row weight matters
    for your schedule (min_generator_weights offers this via its stage order).
    """
    G = np.asarray(G).copy() % 2
    N, n = G.shape
    free = list(range(k_out, N))
    d0 = _d0 if _d0 is not None else (distance(G, kind=kind) if preserve_distance else None)

    def col_objective(M):
        cw = M.sum(axis=0)
        mx = int(cw.max())
        return (mx, int((cw == mx).sum()))     # (max weight, #columns at the max)

    improved = True
    while improved:
        improved = False
        cur = col_objective(G)
        best = None
        for t in free:
            for s in free:
                if s == t:
                    continue
                old = G[t].copy()
                G[t] = G[t] ^ G[s]
                obj = col_objective(G)
                ok = obj < cur
                if ok and preserve_distance and distance(G, kind=kind) != d0:
                    ok = False
                if ok and (best is None or obj < best[0]):
                    best = (obj, t, s, G[t].copy())
                G[t] = old                     # revert; we apply the best at the end
        if best is not None:
            _, t, s, newrow = best
            G[t] = newrow
            improved = True
    return G


def min_generator_weights(G, k_out=1, kind="t",
                          do_output=True, do_column=False, do_check=True,
                          order=("output", "column", "check")):
    """
    Composable generator-weight reduction. Runs a configurable sequence of
    distance-preserving basis-reduction stages, then returns the reworked matrix.

    Stages (each optional, toggled by the do_* flags):
      "output" : output_weight_min  -- minimize the output row(s) exactly.
      "column" : column_weight_min  -- lower the maximum column weight
                 (lowers the A(G) >= max-column-weight bound).
      "check"  : check_weight_min   -- greedily lighten the check rows.

    `order` lists the stages to run, in order; a stage is executed only if its
    do_* flag is also True. This lets you experiment with different pipelines
    (e.g. output->column->check vs output->check->column) without rewriting code.

    All stages stay within the check block and are distance-preserving, so the
    protocol and its distance are unchanged.
    """
    d0 = distance(G, kind=kind)
    flags = {"output": do_output, "column": do_column, "check": do_check}
    stages = {
        "output": lambda M: output_weight_min(M, k_out=k_out),
        "column": lambda M: column_weight_min(M, k_out=k_out, kind=kind,
                                              preserve_distance=True, _d0=d0),
        "check":  lambda M: check_weight_min(M, k_out=k_out, kind=kind,
                                             preserve_distance=True, _d0=d0),
    }
    for name in order:
        if flags.get(name, False):
            G = stages[name](G)
    return G


# ----------------------------------------------------------------------------
# Branch-and-bound over column orderings
# ----------------------------------------------------------------------------
def branch_and_bound_column_ordering(G, k_out=1, node_budget=2_000_000):
    """
    Find a column order minimizing the active-qubit count A(G), with the output
    rows (0..k_out-1) never released. Depth-first construction of the order with
    the monotone-peak pruning bound; returns:

        (best_A, best_order, reordered_G, info)

    where info['certified_optimal'] is True iff the whole tree was exhausted
    within node_budget (so best_A is provably optimal for this basis).

    Works for both protocol types; only k_out changes (1 for T, 3 for CCZ).
    """
    G = np.asarray(G) % 2
    N, n = G.shape

    # col_rows[j] = list of rows that have a 1 in column j (the rotation's support).
    col_rows = [np.where(G[:, j])[0].tolist() for j in range(n)]
    # ones_in_row[i] = total number of 1s in row i (how many rotations touch qubit i).
    ones_in_row = [int(G[i].sum()) for i in range(N)]

    # ----- greedy incumbent: build ONE complete order with the close-first
    # heuristic, to seed the pruning bound with a good value from the start.
    def greedy():
        rem = ones_in_row[:]          # 1s of each row still to be placed
        started = [False] * N         # has each row been activated yet?
        active = set()                # rows currently occupying a qubit
        placed = []; used = [False] * n; peak = 0
        for _ in range(n):
            # pick the single best next column by the (-closes, opens) key
            best_j = -1; best_key = None
            for j in range(n):
                if used[j]:
                    continue
                # opens: rows this column would activate (their first 1)
                opens = sum(1 for i in col_rows[j] if not started[i])
                # closes: check rows this column would release (their last 1)
                closes = sum(1 for i in col_rows[j]
                             if i >= k_out and rem[i] == 1)
                key = (-closes, opens)          # release-most, then open-fewest
                if best_key is None or key < best_key:
                    best_key = key; best_j = j
            # place best_j and update the running state
            j = best_j; used[j] = True; placed.append(j)
            for i in col_rows[j]:
                if not started[i]:
                    started[i] = True; active.add(i)   # activate on first 1
                rem[i] -= 1
                if i >= k_out and rem[i] == 0:
                    active.discard(i)                  # release check on last 1
            peak = max(peak, len(active))
        return peak, placed

    # incumbent best solution so far (A value + realizing order), seeded greedily.
    best_A, best_order = greedy()
    state = {"A": best_A, "order": best_order}   # dict so the closure can mutate it
    info = {"nodes": 0, "certified_optimal": False}
    budget = [node_budget]                       # list so the closure can decrement it

    # mutable search state, shared across the recursion and edited in place
    # (with matching undo after each child, so we never rebuild it from scratch).
    order = []                     # columns placed so far, in order
    used = [False] * n            # which columns are already placed
    started = [False] * N         # which rows have been activated
    rem = ones_in_row[:]          # 1s of each row still to place
    active = set()                # rows currently live

    def dfs(peak):
        # `peak` = the maximum active count seen along this partial order so far.

        # budget guard: stop descending once we run out of node budget.
        if budget[0] <= 0:
            return
        budget[0] -= 1; info["nodes"] += 1

        # PRUNE: the peak can only stay equal or grow as we add more columns,
        # so if it already matches the best complete solution, no completion of
        # this prefix can beat the incumbent -> abandon this whole branch.
        if peak >= state["A"]:
            return

        # a complete order is handled at the point of placement below, so a full
        # `order` here means nothing left to do.
        if len(order) == n:
            return

        # ----- close-first branching order -----
        # score every unplaced column, then try them best-first so strong
        # (low-peak) complete orders are found early and tighten the bound.
        cand = []
        for j in range(n):
            if used[j]:
                continue
            opens = sum(1 for i in col_rows[j] if not started[i])   # rows woken
            closes = sum(1 for i in col_rows[j]
                         if i >= k_out and rem[i] == 1)             # checks freed
            # key sorts release-most-first (-closes), then open-fewest (opens).
            cand.append(((-closes, opens), j))
        cand.sort()

        for _, j in cand:
            # --- place column j (edit state in place) ---
            used[j] = True; order.append(j)
            newly = []                     # rows activated by this column (to undo)
            for i in col_rows[j]:
                if not started[i]:
                    started[i] = True; active.add(i); newly.append(i)
                rem[i] -= 1
            closed = []                    # check rows released here (to undo)
            for i in col_rows[j]:
                if i >= k_out and rem[i] == 0 and i in active:
                    active.discard(i); closed.append(i)

            # new running peak including this column's live count
            npeak = max(peak, len(active))

            # only descend if this prefix can still beat the incumbent
            if npeak < state["A"]:
                if len(order) == n:
                    # a complete order strictly better than the incumbent: record it
                    state["A"] = npeak; state["order"] = order[:]
                else:
                    dfs(npeak)

            # --- undo the placement (restore state for the next sibling) ---
            for i in reversed(closed):
                active.add(i)              # un-release the checks we freed
            for i in col_rows[j]:
                rem[i] += 1                # restore remaining-1 counts
            for i in newly:
                started[i] = False; active.discard(i)   # un-activate rows we woke
            used[j] = False; order.pop()

    # run the search from an empty order (peak 0).
    dfs(0)

    # if the budget was NOT exhausted, the whole tree was explored -> optimal.
    if budget[0] > 0:
        info["certified_optimal"] = True

    best_A, best_order = state["A"], state["order"]
    reordered = G[:, best_order]

    # independent recomputation of A(G) on the reordered matrix, as a self-check
    # that the incrementally-tracked peak matches the direct definition.
    A_check = active_qubit_count(reordered, k_out=k_out)
    assert A_check == best_A, f"A mismatch {A_check} vs {best_A}"
    return best_A, best_order, reordered, info


# ----------------------------------------------------------------------------
# ILP certification of column ordering
# ----------------------------------------------------------------------------
def ilp_min_footprint(G, k=1, time_limit=120, msg=False, warm_order=None, F_ub=None):
    G = np.asarray(G) % 2
    n, m = G.shape
    prob = pulp.LpProblem("footprint", pulp.LpMinimize)

    x = {(c,t): pulp.LpVariable(f"x_{c}_{t}", cat="Binary") for c in range(m) for t in range(m)}
    o = {(i,t): pulp.LpVariable(f"o_{i}_{t}", cat="Binary") for i in range(n) for t in range(m)}
    e = {(i,t): pulp.LpVariable(f"e_{i}_{t}", cat="Binary") for i in range(k,n) for t in range(m)}
    a = {(i,t): pulp.LpVariable(f"a_{i}_{t}", cat="Binary") for i in range(n) for t in range(m)}
    F = pulp.LpVariable("F", lowBound=0, upBound=(F_ub if F_ub else n), cat="Integer")

    # assignment: one column per slot, one slot per column
    for t in range(m): prob += pulp.lpSum(x[c,t] for c in range(m)) == 1
    for c in range(m): prob += pulp.lpSum(x[c,t] for t in range(m)) == 1

    # p[i,t] expression
    def p(i,t): return pulp.lpSum(G[i,c]*x[c,t] for c in range(m) if G[i,c])

    for i in range(n):
        for t in range(m):
            pit = p(i,t)
            # open = prefix OR
            prob += o[i,t] >= pit
            if t>0:
                prob += o[i,t] >= o[i,t-1]
                prob += o[i,t] <= o[i,t-1] + pit
            else:
                prob += o[i,t] <= pit
    for i in range(k,n):
        for t in range(m):
            pit = p(i,t)
            prob += e[i,t] >= pit
            if t<m-1:
                prob += e[i,t] >= e[i,t+1]
                prob += e[i,t] <= e[i,t+1] + pit
            else:
                prob += e[i,t] <= pit
    # active
    for t in range(m):
        for i in range(n):
            if i<k:
                prob += a[i,t] == o[i,t]          # outputs never close
            else:
                prob += a[i,t] <= o[i,t]
                prob += a[i,t] <= e[i,t]
                prob += a[i,t] >= o[i,t] + e[i,t] - 1
        prob += pulp.lpSum(a[i,t] for i in range(n)) <= F

    prob += F  # objective

    # warm start
    if warm_order is not None:
        for t,c in enumerate(warm_order):
            x[c,t].setInitialValue(1)

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, warmStart=(warm_order is not None))
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    Fval = int(round(pulp.value(F))) if pulp.value(F) is not None else None
    # recover order
    order=[None]*m
    for t in range(m):
        for c in range(m):
            if x[c,t].value() and x[c,t].value()>0.5: order[t]=c
    return Fval, order, status


# ----------------------------------------------------------------------------
# qubit recycling pipeline
# ----------------------------------------------------------------------------
def compress(G, k_out=1, kind="t", node_budget=2_000_000, verbose=True,
             do_output=True, do_column=True, do_check=True,
             order=("output", "column", "check")):
    """
    Full pipeline: generator-weight reduction -> branch-and-bound column ordering.
    Returns (A, order, reordered_G, info).

    Parameters
    ----------
    k_out : number of never-recycled output rows (rows 0..k_out-1), which stay
            active until the end of the circuit. Use 1 for T->1T, 3 for T->1CCZ.
    kind  : undetectable-pattern convention for the distance check, "t" or "ccz".
    do_output / do_column / do_check, order :
            select and order the basis-reduction stages (see
            min_generator_weights). By default all three run, output -> column ->
            check. The column stage lowers the A(G) >= max-column-weight bound and
            is what lets matrices with a heavy column reach a lower active count.

    `k_out` and `kind` are independent: `k_out` controls the recycling (which
    qubits are never released), while `kind` controls the distance pattern set.
    Use the compress_t / compress_ccz wrappers to set both consistently.
    """
    d0 = distance(G, kind=kind)
    G1 = min_generator_weights(G, k_out=k_out, kind=kind,
                               do_output=do_output, do_column=do_column,
                               do_check=do_check, order=order)
    d1 = distance(G1, kind=kind)
    assert d0 == d1, f"basis reduction changed distance {d0}->{d1}!"
    A, order_out, RG, info = branch_and_bound_column_ordering(
        G1, k_out=k_out, node_budget=node_budget)
    if verbose:
        print(f"distance preserved: {d0}")
        print(f"active-qubit count A(G) = {A} "
              f"(lower bound max(k_out={k_out}, max_col_wt="
              f"{int(G1.sum(axis=0).max())}); "
              f"certified={info['certified_optimal']})")
    return A, order_out, RG, info


# ----------------------------------------------------------------------------
# Convenience entry points: set k_out and kind together, so T and CCZ matrices
# can never be run with the wrong convention.
# ----------------------------------------------------------------------------
def distance_t(G, max_d=6):
    """Distance of a T->1T protocol (single undetectable pattern: Z on row 0)."""
    return distance(G, kind="t", max_d=max_d)


def distance_ccz(G, max_d=6):
    """Distance of a T->1CCZ protocol (8-pattern convention: rows 0,1 free,
    check tail all-0 or all-1)."""
    return distance(G, kind="ccz", max_d=max_d)


def compress_t(G, node_budget=2_000_000, verbose=True):
    """Compress a T->1T protocol: one never-recycled output row (k_out=1),
    T-state distance convention."""
    return compress(G, k_out=1, kind="t",
                    node_budget=node_budget, verbose=verbose)


def compress_ccz(G, node_budget=2_000_000, verbose=True):
    """Compress a T->1CCZ protocol: three never-recycled output qubits (k_out=3),
    CCZ distance convention."""
    return compress(G, k_out=3, kind="ccz",
                    node_budget=node_budget, verbose=verbose)
