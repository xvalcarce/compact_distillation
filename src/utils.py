import numpy as np
from sympy import Symbol, series

def mappingLitinski(p):
    """
    Map the rotation representation to Litinksi's (not as costly).
    """
    r = np.zeros_like(p)
    r[0] = p[0]
    for k in range(1, len(p)):
        r[k] = p[k] ^ p[k-1]
    return r

def mappingrep(p):
    """
    Map the lintinski repr to the repetition code one
    """
    return np.bitwise_xor.accumulate(p, axis=0)

def is_triorthogonal(G):
    """
    Whether a matrix is triorthogonal (Litinksi mapping required).
    """
    n,_ = G.shape
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(G[i]*G[j])%2!=0:
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if np.sum(G[i]*G[j]*G[k])% 2!=0:
                    return False
    return True

def is_triorthogonal_rep(G):
    """
    Whether a matrix is triorthogonal for the reptition code.
    """
    n,_ = G.shape
    for i in range(n):
        if np.sum(G[i])%2 != 1:
            print(i)
            return False
        for j in range(i + 1, n):
            if np.sum(G[i]*G[j])%2!=1:
                print(i,j)
                return False
            for k in range(j + 1, n):
                if np.sum(G[i]*G[j]*G[k])% 2!=1:
                    print(i,j,k)
                    return False
    return True

def is_CCZ_repr(G):
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
 


def sol_to_mat(v,Sn):
    """makes the rm matrix from the cp-sat solution"""
    nz = np.nonzero(v)[0]
    return Sn[nz,:].T

def coeff_dev(M):
    """return the exponential coeficients |f| appearing in the formula (20)
    at the numerator (cc0 return) and denominator (cc1 return)
   
    """ 
    N = M.transpose()
    n, _ = M.shape
    
    # Gram-Schmidt-like reduction on binary field
    N2 = np.zeros_like(N)
    N2[:, 0] = N[:, 0]
    N2[:, 1:] = np.bitwise_xor(N[:, 1:], N[:, :-1])

    coef = np.array([[(k // (1<<i)) % 2 for i in range(n)] for k in range(1<<n)])

    # G0 contains the weights of the vectors in G0 where they are all the possible combinations of 
    # columns of N2 except the first one (corresponding to logical X operator)
    G0 = []
    for k in range(1<<(n-1)) : 
        t = N2[:,1:]*coef[k,:n-1]
        G0.append(np.bitwise_xor.reduce(t,axis=1).sum())

    G1 = []
    for k in range(1 << n):
        t = N2[:,1:] * coef[k,:n-1] 
        extra = coef[k, n-1] * N2[:, 0] #include first column
        t = np.column_stack([t, extra])
        G1.append(np.bitwise_xor.reduce(t, axis=1).sum())

    max_w = max(max(G0, default=0), max(G1, default=0))
    count0 = np.bincount(G0, minlength=max_w+1).tolist()
    count1 = np.bincount(G1, minlength=max_w+1).tolist()
    # nz
    cc0 = [[w, c] for w, c in enumerate(count0) if c > 0]
    cc1 = [[w, c] for w, c in enumerate(count1) if c > 0]
    return (cc0,cc1)

def Partial_exp(c,dist) : #gives the Taylor serie of (20) at the distance of interest 'dist' 
    cc0 , cc1 = c[0], c[1]
    X = Symbol('X')
    f = 1-(sum(cc1[k][1]*(1-2*X)**cc1[k][0] for k in range(len(cc1)))/sum(cc0[k][1]*(1-2*X)**cc0[k][0] for k in range(len(cc0))))/2
    return series(f,n=dist+1)

def compute_footprint(G,k=1):
    """
    Compute footprint of binary matrix G, with k output rows.
    """
    n, m = G.shape
    f = []
    l = []
    for i in range(n):
        ones = [j for j in range(m) if G[i, j] == 1]
        if not ones:
            f.append(m)
            l.append(-1)
        else:
            f.append(min(ones))
            l.append(max(ones))
    max_active = 0
    for j in range(m):
        active = 0
        for i in range(n):
            if j >= f[i] and (j <= l[i] or i < k):
                active += 1
            if active > max_active:
                max_active = active
    return max_active
