import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# ---- academic typography via LaTeX ----
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
})

_palette = {
    3: '#e6194b',  # vivid red
    4: '#f58231',  # orange
    5: '#bfef45',  # lime
    6: '#3cb44b',  # green
    7: '#4363d8',  # blue
    8: '#911eb4',  # purple
    9: '#42d4f4',  # cyan
}

COL_CHECK  = _palette[8]   # blue for check-qubit cells / segments
COL_OUTPUT = _palette[4]   # orange for the output qubit
COL_WIN    = '#e8e8e8'     # active-window shading
COL_TRACKBG= '#f4f4f4'

B    = np.load('data/49-to-1_bravyi.npy')
MGW  = np.load('data/49-to-1_min_gen_weight.npy')
COMP = np.load('data/49-to-1_compressed.npy')
k=1; n,m = B.shape

def windows(M):
    f=[m]*n; l=[-1]*n
    for i in range(n):
        o=np.where(M[i])[0]
        if len(o): f[i]=int(o[0]); l[i]=int(o[-1])
    tl=[(m-1) if i<k else l[i] for i in range(n)]
    return f,l,tl

def draw_matrix(ax, M, show_windows=False, title=""):
    f,l,tl = windows(M)
    if show_windows:
        for i in range(n):
            if l[i] < f[i]: continue
            ax.add_patch(Rectangle((f[i], i), tl[i]-f[i]+1, 1,
                                   facecolor=COL_WIN, edgecolor='none', zorder=0))
    # cells
    rects_chk=[]; rects_out=[]
    for i in range(n):
        for j in range(m):
            if M[i,j]:
                (rects_out if i<k else rects_chk).append(Rectangle((j,i),1,1))
    ax.add_collection(PatchCollection(rects_chk, facecolor=COL_CHECK, edgecolor='none', zorder=2))
    ax.add_collection(PatchCollection(rects_out, facecolor=COL_OUTPUT, edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((0,0), m, n, fill=False, edgecolor='#555', lw=0.6, zorder=3))
    ax.set_xlim(-0.5, m+1.5); ax.set_ylim(n, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc='left', fontweight='bold', pad=4)
    ax.set_aspect('equal')

def slot_assignment(M):
    """Recycle freely (lowest free slot), then relabel so the output sits on slot 0.
    Reserving a slot for the output would waste it and inflate the count, so we
    recycle first and rename afterwards (a pure cosmetic relabel)."""
    f,l,tl = windows(M)
    opens={p:[] for p in range(m)}; closes={p:[] for p in range(m)}
    for i in range(n):
        if l[i]>=f[i]:
            opens[f[i]].append(i)
            if i>=k: closes[l[i]].append(i)
    slot={}; free=[]; nxt=[0]
    def al(i):
        if free: s=min(free); free.remove(s)
        else: s=nxt[0]; nxt[0]+=1
        slot[i]=s; return s
    for p in range(m):
        for i in sorted(opens[p]): al(i)
        for i in sorted(closes[p]):
            if i>=k: free.append(slot[i])
    nslots = nxt[0]
    # cosmetic relabel: send the output's slot to 0, shift the rest to fill in
    out_slot = slot[0]
    remap = {out_slot: 0}
    nextlabel = 1
    for s in range(nslots):
        if s == out_slot: continue
        remap[s] = nextlabel; nextlabel += 1
    slot = {i: remap[s] for i,s in slot.items()}
    return slot, nslots, f, tl

def draw_schedule(ax, M, title="", target_rows=14):
    """Draw the recycled schedule. Tracks are scaled so the panel occupies the
    same vertical extent as a `target_rows`-row matrix panel (aspect equal)."""
    slot, nslots, f, tl = slot_assignment(M)
    th = target_rows / nslots          # track height so nslots*th == target_rows
    for s in range(nslots):
        y = s*th
        ax.add_patch(Rectangle((0, y), m, th, facecolor=COL_TRACKBG, edgecolor='none', zorder=0))
    for i in range(n):
        if tl[i] < f[i]: continue
        s = slot[i]; y = s*th
        col = COL_OUTPUT if i<k else COL_CHECK
        pad = 0.12*th
        ax.add_patch(Rectangle((f[i], y+pad), tl[i]-f[i]+1, th-2*pad,
                               facecolor=col, edgecolor='none', zorder=2))
        if i>=k:  # measurement marker at the release point
            ax.scatter([tl[i]+1], [y+th/2], marker='D', s=24,
                       facecolor='white', edgecolor='#222', linewidth=0.7, zorder=4)
    ax.add_patch(Rectangle((0,0), m, nslots*th, fill=False, edgecolor='#555', lw=0.6, zorder=3))
    ax.set_xlim(-0.5, m+1.5); ax.set_ylim(nslots*th, 0)
    ax.set_xticks([])
    ax.set_yticks([(s+0.5)*th for s in range(nslots)])
    ax.set_yticklabels([f"$q_{{{s}}}$" for s in range(nslots)])
    ax.set_title(title, loc='left', fontweight='bold', pad=4)
    ax.set_aspect('equal')
    return slot, nslots

def active_trace(M):
    f,l,tl = windows(M)
    return [sum(1 for i in range(n) if f[i]<=j<=tl[i]) for j in range(m)]

# ---- layout: 4 equal-height panels + trace ----
fig = plt.figure(figsize=(7.0, 9.6))
gs = fig.add_gridspec(5, 1, height_ratios=[14,14,14,14,4], hspace=0.3)

axa = fig.add_subplot(gs[0]); draw_matrix(axa, B, title="(a) Original $49$-to-$1$ matrix from [Bravyi2012]")
axb = fig.add_subplot(gs[1]); draw_matrix(axb, MGW, title="(b) After generator weights minimization")
axc = fig.add_subplot(gs[2]); draw_matrix(axc, COMP, show_windows=True, title="(c) After column ordering")
axd = fig.add_subplot(gs[3]); draw_schedule(axd, COMP, title="(d) Recycled schedule on $N=5$ qubits")

axt = fig.add_subplot(gs[4], sharex=axd)
a = active_trace(COMP)
xs = np.arange(m+1)
ys = np.array(a+[a[-1]])
axt.step(xs, ys, where='post', color=COL_CHECK, lw=1.0)
axt.axhline(5, color=_palette[3], ls='--', lw=0.8)
#axt.text(m+0.5, 5, r'$N=5$', va='center', ha='left', fontsize=8, color='#555')
axt.set_xlim(-0.5, m+1.5); axt.set_ylim(0, 6)
axt.set_yticks([0,5]); axt.set_yticklabels(['0','5'], fontsize=8)
axt.set_ylabel('$a(j)$', fontsize=9)
axt.set_xlabel('column index $j$ (Pauli product rotations)', fontsize=9)
axt.spines[['top','right']].set_visible(False)

# legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles=[Patch(facecolor=COL_OUTPUT, label='output qubit (active until end)'),
         Patch(facecolor=COL_CHECK, label='check qubit (measured out, recycled)'),
         Line2D([0],[0], marker='D', markerfacecolor='white', markeredgecolor='#222',
                markersize=7, linewidth=0, label='measurement')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.005))

# align the trace horizontally with the (aspect-equal) panels above it
fig.canvas.draw()
pd = axd.get_position()
pt = axt.get_position()
axt.set_position([pd.x0, pt.y0, pd.width, pt.height])

fig.savefig('recycling.pdf', bbox_inches='tight', dpi=200)
