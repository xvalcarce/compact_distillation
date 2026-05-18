import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ── Data ──────────────────────────────────────────────────────────────────────
points_SAT = [
    (10, 64, 4),
    (11, 65, 5),
]
points_canonical_family = [
    (10, 165, 4),
    (11, 165, 5),
    (16, 696, 6),
    (17, 697, 7),
    (22, 1771, 8),
    (23, 1771, 9),
]

points_15 = [
    (5, 15, 3)
]

points_49 = [
    (14, 49, 5),
]

div = 4
points_CCZ = [
    (9, 47/div, 3),
    (10, 48/div, 4),]



# ── Qualitative palette — one bold colour per integer distance ────────────────
all_d = sorted(set(p[2] for p in points_SAT + points_canonical_family + points_15 ))

_palette = {
    3: '#e6194b',  # vivid red
    4: '#f58231',  # orange
    5: '#bfef45',  # lime
    6: '#3cb44b',  # green
    7: '#4363d8',  # blue
    8: '#911eb4',  # purple
    9: '#42d4f4',  # cyan
}
d_to_color = {d: _palette[d] for d in all_d}

def col(d):
    return d_to_color[d]

# ── Broken y-axis: three segments ─────────────────────────────────────────────
seg_bot = (0,   175)
seg_mid = (680, 710)
seg_top = (1755, 1785)

r_bot = seg_bot[1] - seg_bot[0]   # 175
r_mid = seg_mid[1] - seg_mid[0]   # 30
r_top = seg_top[1] - seg_top[0]   # 30
total = r_bot + r_mid + r_top

fig = plt.figure(figsize=(7, 4.5))
gs = GridSpec(3, 1, figure=fig,
              height_ratios=[r_top/total, r_mid/total, r_bot/total],
              hspace=0.08)

# gs[0] = topmost row in figure  → highest y values
# gs[2] = bottommost row         → lowest y values
ax_top = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1])
ax_bot = fig.add_subplot(gs[2])

panel_info = [
    (ax_top, seg_top),
    (ax_mid, seg_mid),
    (ax_bot, seg_bot),
]

ms = 90

for ax, (ylo, yhi) in panel_info:
    ax.axhspan(0, 48,
               facecolor=d_to_color[3], alpha=0.3,
               hatch='//', edgecolor='tomato', linewidth=0.5, zorder=0)
    ax.axvspan(0, 7,
               facecolor=d_to_color[3], alpha=0.3,
               hatch=r'\\\\', edgecolor='tomato', linewidth=0.5, zorder=0)

    for pts, marker in [(points_SAT, 'D'),
                        (points_canonical_family, '^'),
                        (points_15, 's'),
                        (points_49, 's'),
                        (points_CCZ, '*')]:
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            c=[col(p[2]) for p in pts],
            s=ms, marker=marker, zorder=5,
            edgecolors='black', linewidths=0.8,
        )

    ax.set_xlim(0, 26)
    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, which="both")
    ax.tick_params(labelsize=10)

# ── Spine cosmetics ───────────────────────────────────────────────────────────
# ax_top  : hide bottom spine (break there)
# ax_mid  : hide both top and bottom spines (breaks on both sides)
# ax_bot  : hide top spine (break there)
ax_top.spines['bottom'].set_visible(False)
ax_mid.spines['top'].set_visible(False)
ax_mid.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)

ax_top.tick_params(bottom=False, labelbottom=False)
ax_mid.tick_params(top=False, bottom=False, labelbottom=False)
ax_bot.tick_params(top=False)

# ── Break diagonal marks ──────────────────────────────────────────────────────
# Each break is drawn on the BOTTOM edge of the upper panel and TOP edge of
# the lower panel — matching the hidden spines above.
d_s = 0.022   # half-width of slash in axes-fraction units
slope = 0.5   # dy/dx ratio

kw = dict(color='k', clip_on=False, linewidth=1.2)

def slash_marks(ax, edge):
    """Draw visually consistent diagonal slashes."""
    
    # axes width/height in display units
    bbox = ax.get_window_extent()
    w = bbox.width
    h = bbox.height

    # compensate for aspect distortion
    slope = (w / h)*0.5

    trans = ax.transAxes
    y0 = 1.0 if edge == 'top' else 0.0

    for x0 in (0.0, 1.0):
        ax.plot(
            [x0 - d_s, x0 + d_s],
            [y0 - d_s * slope, y0 + d_s * slope],
            transform=trans,
            **kw
        )

# Between ax_top (upper) and ax_mid (lower): marks on bottom of ax_top and top of ax_mid
slash_marks(ax_top, 'bottom')
slash_marks(ax_mid, 'top')

# Between ax_mid (upper) and ax_bot (lower): marks on bottom of ax_mid and top of ax_bot
slash_marks(ax_mid, 'bottom')
slash_marks(ax_bot, 'top')

# ── Tick locators ─────────────────────────────────────────────────────────────
ax_bot.yaxis.set_major_locator(ticker.FixedLocator([0,15,50,65,100,150,165]))
ax_mid.yaxis.set_major_locator(ticker.FixedLocator([690,700]))
ax_top.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_mid.tick_params(axis='x', which='minor', bottom=False, top=False)
ax_top.tick_params(axis='x', which='minor', bottom=False, top=False)

# ── Axis labels ───────────────────────────────────────────────────────────────
fig.text(0.02, 0.5, r'$T$-count $n$', va='center', rotation='vertical', fontsize=12)
ax_bot.set_xlabel(r'Number of logical qubits $N$', fontsize=12)

# ── Distance legend (colour swatches, placed in top-right of ax_bot) ──────────
# We build it as a second legend on ax_bot using circular patch proxies.
dist_handles = []
for d in all_d:
    patch = mlines.Line2D([], [], color=d_to_color[d], marker='o',
                          linestyle='None', markersize=8,
                          markeredgecolor='black', markeredgewidth=0.6,
                          label=fr'$d = {d}$')
    dist_handles.append(patch)

dist_legend = ax_bot.legend(
    handles=dist_handles,
    title=r'\textbf{Distance}',
    title_fontsize=10,
    loc='upper left',
    fontsize=9,
    framealpha=0.92,
    edgecolor='white',
    handlelength=0.8,
    borderpad=0.6,
)
ax_bot.add_artist(dist_legend)  # keep it when we add the second legend

# ── Main legend (shapes + regions) ───────────────────────────────────────────
proxy_SAT = mlines.Line2D([], [], color='grey', marker='D', linestyle='None',
                           markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                           label=r'SAT decoder')
proxy_can = mlines.Line2D([], [], color='grey', marker='^', linestyle='None',
                           markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                           label=r'Canonical family')
proxy_CCZ = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',
                           markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                           label=r'CCZ')
proxy_15 = mlines.Line2D([], [], color=d_to_color[3], marker='s', linestyle='None',
                           markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                           label=r'15-to-1 [Bravyi2005]')
proxy_49 = mlines.Line2D([], [], color=d_to_color[5], marker='s', linestyle='None',
                           markersize=7, markeredgecolor='black', markeredgewidth=0.7,
                           label=r'49-to-1 [Bravyi2012]')
patch_n = mpatches.Patch(facecolor=d_to_color[3], alpha=.3, hatch='//',
                         edgecolor='tomato', label=r'$n \leq 48: d\leq 3$ [Bravyi2012]')
patch_N = mpatches.Patch(facecolor=d_to_color[3], alpha=0.3, hatch=r'\\\\',
                         edgecolor='tomato', label=r'$N \leq 7: d\leq 3$')

ax_bot.legend(
    handles=[
        Patch(fill=False, edgecolor='none', linewidth=0, label=r'\textbf{This work}'),
        proxy_SAT,
        proxy_can,
        proxy_CCZ,
        patch_N,
        Patch(fill=False, edgecolor='none', linewidth=0, label=r'\textbf{Previous work}'),
        proxy_15,
        proxy_49,
        patch_n,
    ],
    loc='upper right',
    fontsize=10,
    framealpha=0.92,
    edgecolor='grey',
    handlelength=1.6,
)

fig.savefig('./figure.pdf', dpi=500, bbox_inches='tight')
print("Saved.")
