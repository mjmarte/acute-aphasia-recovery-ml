#!/usr/bin/env python3
"""Assembles the anatomical figure (Figure 4) in two rows, one per outcome: cortical parcels, white matter tracts and
structural connections. The tract and connection panels come from anatomical_panels.py; the cortical panels are drawn
here from the ggseg render and centroids written by anatomical_cortical.R. The figure prints 170 mm wide."""
from pathlib import Path
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.font_manager as _fm
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / 'manuscript' / 'figures'; F = HERE / 'panels'
_fb = HERE / 'fonts' / 'Helvetica-Bold.ttf'
if _fb.exists(): _fm.fontManager.addfont(str(_fb))
assert any(f.name == 'Helvetica' and int(f.weight) >= 700 for f in _fm.fontManager.ttflist), 'Helvetica Bold not registered'
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 9})
DPI = 600; MM = DPI / 25.4; PRINT_W_MM = 170; ROW_GAP = round(3 * MM); PAD = round(0.5 * MM); LETTER_PT = 9
# type and line sizes of anatomical_panels.py
TITLE_PT = 7; LABEL_PT = 6.5; LEGEND_TITLE_PT = 6; LEGEND_TICK_PT = 6; LEADER_LW = 1.0; TEXT_COLOR = '#262626'
LABEL_BOX = dict(boxstyle='round,pad=0.18', fc='white', ec='#404040', lw=0.4, alpha=0.97)
INFERNO_R = matplotlib.colormaps['inferno_r']
TASKS = {'wab': ('Aphasia Resolution (FS4 RF)', float((F / 'wab_max_shap.txt').read_text().strip())),
         'nct': ('Discourse Content (FS4 SVR)', float((F / 'nct_max_shap.txt').read_text().strip()))}
CW, CH = 80 / 25.4, 40 / 25.4            # cortical panel, in
C_CB_X = 0.862                           # left edge of the colourbar strip (figure fraction)
C_LABEL_X = 0.575; C_SLOTS = [0.68, 0.45, 0.22]

def cortical(task):
    title, vmax = TASKS[task]
    img = plt.imread(F / f'top3_cortical_{task}_brain.png'); h, w = img.shape[:2]
    cen = pd.read_csv(F / f'top3_cortical_{task}_centroids.csv').sort_values('rank')
    fig = plt.figure(figsize=(CW, CH), facecolor='white')
    ax = fig.add_axes([0.005, 0.02, 0.555, 0.84])
    ax.imshow(img, extent=[0, w, h, 0], interpolation='lanczos'); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    for idx, r in enumerate(cen.itertuples()):
        color = matplotlib.colors.rgb2hex(INFERNO_R(Normalize(0, vmax)(r.shap))[:3])
        ax.annotate(f'{int(r.rank)}. {r.nice}', xy=(r.px, r.py), xytext=(C_LABEL_X, C_SLOTS[idx]), textcoords='figure fraction',
                    fontsize=LABEL_PT, fontweight='bold', color=TEXT_COLOR, linespacing=1.05, fontfamily='Helvetica', ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color=color, lw=LEADER_LW, shrinkA=0, shrinkB=1), bbox=LABEL_BOX, zorder=9999, annotation_clip=False)
    w_in, h_in = 0.18 / 2.54, 1.3 / 2.54
    cax = fig.add_axes([C_CB_X + 0.012, 0.12, w_in / CW, h_in / CH])
    sm = cm.ScalarMappable(cmap=INFERNO_R, norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.ax.set_yticklabels(['0.0', f'{round(vmax / 2, 2):.2f}', f'{vmax:.1f}'], fontsize=LEGEND_TICK_PT, color=TEXT_COLOR)
    cb.ax.tick_params(length=2, width=0.5, color='black', direction='in', pad=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor('black')
    cax.set_title('Mean\nabsolute\nSHAP', fontsize=LEGEND_TITLE_PT, color=TEXT_COLOR, pad=4, loc='left', linespacing=1.0)
    fig.text(C_CB_X / 2, 0.985, title, ha='center', va='top', fontsize=TITLE_PT, fontweight='bold', color='black')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=DPI, facecolor='white'); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB')

def font(size_px):
    for cand in [HERE / 'fonts' / 'Helvetica-Bold.ttf', Path('/System/Library/Fonts/Helvetica.ttc')]:
        try: return ImageFont.truetype(str(cand), size_px, index=1 if str(cand).endswith('.ttc') else 0)
        except Exception: continue
    raise SystemExit('no bold font for panel letters')
LETTER = font(round(LETTER_PT / 72 * DPI))
def load(name):
    f = F / name
    if not f.exists(): raise SystemExit(f'missing {f} (run anatomical_panels.py)')
    return Image.open(f).convert('RGB')
def letter(im, ch):
    ImageDraw.Draw(im).text((round(0.4 * MM), round(0.1 * MM)), ch, fill='black', font=LETTER); return im

rows = []
for task, letters in (('wab', 'ABC'), ('nct', 'DEF')):
    panels = [letter(cortical(task), letters[0]), letter(load(f'top3_tracts3_{task}.tif'), letters[1]), letter(load(f'top3_network3_{task}.tif'), letters[2])]
    hgt = max(p.height for p in panels); row = Image.new('RGB', (sum(p.width for p in panels), hgt), 'white'); x = 0
    for p in panels: row.paste(p, (x, 0)); x += p.width
    rows.append(row)
W = max(r.width for r in rows); H = sum(r.height for r in rows) + ROW_GAP * (len(rows) - 1) + 2 * PAD
fig = Image.new('RGB', (W, H), 'white'); y = PAD
for r in rows: fig.paste(r, (0, y)); y += r.height + ROW_GAP
out_dpi = W / (PRINT_W_MM / 25.4)
fig.save(OUT / 'fig_brain.png', dpi=(out_dpi, out_dpi)); fig.save(OUT / 'fig_brain.tif', dpi=(out_dpi, out_dpi), compression='tiff_lzw')
print('composed', OUT / 'fig_brain.tif', fig.size, f'= {PRINT_W_MM} x {H / out_dpi * 25.4:.1f} mm at {out_dpi:.0f} dpi; labels print at {LABEL_PT * PRINT_W_MM / (W / MM):.1f} pt')
