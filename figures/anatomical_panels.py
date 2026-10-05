#!/usr/local/bin/python3.12
"""Anatomical figure panels: top three imaging features per class shaded by mean absolute SHAP."""
import os
os.environ['DISPLAY'] = ''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy import ndimage
from nilearn import datasets, plotting
from nilearn.image import resample_to_img
from pathlib import Path

import matplotlib.font_manager as _fm
_fb = Path(__file__).resolve().parent / 'fonts' / 'Helvetica-Bold.ttf'
if _fb.exists(): _fm.fontManager.addfont(str(_fb))
assert any(f.name == 'Helvetica' and int(f.weight) >= 700 for f in _fm.fontManager.ttflist), 'Helvetica Bold not registered; labels would render regular'
matplotlib.rcParams.update({'font.family': 'Helvetica', 'font.size': 9})

OUT = Path("panels")
OUT.mkdir(exist_ok=True)

FIG_W = 51.5 / 25.4
FIG_H = 30 / 25.4
TITLE_PT = 7
LABEL_PT = 6.5
LEGEND_TITLE_PT = 6
LEGEND_TICK_PT = 6
LEADER_LW = 1.0
LABEL_BOX = dict(boxstyle='round,pad=0.18', fc='white', ec='#404040', lw=0.4, alpha=0.97)
LABEL_SLOTS = [0.66, 0.43, 0.20]
LABEL_X = 0.595
LABEL_COL_MM = 51.5 * (1 - LABEL_X) - 1.2
TEXT_COLOR = '#262626'

def tif(fig, name):
    fig.savefig(OUT / f"{name}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  Saved {name}.tif")

_long = pd.read_csv("shap/shap_values_long.csv")
shap = (_long.groupby(["outcome_group", "fs", "model", "outcome_label", "feature"])["shap_value"]
        .apply(lambda s: s.abs().mean()).reset_index(name="mean_abs_shap"))

WAB_VMAX = float(open(OUT / "wab_max_shap.txt").read().strip())
NCT_VMAX = float(open(OUT / "nct_max_shap.txt").read().strip())

TASKS = {
    "wab": {"outcome": "aphasia_resolution", "fs": "FS4", "model": "rf",
            "title": "Aphasia Resolution (FS4 RF)", "vmax": WAB_VMAX},
    "nct": {"outcome": "discourse_content",  "fs": "FS4", "model": "svr",
            "title": "Discourse Content (FS4 SVR)", "vmax": NCT_VMAX},
}
INFERNO_R = matplotlib.colormaps["inferno_r"]

def shap_to_hex(val, vmax):
    return matplotlib.colors.rgb2hex(INFERNO_R(Normalize(0, vmax)(val))[:3])

def get_features(task_key, modality):
    t = TASKS[task_key]
    d = shap[(shap.outcome_label == t["outcome"]) & (shap.fs == t["fs"]) & (shap.model == t["model"])].copy()
    if modality == "tract":
        d = d[d.feature.str.endswith("_prob")]
    elif modality == "network":
        d = d[d.feature.str.startswith("pair_")]
    return d.nlargest(3, "mean_abs_shap")[["feature", "mean_abs_shap"]].reset_index(drop=True)

def ggseg_tick_labels(vmax):
    return ["0.0", f"{round(vmax / 2, 2):.2f}", f"{vmax:.1f}"]

def add_colorbar(fig, vmax):
    w_in, h_in = 0.2 / 2.54, 1.3 / 2.54
    cax = fig.add_axes([0.845, 0.36, w_in / FIG_W, h_in / FIG_H])
    sm = cm.ScalarMappable(cmap=INFERNO_R, norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.ax.set_yticklabels(ggseg_tick_labels(vmax), fontsize=LEGEND_TICK_PT, color=TEXT_COLOR)
    cb.ax.tick_params(length=2, width=0.5, color='black', direction='in', pad=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("black")
    cax.set_title("Mean\nabsolute\nSHAP", fontsize=LEGEND_TITLE_PT, color=TEXT_COLOR, pad=4, loc='center', linespacing=1.0)

def add_title(fig, text):
    fig.text(0.5, 0.975, text, ha='center', va='top', fontsize=TITLE_PT, fontweight='bold', color='black')

def _proj(mni, dm):
    x, y, z = mni
    if dm == 'z':   return x, y
    elif dm == 'y': return x, z
    else:           return y, z

def mask_centroid_mni(mask_3d, affine):
    return tuple(nib.affines.apply_affine(affine, ndimage.center_of_mass(mask_3d)))

def _fits(fig, text):
    t = fig.text(0, 0, text, fontsize=LABEL_PT, fontweight='bold', fontfamily='Helvetica'); r = fig.canvas.get_renderer()
    w_mm = t.get_window_extent(renderer=r).width / fig.dpi * 25.4; t.remove(); return w_mm <= LABEL_COL_MM
def leader_label(inner, label, xy, slot, color):
    fig = inner.figure
    if not _fits(fig, label) and '–' in label: label = label.replace('–', '–\n', 1)
    inner.annotate(label, xy=xy, xytext=(LABEL_X, LABEL_SLOTS[slot]), textcoords='figure fraction', fontsize=LABEL_PT, fontweight='bold', color=TEXT_COLOR, linespacing=1.05,
                   fontfamily='Helvetica', ha='left', va='center',
                   arrowprops=dict(arrowstyle='-', color=color, lw=LEADER_LW, shrinkA=0, shrinkB=1),
                   bbox=LABEL_BOX, zorder=9999, annotation_clip=False)

def view_axes(fig):
    return [fig.add_axes([0.0, 0.02, 0.575, 0.80])]

def clean(disp, ax):
    for glass_ax in disp.axes.values():
        for txt in glass_ax.ax.texts:
            if txt.get_text() in ('L', 'R', 'A', 'P', 'S', 'I'):
                txt.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

JHU_TRACTS = {
    'SLF_L_prob':  (14, 'L. SLF'),
    'SLFt_L_prob': (18, 'L. SLFt'),
    'IFOF_L_prob': (10, 'L. IFOF'),
    'ILF_L_prob':  (12, 'L. ILF'),
    'UF_L_prob':   (16, 'L. UF'),
}
PROB_THR = 10

def make_tract_figure(task_key):
    top3 = get_features(task_key, "tract")
    t = TASKS[task_key]; vmax = t["vmax"]
    atlas_img = nib.load("JHU-ICBM-tracts-prob-1mm.nii.gz")
    atlas_data = atlas_img.get_fdata(); affine = atlas_img.affine
    try:
        brain_mask = resample_to_img(datasets.load_mni152_brain_mask(resolution=1),
                                     nib.Nifti1Image(atlas_data[:, :, :, 0], affine), interpolation='nearest').get_fdata()
    except Exception:
        brain_mask = np.ones(atlas_data.shape[:3])
    tracts = []
    for rank, (_, row) in enumerate(top3.iterrows()):
        feat = row.feature
        if feat not in JHU_TRACTS:
            continue
        vol_idx, nice = JHU_TRACTS[feat]
        mask = (atlas_data[:, :, :, vol_idx] >= PROB_THR) & (brain_mask > 0)
        nii = nib.Nifti1Image(mask.astype(float), affine)
        tracts.append(dict(nii=nii, mni=mask_centroid_mni(mask, affine), hex=shap_to_hex(row.mean_abs_shap, vmax),
                           label=f"{rank + 1}. {nice}", sv=row.mean_abs_shap, rank=rank + 1))
    if not tracts:
        print(f"  No tract data for {task_key}"); return
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    for ax, dm in zip(view_axes(fig), ['l']):
        disp = plotting.plot_glass_brain(None, display_mode=dm, axes=ax, colorbar=False, annotate=False, black_bg=False)
        for tr in tracts:
            try:
                disp.add_contours(tr['nii'], levels=[0.5], colors=[tr['hex']], linewidths=[1.2], filled=True)
                disp.add_contours(tr['nii'], levels=[0.5], colors=['black'], linewidths=[0.2])
            except Exception:
                pass
        for glass_ax in disp.axes.values():
            inner = glass_ax.ax
            for idx, tr in enumerate(tracts):
                px, py = _proj(tr['mni'], dm)
                leader_label(inner, tr['label'], (px, py), idx, tr['hex'])
        clean(disp, ax)
    add_title(fig, t['title'])
    tif(fig, f"top3_tracts_{task_key}")

FS86_NODES = {
    'POP':   ((-49.4, 11.6, 16.6), 'L. IFG Oper.'),
    'PTR':   ((-47.6, 25.1, 10.9), 'L. IFG Tri.'),
    'STG':   ((-54.0, -15.1,  0.1), 'L. STG'),
    'MTG':   ((-60.6, -16.0, -18.5), 'L. MTG'),
    'SMG':   ((-56.4, -35.1, 36.3), 'L. SMG'),
    'IPG':   ((-44.3, -60.9, 42.4), 'L. Inf. Par.'),
    'ITG':   ((-46.5, -10.4, -34.6), 'L. ITG'),
    'IN':    ((-34.5, 11.6, -2.5), 'L. Insula'),
    'PrCG':  ((-37.6, -10.5, 45.1), 'L. PreCG'),
    'SFG':   ((-13.0,  9.3, 60.1), 'L. SFG'),
    'FG':    ((-33.7, -46.5, -22.4), 'L. Fusiform'),
    'L.TH':  ((-10.0, -18.0,  7.0), 'L. Thalamus'),
    'L.PU':  ((-24.0,  3.0, -3.0), 'L. Putamen'),
    'R.POP': (( 49.6, 16.0, 15.0), 'R. IFG Oper.'),
    'R.SMG': (( 57.5, -30.0, 32.9), 'R. SMG'),
    'R.STG': (( 55.6, -13.4,  2.1), 'R. STG'),
}
SHORT_MAP = {
    "POP": "POP", "PTR": "PTR", "STG": "STG", "MTG": "MTG", "SMG": "SMG",
    "IPG": "IPG", "ITG": "ITG", "IN": "IN", "PrCG": "PrCG", "SFG": "SFG", "FG": "FG",
    "L.POP": "POP", "L.PTR": "PTR", "L.STG": "STG", "L.MTG": "MTG", "L.SMG": "SMG",
    "L.TH": "L.TH", "L.PU": "L.PU",
    "R.POP": "R.POP", "R.SMG": "R.SMG", "R.STG": "R.STG",
}

def parse_pair(feat):
    parts = feat.replace("pair_", "").split("-")
    if len(parts) != 2: return None, None
    ka = SHORT_MAP.get(parts[0], parts[0]); kb = SHORT_MAP.get(parts[1], parts[1])
    if ka in FS86_NODES and kb in FS86_NODES: return ka, kb
    return None, None

def make_network_figure(task_key):
    top3 = get_features(task_key, "network")
    t = TASKS[task_key]; vmax = t["vmax"]
    top3_pairs = top3[top3.feature.str.startswith("pair_")]
    if len(top3_pairs) == 0:
        print(f"  No network pair data for {task_key}"); return
    edges = []; all_node_keys = set()
    for _, row in top3_pairs.iterrows():
        ka, kb = parse_pair(row.feature)
        if ka and kb:
            n1, n2 = FS86_NODES[ka][1], FS86_NODES[kb][1]
            edges.append(dict(ka=ka, kb=kb, sv=row.mean_abs_shap, hex=shap_to_hex(row.mean_abs_shap, vmax),
                              label=f"{len(edges) + 1}. {n1}–{n2}", lw=0.9 + 2.2 * (row.mean_abs_shap / vmax)))
            all_node_keys.update([ka, kb])
    if not edges:
        print(f"  No valid edges for {task_key}"); return
    node_list = sorted(all_node_keys); coords = np.array([FS86_NODES[k][0] for k in node_list]); n = len(coords)
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    for ax, dm in zip(view_axes(fig), ['l']):
        disp = plotting.plot_connectome(np.zeros((n, n)), coords, node_color=['#888888'] * n, node_size=[10] * n,
                                        display_mode=dm, axes=ax, colorbar=False, annotate=False, black_bg=False, edge_threshold='100%')
        for glass_ax in disp.axes.values():
            inner = glass_ax.ax
            for idx, e in enumerate(edges):
                p1 = _proj(FS86_NODES[e['ka']][0], dm); p2 = _proj(FS86_NODES[e['kb']][0], dm)
                inner.plot([p1[0], p2[0]], [p1[1], p2[1]], color=e['hex'], linewidth=e['lw'], alpha=0.85, zorder=2, solid_capstyle='round')
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                leader_label(inner, e['label'], (mx, my), idx, e['hex'])
        clean(disp, ax)
    add_title(fig, t['title'])
    tif(fig, f"top3_network_{task_key}")

def make_cortical_figure(task_key):
    t = TASKS[task_key]; vmax = t["vmax"]
    img = plt.imread(OUT / f"top3_cortical_{task_key}_brain.png"); h, w = img.shape[:2]
    cen = pd.read_csv(OUT / f"top3_cortical_{task_key}_centroids.csv").sort_values("rank")
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='white')
    ax = view_axes(fig)[0]
    ax.imshow(img, extent=[0, w, h, 0], interpolation='lanczos'); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    for idx, r in enumerate(cen.itertuples()):
        leader_label(ax, f"{int(r.rank)}. {r.nice}", (r.px, r.py), idx, shap_to_hex(r.shap, vmax))
    add_title(fig, t['title'])
    tif(fig, f"top3_cortical_{task_key}")

def make_row_colorbar(task_key):
    t = TASKS[task_key]; vmax = t["vmax"]
    fig = plt.figure(figsize=(12 / 25.4, FIG_H), facecolor='white')
    w_in, h_in = 0.18 / 2.54, 1.25 / 2.54
    cax = fig.add_axes([0.06, 0.12, w_in / (12 / 25.4), h_in / FIG_H])
    sm = cm.ScalarMappable(cmap=INFERNO_R, norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.ax.set_yticklabels(ggseg_tick_labels(vmax), fontsize=LEGEND_TICK_PT, color=TEXT_COLOR)
    cb.ax.tick_params(length=2, width=0.5, color='black', direction='in', pad=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("black")
    cax.set_title("Mean\nabsolute\nSHAP", fontsize=LEGEND_TITLE_PT, color=TEXT_COLOR, pad=4, loc='left', linespacing=1.0)
    tif(fig, f"colorbar_{task_key}")

TV_W = 170 / 25.4; TV_H = 40 / 25.4
TV_CB_X = 0.918
TV_WRAP_MM = 20

def tv_view_axes(fig):
    slot = (TV_CB_X - 0.004) / 3
    return [fig.add_axes([0.002 + i * slot, 0.015, slot * 0.985, 0.845]) for i in range(3)]

def _width_mm(fig, text):
    t = fig.text(0, 0, text, fontsize=LABEL_PT, fontweight='bold', fontfamily='Helvetica'); r = fig.canvas.get_renderer()
    w = t.get_window_extent(renderer=r).width / fig.dpi * 25.4; t.remove(); return w

def tv_label(inner, label, xy, y_slot, color):
    fig = inner.figure
    if '–' in label and _width_mm(fig, label) > TV_WRAP_MM: label = label.replace('–', '–\n', 1)
    xlim = inner.get_xlim(); xspan = xlim[1] - xlim[0]
    inner.annotate(label, xy=xy, xytext=(xlim[1] - xspan * 0.02, y_slot), fontsize=LABEL_PT, fontweight='bold', color=TEXT_COLOR,
                   fontfamily='Helvetica', ha='right', va='center', linespacing=1.05,
                   arrowprops=dict(arrowstyle='-', color=color, lw=LEADER_LW, shrinkA=0, shrinkB=1),
                   bbox=LABEL_BOX, zorder=9999, annotation_clip=False)

def tv_slots(inner, n):
    ylim = inner.get_ylim(); yspan = ylim[1] - ylim[0]
    return np.linspace(ylim[1] - yspan * 0.14, ylim[0] + yspan * 0.14, n)

def tv_colorbar(fig, vmax):
    w_in, h_in = 0.18 / 2.54, 1.3 / 2.54
    cax = fig.add_axes([TV_CB_X + 0.006, 0.12, w_in / TV_W, h_in / TV_H])
    sm = cm.ScalarMappable(cmap=INFERNO_R, norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.ax.set_yticklabels(ggseg_tick_labels(vmax), fontsize=LEGEND_TICK_PT, color=TEXT_COLOR)
    cb.ax.tick_params(length=2, width=0.5, color='black', direction='in', pad=2)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("black")
    cax.set_title("Mean\nabsolute\nSHAP", fontsize=LEGEND_TITLE_PT, color=TEXT_COLOR, pad=4, loc='left', linespacing=1.0)

def tv_title(fig, text):
    fig.text(TV_CB_X / 2, 0.985, text, ha='center', va='top', fontsize=TITLE_PT, fontweight='bold', color='black')

def tv_save(fig, name):
    fig.savefig(OUT / f"{name}.tif", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig); print(f"  Saved {name}.tif")

def make_tract_figure3(task_key):
    top3 = get_features(task_key, "tract"); t = TASKS[task_key]; vmax = t["vmax"]
    atlas_img = nib.load("JHU-ICBM-tracts-prob-1mm.nii.gz"); atlas_data = atlas_img.get_fdata(); affine = atlas_img.affine
    try:
        brain_mask = resample_to_img(datasets.load_mni152_brain_mask(resolution=1),
                                     nib.Nifti1Image(atlas_data[:, :, :, 0], affine), interpolation='nearest').get_fdata()
    except Exception:
        brain_mask = np.ones(atlas_data.shape[:3])
    tracts = []
    for rank, (_, row) in enumerate(top3.iterrows()):
        if row.feature not in JHU_TRACTS: continue
        vol_idx, nice = JHU_TRACTS[row.feature]
        mask = (atlas_data[:, :, :, vol_idx] >= PROB_THR) & (brain_mask > 0)
        tracts.append(dict(nii=nib.Nifti1Image(mask.astype(float), affine), mni=mask_centroid_mni(mask, affine),
                           hex=shap_to_hex(row.mean_abs_shap, vmax), label=f"{rank + 1}. {nice}"))
    fig = plt.figure(figsize=(TV_W, TV_H), facecolor='white')
    for ax, dm in zip(tv_view_axes(fig), ['x', 'y', 'z']):
        disp = plotting.plot_glass_brain(None, display_mode=dm, axes=ax, colorbar=False, annotate=False, black_bg=False)
        for tr in tracts:
            try:
                disp.add_contours(tr['nii'], levels=[0.5], colors=[tr['hex']], linewidths=[1.2], filled=True)
                disp.add_contours(tr['nii'], levels=[0.5], colors=['black'], linewidths=[0.2])
            except Exception:
                pass
        for glass_ax in disp.axes.values():
            inner = glass_ax.ax; ys = tv_slots(inner, len(tracts))
            for idx, tr in enumerate(tracts):
                tv_label(inner, tr['label'], _proj(tr['mni'], dm), ys[idx], tr['hex'])
        clean(disp, ax)
    tv_colorbar(fig, vmax); tv_title(fig, t['title'])
    tv_save(fig, f"top3_tracts3_{task_key}")

def make_network_figure3(task_key):
    top3 = get_features(task_key, "network"); t = TASKS[task_key]; vmax = t["vmax"]
    edges = []; keys = set()
    for _, row in top3[top3.feature.str.startswith("pair_")].iterrows():
        ka, kb = parse_pair(row.feature)
        if ka and kb:
            edges.append(dict(ka=ka, kb=kb, hex=shap_to_hex(row.mean_abs_shap, vmax), lw=0.9 + 2.2 * (row.mean_abs_shap / vmax),
                              label=f"{len(edges) + 1}. {FS86_NODES[ka][1]}–{FS86_NODES[kb][1]}"))
            keys.update([ka, kb])
    node_list = sorted(keys); coords = np.array([FS86_NODES[k][0] for k in node_list]); n = len(coords)
    fig = plt.figure(figsize=(TV_W, TV_H), facecolor='white')
    for ax, dm in zip(tv_view_axes(fig), ['x', 'y', 'z']):
        disp = plotting.plot_connectome(np.zeros((n, n)), coords, node_color=['#888888'] * n, node_size=[10] * n,
                                        display_mode=dm, axes=ax, colorbar=False, annotate=False, black_bg=False, edge_threshold='100%')
        for glass_ax in disp.axes.values():
            inner = glass_ax.ax; ys = tv_slots(inner, len(edges))
            for idx, e in enumerate(edges):
                p1 = _proj(FS86_NODES[e['ka']][0], dm); p2 = _proj(FS86_NODES[e['kb']][0], dm)
                inner.plot([p1[0], p2[0]], [p1[1], p2[1]], color=e['hex'], linewidth=e['lw'], alpha=0.85, zorder=2, solid_capstyle='round')
                tv_label(inner, e['label'], ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), ys[idx], e['hex'])
        clean(disp, ax)
    tv_colorbar(fig, vmax); tv_title(fig, t['title'])
    tv_save(fig, f"top3_network3_{task_key}")

import sys
if '--three-view-only' in sys.argv:
    for task in ["wab", "nct"]:
        print(f"--- {TASKS[task]['title']} (three views) ---"); make_tract_figure3(task); make_network_figure3(task)
    sys.exit(0)

print("=== Top-3 labeled brain figures (v7: single lateral view, final size) ===\n")
for task in ["wab", "nct"]:
    print(f"--- {TASKS[task]['title']} ---")
    make_cortical_figure(task)
    make_tract_figure(task)
    make_network_figure(task)
    make_tract_figure3(task)
    make_network_figure3(task)
    make_row_colorbar(task)
    print()
print("Done!")
