"""
app.py  —  Nematode Community Explorer  (Streamlit)
====================================================
CHANGES in this version:
  • Phylum filter: multiselect — choose which phyla to include
    (auto-populated from data, works on any phylum column)
  • Removed: is-nematode flag column + filter checkbox

INSTALL:
    pip install streamlit pandas numpy matplotlib scipy scikit-learn openpyxl

RUN:
    streamlit run app.py

DEPLOY (free):
    1. Push this file + requirements.txt to GitHub
    2. Go to https://streamlit.io/cloud → connect repo → deploy → share URL
"""

import re
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, mannwhitneyu
from scipy.spatial.distance import braycurtis, pdist, squareform
from sklearn.manifold import MDS

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nematode Community Explorer",
    page_icon="🪱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def shannon_index(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def compute_relative_composition(df, taxon_col, reads_col):
    rel_dict = {}
    for (site, tr), sub in df.groupby(["_site", "_treatment"]):
        grp   = sub.groupby(taxon_col)[reads_col].sum()
        total = grp.sum()
        rel_dict[(site, tr)] = grp / total * 100.0 if total > 0 else grp * 0.0
    return rel_dict


def make_color_map(taxa):
    c1 = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    c2 = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))
    c3 = plt.get_cmap("tab20c")(np.linspace(0, 1, 20))
    all_c = np.vstack([c1, c2, c3])
    return {t: all_c[i % len(all_c)] for i, t in enumerate(sorted(taxa))}


def bc_sim(va, vb):
    a, b = np.asarray(va, float), np.asarray(vb, float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    return (1 - braycurtis(a, b)) * 100.0


def sig_stars(p):
    if np.isnan(p):  return "ns"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


def build_community_matrix(df, taxon_col, reads_col, combine_reps=False):
    records, labels = [], []
    if combine_reps:
        for (site, tr), sub in df.groupby(["_site", "_treatment"]):
            grp   = sub.groupby(taxon_col)[reads_col].sum()
            total = grp.sum()
            rel   = grp / total * 100.0 if total > 0 else grp * 0.0
            records.append(rel)
            labels.append(f"{site}{tr}")
    else:
        for (site, tr, rep), sub in df.groupby(["_site", "_treatment", "_rep"]):
            grp   = sub.groupby(taxon_col)[reads_col].sum()
            total = grp.sum()
            rel   = grp / total * 100.0 if total > 0 else grp * 0.0
            records.append(rel)
            labels.append(f"{site}{tr}{rep}")
    return pd.DataFrame(records).fillna(0), labels


def permanova(dist_matrix, grouping, n_perm=999):
    groups = np.asarray(grouping)
    n      = len(groups)
    k      = len(np.unique(groups))

    def _f(d, g):
        ss_tot = np.sum(d**2) / n
        ss_w   = 0.0
        for lbl in np.unique(g):
            idx = np.where(g == lbl)[0]
            ni  = len(idx)
            if ni < 2: continue
            ss_w += np.sum(d[np.ix_(idx, idx)]**2) / ni
        dfa = k - 1; dfw = n - k
        if dfw == 0 or ss_w == 0: return np.nan
        return ((ss_tot - ss_w) / dfa) / (ss_w / dfw)

    f_obs = _f(dist_matrix, groups)
    if np.isnan(f_obs): return f_obs, np.nan
    rng   = np.random.default_rng(42)
    count = sum(1 for _ in range(n_perm)
                if _f(dist_matrix, rng.permutation(groups)) >= f_obs)
    return f_obs, (count + 1) / (n_perm + 1)


def run_mds(dist_matrix):
    mds = MDS(n_components=2, dissimilarity="precomputed",
              max_iter=500, n_init=10, random_state=42,
              normalized_stress="auto")
    coords = mds.fit_transform(dist_matrix)
    return coords, mds.stress_


def confidence_ellipse(xs, ys, ax, n_std=2.0, **kw):
    from matplotlib.patches import Ellipse
    if len(xs) < 3: return
    cov  = np.cov(xs, ys)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy=(np.mean(xs), np.mean(ys)),
                         width=w, height=h, angle=angle, **kw))


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def parse_labels(df, regex, tr_a_chars, tr_d_chars):
    pattern = re.compile(regex)
    sites, treatments, reps = [], [], []
    for lbl in df["_label"].astype(str):
        m = pattern.match(lbl)
        if not m:
            raise ValueError(f"Label '{lbl}' does not match regex:\n{regex}")
        sites.append(int(m.group("site")))
        treatments.append(m.group("treatment"))
        reps.append(int(m.group("rep")))
    df = df.copy()
    df["_site"]      = sites
    df["_treatment"] = treatments
    df["_rep"]       = reps
    a_set = set(c.strip() for c in tr_a_chars.split(","))
    d_set = set(c.strip() for c in tr_d_chars.split(","))
    df["_treatment"] = df["_treatment"].map(
        lambda t: "A" if t in a_set else ("D" if t in d_set else t.upper()))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🪱 Community Explorer")
st.sidebar.markdown("---")

# ── 1. Upload ─────────────────────────────────────────────────────────────────
st.sidebar.header("① Upload data")
uploaded = st.sidebar.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])

@st.cache_data
def load_file(file_bytes, file_name):
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded is None:
    st.title("🪱 Nematode Community Explorer")
    st.info("👈  Upload your Excel or CSV file in the sidebar to get started.")
    st.stop()

df_raw = load_file(uploaded.read(), uploaded.name)
st.sidebar.success(f"Loaded: {len(df_raw):,} rows × {len(df_raw.columns)} cols")

all_cols  = list(df_raw.columns)
none_cols = ["(none)"] + all_cols
num_cols  = df_raw.select_dtypes(include=[np.number]).columns.tolist()
obj_cols  = df_raw.select_dtypes(include=["object"]).columns.tolist()

def first_match(candidates, pool, fallback="(none)"):
    return next((c for c in candidates if c in pool), fallback)

# ── 2. Column mapping ─────────────────────────────────────────────────────────
st.sidebar.header("② Column mapping")
reads_col  = st.sidebar.selectbox("Read counts *", none_cols,
    index=none_cols.index(first_match(
        ["total supporting reads", "reads", "count"], all_cols)))
label_col  = st.sidebar.selectbox("Sample label *", none_cols,
    index=none_cols.index(first_match(
        ["sites", "label", "sample", "sample_id"], all_cols)))
taxon_col  = st.sidebar.selectbox("Species / taxon *", none_cols,
    index=none_cols.index(first_match(
        ["blast_species", "species", "taxon"], all_cols)))
phylum_col = st.sidebar.selectbox("Phylum column", none_cols,
    index=none_cols.index(first_match(
        ["tax_phylum", "phylum"], all_cols)))

# ── 3. Label parsing ──────────────────────────────────────────────────────────
st.sidebar.header("③ Label parsing")
regex_val  = st.sidebar.text_input(
    "Regex (named groups: site, treatment, rep)",
    value=r"^(?P<site>[0-9]+)(?P<treatment>[aAdD])(?P<rep>[0-9]+)$")
tr_a_chars = st.sidebar.text_input("Treatment A chars (comma-sep)", value="a,A")
tr_d_chars = st.sidebar.text_input("Treatment D chars (comma-sep)", value="d,D")

# ── 4. Filters ────────────────────────────────────────────────────────────────
st.sidebar.header("④ Filters")

# Phylum filter — dynamic, populated from data
if phylum_col != "(none)" and phylum_col in df_raw.columns:
    available_phyla = sorted(df_raw[phylum_col].dropna().astype(str).unique().tolist())
    selected_phyla  = st.sidebar.multiselect(
        "Include phyla",
        options=available_phyla,
        default=available_phyla,
        help="Only rows matching selected phyla will be used in all plots.")
else:
    selected_phyla = None
    st.sidebar.info("Set a Phylum column in ② to enable phylum filtering.")

sites_input = st.sidebar.text_input(
    "Sites to include (comma-sep, empty = all)", value="")

# ── 5. Plot type ──────────────────────────────────────────────────────────────
st.sidebar.header("⑤ Plot type")
plot_type = st.sidebar.radio("Choose plot", [
    "Stacked bar  (A vs D)",
    "Shannon diversity",
    "Shannon vs environment",
    "NMDS / PCoA ordination",
    "Replicate similarity table",
])

# ─────────────────────────────────────────────────────────────────────────────
# BUILD WORKING DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def build_df(file_bytes, file_name,
             reads_col, label_col, taxon_col, phylum_col,
             selected_phyla, sites_input, regex_val, tr_a_chars, tr_d_chars):
    df = load_file(file_bytes, file_name)

    if reads_col == "(none)" or label_col == "(none)" or taxon_col == "(none)":
        return None, "Set Read counts, Sample label, and Species/taxon columns."

    # reads > 0
    df = df[pd.to_numeric(df[reads_col], errors="coerce").fillna(0) > 0].copy()

    # phylum filter
    if (phylum_col != "(none)" and phylum_col in df.columns
            and selected_phyla is not None and len(selected_phyla) > 0):
        df = df[df[phylum_col].astype(str).isin(selected_phyla)]

    df["_label"] = df[label_col].astype(str)

    try:
        df = parse_labels(df, regex_val, tr_a_chars, tr_d_chars)
    except Exception as e:
        return None, str(e)

    if sites_input.strip():
        try:
            sites = [int(s.strip()) for s in
                     sites_input.replace(";", ",").split(",") if s.strip()]
            df = df[df["_site"].isin(sites)]
        except:
            return None, "Sites filter: enter comma-separated integers."

    if df.empty:
        return None, "All rows removed by filters."

    return df, None

uploaded.seek(0)
raw_bytes = uploaded.read()

df, err = build_df(
    raw_bytes, uploaded.name,
    reads_col, label_col, taxon_col, phylum_col,
    tuple(selected_phyla) if selected_phyla is not None else None,
    sites_input, regex_val, tr_a_chars, tr_d_chars)

if err:
    st.error(err)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🪱 Nematode Community Explorer")

# show active phylum filter as a badge
phylum_badge = ""
if selected_phyla is not None:
    phylum_badge = f" · Phyla: **{', '.join(selected_phyla)}**"

st.caption(
    f"**{uploaded.name}** — {len(df):,} rows · "
    f"{df['_site'].nunique()} sites · "
    f"{df['_treatment'].nunique()} treatments · "
    f"{df['_rep'].nunique()} replicates"
    + phylum_badge)

# ─────────────────────────────────────────────────────────────────────────────
# TAXON COLUMN CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

tax_candidates = [c for c in [phylum_col, taxon_col] + obj_cols
                  if c != "(none)" and c in df.columns]
tax_candidates = list(dict.fromkeys(tax_candidates))

# ─────────────────────────────────────────────────────────────────────────────
# ① STACKED BAR
# ─────────────────────────────────────────────────────────────────────────────

if plot_type.startswith("Stacked"):
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    comp_col = c1.selectbox("Composition level", tax_candidates,
                             index=tax_candidates.index(taxon_col)
                             if taxon_col in tax_candidates else 0)
    min_pct  = c2.number_input("Min % to show", 0.0, 50.0, 1.0, 0.5)
    max_leg  = c3.number_input("Max taxa in legend", 1, 60, 20, 1)
    show_pct = c4.checkbox("% labels in bars", value=True)
    show_bc  = st.checkbox("Show Bray-Curtis similarity + significance below bars",
                            value=True)

    if comp_col not in df.columns:
        st.error(f"Column '{comp_col}' not found."); st.stop()

    df_s      = df.dropna(subset=[comp_col])
    rel_dict  = compute_relative_composition(df_s, comp_col, reads_col)
    sites     = sorted({s for (s, _) in rel_dict})
    all_taxa  = sorted(set().union(*[r.index for r in rel_dict.values()]))
    color_map = make_color_map(all_taxa)

    n_sites = len(sites)
    fig_w   = max(8, n_sites * 1.4 + 4)

    if show_bc:
        fig, (ax, axb) = plt.subplots(2, 1, figsize=(fig_w, 7),
                                       gridspec_kw={"height_ratios": [6, 1],
                                                    "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        axb = None

    bar_w   = 0.35
    xs      = np.arange(n_sites)
    bottoms = {(s, tr): 0.0 for s in sites for tr in ["A", "D"]}

    for tax in all_taxa:
        for i, s in enumerate(sites):
            for tr, offset in [("A", -bar_w/2), ("D", bar_w/2)]:
                rel = rel_dict.get((s, tr))
                h   = float(rel.get(tax, 0.0)) if rel is not None else 0.0
                if h < min_pct: continue
                b = bottoms[(s, tr)]
                ax.bar(xs[i]+offset, h, bar_w, bottom=b,
                       color=color_map[tax], edgecolor="white", linewidth=0.3)
                if show_pct and h >= 3.0:
                    ax.text(xs[i]+offset, b+h/2, f"{h:.0f}%",
                            ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
                bottoms[(s, tr)] = b + h

    for i, s in enumerate(sites):
        ax.text(xs[i]-bar_w/2, 101.5, "A", ha="center", va="bottom",
                fontsize=8, color="#2c7bb6", fontweight="bold")
        ax.text(xs[i]+bar_w/2, 101.5, "D", ha="center", va="bottom",
                fontsize=8, color="#d7191c", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Site {s}" for s in sites], fontsize=10)
    ax.set_ylabel("Relative abundance (%)", fontsize=11)
    ax.set_ylim(0, 107)
    ax.set_title(f"Community composition (A vs D) — {comp_col}",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    total_abund = {t: sum(float(r.get(t, 0)) for r in rel_dict.values())
                   for t in all_taxa}
    top_taxa = sorted(all_taxa, key=lambda t: total_abund[t], reverse=True)[:int(max_leg)]
    handles  = [mpatches.Patch(facecolor=color_map[t], edgecolor="white",
                               linewidth=0.5, label=t) for t in top_taxa]
    ax.legend(handles=handles, title=f"Top {len(top_taxa)} taxa",
              title_fontsize=8, fontsize=7,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=True, edgecolor="#ccc")

    if show_bc and axb is not None:
        axb.set_xlim(ax.get_xlim())
        axb.set_ylim(0, 1)
        axb.axis("off")

        for i, s in enumerate(sites):
            rel_a = rel_dict.get((s, "A"))
            rel_d = rel_dict.get((s, "D"))
            if rel_a is None or rel_d is None: continue
            all_sp = sorted(set(rel_a.index) | set(rel_d.index))
            va  = np.array([float(rel_a.get(sp, 0)) for sp in all_sp])
            vd  = np.array([float(rel_d.get(sp, 0)) for sp in all_sp])
            sim = bc_sim(va, vd)

            reps_a = df_s[(df_s["_site"]==s) & (df_s["_treatment"]=="A")]
            reps_d = df_s[(df_s["_site"]==s) & (df_s["_treatment"]=="D")]

            def _h_reps(sub):
                out = []
                for _, rsub in sub.groupby("_rep"):
                    grp = rsub.groupby(comp_col)[reads_col].sum()
                    out.append(shannon_index(grp.values))
                return out

            h_a = _h_reps(reps_a); h_d = _h_reps(reps_d)
            p_val = np.nan
            if len(h_a) >= 2 and len(h_d) >= 2:
                try: _, p_val = mannwhitneyu(h_a, h_d, alternative="two-sided")
                except: pass

            sig    = sig_stars(p_val)
            bc_str = f"BC: {sim:.1f}%" if not np.isnan(sim) else "BC: N/A"
            p_str  = f"  p={p_val:.3f}" if not np.isnan(p_val) else ""
            col    = ("#27ae60" if (not np.isnan(sim) and sim >= 70) else
                      "#f39c12" if (not np.isnan(sim) and sim >= 40) else "#c0392b")

            axb.add_patch(mpatches.FancyBboxPatch(
                (xs[i]-0.42, 0.05), 0.84, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=col, edgecolor="white", alpha=0.85,
                linewidth=1, transform=axb.transData))
            axb.text(xs[i], 0.50, f"{bc_str}  {sig}{p_str}",
                     ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold",
                     transform=axb.transData)

        bc_leg = [mpatches.Patch(color="#27ae60", label="BC ≥ 70%"),
                  mpatches.Patch(color="#f39c12", label="BC 40–70%"),
                  mpatches.Patch(color="#c0392b", label="BC < 40%")]
        axb.legend(handles=bc_leg, loc="lower right", fontsize=7,
                   frameon=True, edgecolor="#ccc", bbox_to_anchor=(1.0, -0.1))
        axb.set_title(
            "A vs D  Bray-Curtis similarity  |  "
            "* p<0.05  ** p<0.01  *** p<0.001  (Mann-Whitney on Shannon H)",
            fontsize=8, color="#444", loc="left", pad=3)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    st.pyplot(fig, use_container_width=False)
    st.download_button("⬇️ Download PNG", fig_to_bytes(fig),
                       file_name="stacked_bar.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# ② SHANNON DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

elif plot_type.startswith("Shannon div"):
    c1, c2 = st.columns([2, 2])
    sh_col   = c1.selectbox("Diversity level", tax_candidates,
                             index=tax_candidates.index(taxon_col)
                             if taxon_col in tax_candidates else 0)
    grouping = c2.radio("Group by", [
        "site × treatment × replicate",
        "site × treatment",
        "treatment only",
        "site only",
    ])

    records = []
    if grouping == "site × treatment × replicate":
        for (site, tr, rep), sub in df.groupby(["_site", "_treatment", "_rep"]):
            grp = sub.groupby(sh_col)[reads_col].sum()
            records.append({"label": f"{site}{tr}{rep}", "treatment": tr,
                             "shannon": shannon_index(grp.values)})
    elif grouping == "site × treatment":
        for (site, tr), sub in df.groupby(["_site", "_treatment"]):
            grp = sub.groupby(sh_col)[reads_col].sum()
            records.append({"label": f"{site}{tr}", "treatment": tr,
                             "shannon": shannon_index(grp.values)})
    elif grouping == "treatment only":
        for tr, sub in df.groupby("_treatment"):
            grp = sub.groupby(sh_col)[reads_col].sum()
            records.append({"label": f"Tr. {tr}", "treatment": tr,
                             "shannon": shannon_index(grp.values)})
    else:
        for site, sub in df.groupby("_site"):
            grp = sub.groupby(sh_col)[reads_col].sum()
            records.append({"label": f"Site {site}", "treatment": "?",
                             "shannon": shannon_index(grp.values)})

    out = pd.DataFrame(records).sort_values("label")
    tr_colors = {"A": "#2c7bb6", "D": "#d7191c", "?": "#888888"}
    x   = np.arange(len(out))
    fig, ax = plt.subplots(figsize=(max(7, len(out)*0.6+2), 5))
    bars = ax.bar(x, out["shannon"],
                  color=[tr_colors.get(t, "#888") for t in out["treatment"]],
                  edgecolor="white", linewidth=0.5, width=0.7, zorder=2)
    for bar, val in zip(bars, out["shannon"]):
        if not np.isnan(val):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(out["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Shannon index (H)", fontsize=11)
    ax.set_title(f"Shannon diversity — {sh_col}  [{grouping}]",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [mpatches.Patch(color="#2c7bb6", label="Treatment A"),
               mpatches.Patch(color="#d7191c", label="Treatment D")]
    ax.legend(handles=handles, fontsize=9, frameon=True, edgecolor="#ccc")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.download_button("⬇️ Download PNG", fig_to_bytes(fig),
                       file_name="shannon.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# ③ SHANNON vs ENV
# ─────────────────────────────────────────────────────────────────────────────

elif plot_type.startswith("Shannon vs"):
    c1, c2 = st.columns([2, 2])
    sh_col  = c1.selectbox("Diversity level", tax_candidates,
                            index=tax_candidates.index(taxon_col)
                            if taxon_col in tax_candidates else 0)
    env_choices = ["(none)"] + num_cols
    c3, c4, c5, c6 = st.columns(4)
    env1 = c3.selectbox("X-axis 1", env_choices,
                         index=1 if len(env_choices) > 1 else 0)
    env2 = c4.selectbox("X-axis 2", env_choices, index=0)
    env3 = c5.selectbox("X-axis 3", env_choices, index=0)
    env4 = c6.selectbox("X-axis 4", env_choices, index=0)

    env_cols = [e for e in [env1, env2, env3, env4] if e != "(none)"]
    if not env_cols:
        st.warning("Select at least one X-axis variable."); st.stop()

    records = []
    for (site, tr), sub in df.groupby(["_site", "_treatment"]):
        grp = sub.groupby(sh_col)[reads_col].sum()
        rec = {"site": site, "treatment": tr,
               "shannon": shannon_index(grp.values)}
        for col in env_cols:
            rec[col] = pd.to_numeric(sub[col], errors="coerce").mean() \
                       if col in sub.columns else np.nan
        records.append(rec)
    out = pd.DataFrame(records)

    n    = len(env_cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    axes = axes[0]
    styles = {"A": {"color": "#2c7bb6", "marker": "o"},
              "D": {"color": "#d7191c", "marker": "s"}}

    for ax, env in zip(axes, env_cols):
        for tr, st_s in styles.items():
            sub = out[out["treatment"] == tr]
            x   = pd.to_numeric(sub[env], errors="coerce").values
            y   = sub["shannon"].values
            if x.size == 0: continue
            ax.scatter(x, y, color=st_s["color"], marker=st_s["marker"],
                       s=120, edgecolors="white", lw=0.8,
                       label=f"Treatment {tr}", zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(str(int(row["site"])),
                            xy=(row[env], row["shannon"]),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=9, fontweight="bold", color=st_s["color"])
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() >= 3 and np.nanstd(x[mask]) > 0:
                xm, ym = x[mask], y[mask]
                m, b   = np.polyfit(xm, ym, 1)
                xl     = np.linspace(xm.min(), xm.max(), 100)
                ax.plot(xl, m*xl+b, color=st_s["color"],
                        ls="--", lw=1.5, alpha=0.7)
                r, p  = pearsonr(xm, ym)
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                off   = 0.22 if tr == "A" else 0.05
                ax.text(0.98, off, f"Tr.{tr}  r={r:.2f}, {p_str}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color=st_s["color"],
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=st_s["color"], lw=1))
        ax.set_xlabel(env, fontsize=10)
        ax.set_ylabel("Shannon H" if env == env_cols[0] else "", fontsize=10)
        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        if env == env_cols[0]:
            ax.legend(fontsize=9, frameon=True, edgecolor="#ccc")

    fig.suptitle(f"Shannon vs environment — {sh_col}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.download_button("⬇️ Download PNG", fig_to_bytes(fig),
                       file_name="shannon_env.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# ④ NMDS / PCoA
# ─────────────────────────────────────────────────────────────────────────────

elif plot_type.startswith("NMDS"):
    c1, c2, c3, c4 = st.columns(4)
    sh_col   = c1.selectbox("Distance level", tax_candidates,
                             index=tax_candidates.index(taxon_col)
                             if taxon_col in tax_candidates else 0)
    color_by = c2.radio("Colour by", ["treatment", "site", "rep"])
    combine  = c3.checkbox("Combine replicates", value=False)
    show_ell = c3.checkbox("95% ellipses", value=True)
    n_perm   = c4.number_input("PERMANOVA permutations", 99, 9999, 999, 100)

    mat, labels = build_community_matrix(df, sh_col, reads_col,
                                         combine_reps=combine)
    if len(mat) < 3:
        st.error("Need at least 3 samples."); st.stop()

    pat   = (re.compile(r"^(?P<site>[0-9]+)(?P<treatment>[A-Z])$")
             if combine else re.compile(regex_val))
    a_set = set(c.strip() for c in tr_a_chars.split(","))
    d_set = set(c.strip() for c in tr_d_chars.split(","))
    meta  = []
    for lbl in labels:
        m = pat.match(lbl)
        if m:
            tr_raw = m.group("treatment")
            tr     = "A" if tr_raw in a_set else ("D" if tr_raw in d_set else tr_raw.upper())
            rep    = int(m.group("rep")) if "rep" in pat.groupindex else 0
            meta.append({"label": lbl, "site": int(m.group("site")),
                         "treatment": tr, "rep": rep})
        else:
            meta.append({"label": lbl, "site": 0, "treatment": "?", "rep": 0})
    meta_df = pd.DataFrame(meta)

    dist_mat       = squareform(pdist(mat.values, metric="braycurtis"))
    f_stat, p_val  = permanova(dist_mat, meta_df["treatment"].values, n_perm=int(n_perm))
    coords, stress = run_mds(dist_mat)

    if color_by == "treatment":
        palette    = {"A": "#2c7bb6", "D": "#d7191c"}
        group_vals = meta_df["treatment"].tolist()
    elif color_by == "site":
        su      = sorted(meta_df["site"].unique())
        cmap    = plt.get_cmap("tab10")
        palette = {s: cmap(i/max(len(su)-1, 1)) for i, s in enumerate(su)}
        group_vals = meta_df["site"].tolist()
    else:
        ru      = sorted(meta_df["rep"].unique())
        cmap    = plt.get_cmap("Set2")
        palette = {r: cmap(i/max(len(ru)-1, 1)) for i, r in enumerate(ru)}
        group_vals = meta_df["rep"].tolist()

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(12, 6),
                                    gridspec_kw={"width_ratios": [3, 1],
                                                 "wspace": 0.05})
    ax_s.axis("off")

    plotted = set()
    for i, (x, y) in enumerate(coords):
        gv     = group_vals[i]
        color  = palette.get(gv, "#888")
        lbl    = meta_df.loc[i, "label"]
        tr     = meta_df.loc[i, "treatment"]
        marker = "o" if tr == "A" else "s"
        ax.scatter(x, y, color=color, marker=marker, s=110,
                   edgecolors="white", linewidths=0.8, zorder=3,
                   label=str(gv) if gv not in plotted else "")
        ax.annotate(lbl, xy=(x, y), xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")
        plotted.add(gv)

    if show_ell:
        for gv, color in palette.items():
            idx = [i for i, g in enumerate(group_vals) if g == gv]
            if len(idx) < 3: continue
            confidence_ellipse(coords[idx, 0], coords[idx, 1], ax,
                               facecolor=color, alpha=0.12,
                               edgecolor=color, linewidth=1.5, linestyle="--")

    ax.set_xlabel("NMDS axis 1", fontsize=11)
    ax.set_ylabel("NMDS axis 2", fontsize=11)
    ax.set_title(f"NMDS — Bray-Curtis  |  colour by {color_by}"
                 + ("  [reps combined]" if combine else ""),
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="#ccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#ccc", lw=0.8, zorder=1)
    ax.grid(True, ls="--", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [mpatches.Patch(color=c, label=str(g)) for g, c in palette.items()]
    tr_h    = [ax.scatter([], [], marker="o", color="#555", s=60, label="Tr. A"),
               ax.scatter([], [], marker="s", color="#555", s=60, label="Tr. D")]
    ax.legend(handles=handles+tr_h, title=f"Colour: {color_by}",
              fontsize=8, title_fontsize=8, frameon=True,
              edgecolor="#ccc", loc="lower left")

    p_col   = "#c0392b" if (not np.isnan(p_val) and p_val < 0.05) else "#27ae60"
    p_str   = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    f_str   = f"{f_stat:.3f}" if not np.isnan(f_stat) else "N/A"
    sig_txt = ("p < 0.05\nCommunities differ\nsignificantly A vs D"
               if not np.isnan(p_val) and p_val < 0.05
               else "p ≥ 0.05\nNo significant\ndifference detected")

    ax_s.text(0.05, 0.97,
              f"PERMANOVA\n(A vs D)\n{'─'*20}\n"
              f"F : {f_str}\np : {p_str}\nPerms: {int(n_perm)}\n\n{'─'*20}\n"
              f"Stress: {stress:.4f}\n\n{'─'*20}\n"
              f"< 0.05 excellent\n< 0.10 good\n< 0.20 ok\n> 0.20 poor",
              transform=ax_s.transAxes, va="top", ha="left",
              fontsize=9, fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#ccc", lw=1))
    ax_s.text(0.05, 0.28, sig_txt,
              transform=ax_s.transAxes, va="top", ha="left",
              fontsize=9, color=p_col, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=p_col, lw=1.5))

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.download_button("⬇️ Download PNG", fig_to_bytes(fig),
                       file_name="nmds.png", mime="image/png")

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ REPLICATE SIMILARITY TABLE
# ─────────────────────────────────────────────────────────────────────────────

elif plot_type.startswith("Replicate"):
    c1, c2 = st.columns([2, 2])
    sim_col   = c1.selectbox("Taxon level for BC", tax_candidates,
                              index=tax_candidates.index(taxon_col)
                              if taxon_col in tax_candidates else 0)
    sim_group = c2.radio("Group by", ["site × treatment", "site only"])

    if sim_group == "site × treatment":
        group_keys = ["_site", "_treatment", "_rep"]
        label_fn   = lambda k: f"{k[0]}{k[1]}{k[2]}"
    else:
        group_keys = ["_site", "_rep"]
        label_fn   = lambda k: f"Site{k[0]}_Rep{k[1]}"

    vecs, labels = {}, []
    for keys, sub in df.groupby(group_keys):
        keys  = keys if isinstance(keys, tuple) else (keys,)
        lbl   = label_fn(keys)
        grp   = sub.groupby(sim_col)[reads_col].sum()
        total = grp.sum()
        vecs[lbl] = grp / total * 100.0 if total > 0 else grp * 0.0
        labels.append(lbl)

    labels   = sorted(labels)
    n        = len(labels)
    all_taxa = sorted(set().union(*[v.index for v in vecs.values()]))
    mat      = np.zeros((n, n))
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i == j:
                mat[i, j] = 100.0
            elif i < j:
                va = np.array([float(vecs[la].get(t, 0)) for t in all_taxa])
                vb = np.array([float(vecs[lb].get(t, 0)) for t in all_taxa])
                v  = bc_sim(va, vb)
                mat[i, j] = v if not np.isnan(v) else 0.0
                mat[j, i] = mat[i, j]

    cell  = max(0.55, min(1.2, 12.0/n))
    fig_w = max(8, n*cell+3)
    fig_h = max(6, n*cell+2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap_bc = mcolors.LinearSegmentedColormap.from_list(
        "bc", ["#c0392b", "#f39c12", "#27ae60"])
    im = ax.imshow(mat, cmap=cmap_bc, vmin=0, vmax=100, aspect="auto")
    fig.colorbar(im, ax=ax, label="Bray-Curtis similarity (%)",
                 fraction=0.03, pad=0.02)

    for i in range(n):
        for j in range(n):
            val     = mat[i, j]
            txt_col = "white" if (val < 35 or val > 80) else "black"
            ax.text(j, i, f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=max(6, min(10, 90//n)),
                    color=txt_col, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=max(6, min(9, 80//n)))
    ax.set_yticklabels(labels, fontsize=max(6, min(9, 80//n)))

    for tick, lbl in zip(ax.get_xticklabels(), labels):
        if "A" in lbl: tick.set_color("#2c7bb6")
        elif "D" in lbl: tick.set_color("#d7191c")
    for tick, lbl in zip(ax.get_yticklabels(), labels):
        if "A" in lbl: tick.set_color("#2c7bb6")
        elif "D" in lbl: tick.set_color("#d7191c")

    if sim_group == "site × treatment":
        prev = None
        for j, lbl in enumerate(labels):
            m    = re.match(r"^(\d+)", lbl)
            site = m.group(1) if m else None
            if site != prev and j > 0:
                ax.axhline(j-0.5, color="white", lw=2)
                ax.axvline(j-0.5, color="white", lw=2)
            prev = site

    ax.set_title(f"Replicate BC similarity matrix — {sim_col}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sample", fontsize=11)
    ax.set_ylabel("Sample", fontsize=11)
    handles = [mpatches.Patch(color="#2c7bb6", label="Treatment A"),
               mpatches.Patch(color="#d7191c", label="Treatment D")]
    ax.legend(handles=handles, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.18, 1.12), frameon=True, edgecolor="#ccc")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    st.download_button("⬇️ Download PNG", fig_to_bytes(fig),
                       file_name="similarity_table.png", mime="image/png")

    st.subheader("Similarity values (table)")
    sim_df = pd.DataFrame(mat, index=labels, columns=labels).round(1)
    st.dataframe(sim_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100),
                 use_container_width=True)
