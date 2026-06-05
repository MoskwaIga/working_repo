import numpy as np
import pandas as pd

EPSILON = 1e-6


# ===========================================================================
# FUNKCJE POMOCNICZE
# ===========================================================================

def _breakpoints(reference: pd.DataFrame, feature: str, bins: int) -> np.ndarray:
    """Breakpointy ZAWSZE z rozkładu referencyjnego — zgodnie z literaturą."""
    vals = reference[feature].dropna().values.astype(float)
    bp   = np.nanquantile(vals, np.linspace(0, 1, bins + 1))
    return np.unique(bp)[1:-1]


def _bin_numeric(vals: np.ndarray, bp_inner: np.ndarray):
    nan_mask      = np.isnan(vals)
    n_cats        = len(bp_inner) + 2
    idx           = np.searchsorted(bp_inner, vals, side='left')
    idx[nan_mask] = len(bp_inner) + 1
    return idx, n_cats


def _psi(ref_counts, cur_counts, n_ref, n_cur) -> float:
    """PSI = Σ (observed% − expected%) × ln(observed% / expected%)"""
    expected = np.clip(ref_counts / n_ref, EPSILON, None)
    observed = np.clip(cur_counts / n_cur, EPSILON, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _psi_matrix(ref_pct: np.ndarray, cur_pct: np.ndarray) -> np.ndarray:
    """PSI wektorowo (n_groups, n_bins) → (n_groups,)."""
    expected = np.clip(ref_pct, EPSILON, None)
    observed = np.clip(cur_pct, EPSILON, None)
    return np.sum((observed - expected) * np.log(observed / expected), axis=1)


def _counts_pivot(bin_idx, group_codes, n_groups, n_cats) -> np.ndarray:
    combined = group_codes * n_cats + bin_idx
    flat     = np.bincount(combined, minlength=n_groups * n_cats)
    return flat.reshape(n_groups, n_cats).astype(float)


def _window_map(months: list, month_map: dict, window_size: int) -> dict:
    n_windows = -(-len(months) // window_size)
    return {
        w: [m for m, i in month_map.items() if i // window_size == w]
        for w in range(n_windows)
    }


def _window_label(w_months: list, window_size: int) -> str:
    return min(w_months) if window_size == 1 else f"{min(w_months)} – {max(w_months)}"


# ===========================================================================
# FUNKCJA GŁÓWNA — porównanie dowolnych dwóch DataFramów
# ===========================================================================

def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list,
    bins: int = 10,
    label: str = "custom",
) -> pd.DataFrame:
    """
    Oblicza PSI dla dwóch DataFramów.
    Breakpointy zawsze z reference — zgodnie z literaturą.

    Użycie
    ------
    compare(train_df, test_df, feature_cols)
    compare(df[df["target"]==0], df[df["target"]==1], feature_cols, label="target_0_vs_1")
    compare(population_df, sample_df, feature_cols, label="sample_vs_population")
    """
    n_ref   = len(reference)
    n_cur   = len(current)
    results = []

    for feat in feature_cols:

        if pd.api.types.is_numeric_dtype(reference[feat]):
            bp_inner = _breakpoints(reference, feat, bins)

            if len(bp_inner) == 0:
                results.append({"feature": feat, "comparison": label, "psi": 0.0})
                continue

            ref_idx, n_cats = _bin_numeric(reference[feat].values.astype(float), bp_inner)
            cur_idx, _      = _bin_numeric(current[feat].values.astype(float),   bp_inner)

            ref_counts = np.bincount(ref_idx, minlength=n_cats).astype(float)
            cur_counts = np.bincount(cur_idx, minlength=n_cats).astype(float)

        else:
            ref_str  = reference[feat].astype(str).where(~reference[feat].isna(), "MISSING")
            cur_str  = current[feat].astype(str).where(~current[feat].isna(),     "MISSING")
            all_cats = sorted(set(ref_str.unique()) | set(cur_str.unique()))
            cat_map  = {v: i for i, v in enumerate(all_cats)}
            n_cats   = len(all_cats)

            ref_counts = np.bincount(ref_str.map(cat_map).values.astype(int), minlength=n_cats).astype(float)
            cur_counts = np.bincount(cur_str.map(cat_map).values.astype(int), minlength=n_cats).astype(float)

        results.append({
            "feature":    feat,
            "comparison": label,
            "psi":        _psi(ref_counts, cur_counts, n_ref, n_cur),
        })

    return pd.DataFrame(results)


# ===========================================================================
# WRAPPER: każde okno vs reszta populacji
# ===========================================================================

def psi_month_vs_all(
    df: pd.DataFrame,
    feature_cols: list,
    month_col: str,
    bins: int = 10,
    window_size: int = 1,
) -> pd.DataFrame:

    months         = sorted(df[month_col].dropna().unique())
    month_map      = {m: i for i, m in enumerate(months)}
    win_map        = _window_map(months, month_map, window_size)
    n_windows      = len(win_map)
    month_to_wcode = {m: i // window_size for m, i in month_map.items()}
    window_codes   = df[month_col].map(month_to_wcode).values.astype(int)
    n_total        = len(df)
    results        = []

    for feat in feature_cols:

        if pd.api.types.is_numeric_dtype(df[feat]):

            # ✅ Breakpointy RAZ z df (referencja = cała populacja)
            bp_inner = _breakpoints(df, feat, bins)

            if len(bp_inner) == 0:
                for w, w_months in win_map.items():
                    results.append({"feature": feat,
                                    "window": _window_label(w_months, window_size),
                                    "comparison": "month_vs_all", "psi": 0.0})
                continue

            # ✅ Binowanie RAZ na całej kolumnie
            bin_idx, n_cats = _bin_numeric(df[feat].values.astype(float), bp_inner)

        else:
            series  = df[feat].astype(str).where(~df[feat].isna(), "MISSING")
            cats    = sorted(series.unique())
            cat_map = {v: i for i, v in enumerate(cats)}
            n_cats  = len(cats)
            bin_idx = series.map(cat_map).values.astype(int)

        # ✅ Zliczenia wszystkich okien jedną operacją
        pivot        = _counts_pivot(bin_idx, window_codes, n_windows, n_cats)
        total        = pivot.sum(axis=0)
        window_sizes = pivot.sum(axis=1, keepdims=True)

        # ✅ Self-contamination fix na poziomie proporcji
        ref_counts = total - pivot
        ref_sizes  = n_total - window_sizes

        ref_pct  = np.clip(ref_counts / ref_sizes,    EPSILON, None)
        cur_pct  = np.clip(pivot      / window_sizes, EPSILON, None)
        psi_vals = _psi_matrix(ref_pct, cur_pct)

        for w, psi in enumerate(psi_vals):
            results.append({
                "feature":    feat,
                "window":     _window_label(win_map[w], window_size),
                "comparison": "month_vs_all",
                "psi":        float(psi),
            })

    return pd.DataFrame(results)


# ===========================================================================
# WRAPPER: każde okno vs pierwsze okno
# ===========================================================================

def psi_month_vs_first(
    df: pd.DataFrame,
    feature_cols: list,
    month_col: str,
    bins: int = 10,
    window_size: int = 1,
) -> pd.DataFrame:
    """
    Reference = pierwsze okno → breakpointy z pierwszego okna. ✅
    Wektoryzacja przez bincount — breakpointy stałe dla wszystkich okien.
    """
    months    = sorted(df[month_col].dropna().unique())
    month_map = {m: i for i, m in enumerate(months)}
    win_map   = _window_map(months, month_map, window_size)

    first_months = win_map[0]
    first_label  = _window_label(first_months, window_size)
    reference    = df[df[month_col].isin(first_months)]
    n_windows    = len(win_map)

    month_to_wcode = {m: i // window_size for m, i in month_map.items()}
    window_codes   = df[month_col].map(month_to_wcode).values.astype(int)
    results        = []

    for feat in feature_cols:

        if pd.api.types.is_numeric_dtype(df[feat]):
            bp_inner = _breakpoints(reference, feat, bins)

            if len(bp_inner) == 0:
                for w, w_months in win_map.items():
                    results.append({
                        "feature": feat,
                        "window":  _window_label(w_months, window_size),
                        "reference_window": first_label,
                        "comparison": "month_vs_first",
                        "psi": 0.0,
                    })
                continue

            bin_idx, n_cats = _bin_numeric(df[feat].values.astype(float), bp_inner)

        else:
            ref_str  = reference[feat].astype(str).where(~reference[feat].isna(), "MISSING")
            all_str  = df[feat].astype(str).where(~df[feat].isna(), "MISSING")
            all_cats = sorted(set(ref_str.unique()) | set(all_str.unique()))
            cat_map  = {v: i for i, v in enumerate(all_cats)}
            n_cats   = len(all_cats)
            bin_idx  = all_str.map(cat_map).values.astype(int)

        pivot        = _counts_pivot(bin_idx, window_codes, n_windows, n_cats)
        window_sizes = pivot.sum(axis=1, keepdims=True)
        ref_counts   = pivot[0]
        ref_size     = float(window_sizes[0, 0])

        ref_pct  = np.clip(np.tile(ref_counts / ref_size, (n_windows, 1)), EPSILON, None)
        cur_pct  = np.clip(pivot / window_sizes, EPSILON, None)
        psi_vals = _psi_matrix(ref_pct, cur_pct)

        for w, psi in enumerate(psi_vals):
            results.append({
                "feature":          feat,
                "window":           _window_label(win_map[w], window_size),
                "reference_window": first_label,
                "comparison":       "month_vs_first",
                "psi":              float(psi),
            })

    return pd.DataFrame(results)