def plot_distribution_split(
    df: pd.DataFrame,
    features: list,
    split_month: str,
    month_col: str,
    bins: int = 10,
    ncols: int = 2,
    figsize_per_plot: tuple = (10, 4),
    segment_col: str | None = None,
    segment_values: list | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("Żadna z podanych zmiennych nie istnieje w df.")

    # ==================================================
    # WERSJA BEZ SEGMENTACJI
    # ==================================================

    if segment_col is None:

        n_plots      = len(features)
        nrows        = -(-n_plots // ncols)
        actual_ncols = min(ncols, n_plots)

        fig, axes = plt.subplots(
            nrows, actual_ncols,
            figsize=(figsize_per_plot[0] * actual_ncols, figsize_per_plot[1] * nrows),
            squeeze=False,   # ✅ zawsze 2D — bez ręcznego reshape
        )
        axes_flat = axes.flatten()

        for ax, feature in zip(axes_flat, features):

            all_ = df[feature].dropna().values.astype(float)
            if len(all_) == 0:
                continue

            before = df[df[month_col] <  split_month][feature].dropna().values.astype(float)
            after  = df[df[month_col] >= split_month][feature].dropna().values.astype(float)

            bp_inner  = _breakpoints_from_series(all_, bins)
            x_min     = np.percentile(all_, 0.5)
            x_max     = np.percentile(all_, 99.5)
            hist_bins = np.linspace(x_min, x_max, 50)

            # ✅ seaborn histplot z KDE zamiast ax.hist
            sns.histplot(
                before, ax=ax, bins=hist_bins, stat="density", kde=True,
                alpha=0.4, color="#378ADD",
                label=f"przed {split_month} (n={len(before):,})",
                line_kws={"linewidth": 1.8},
            )
            sns.histplot(
                after, ax=ax, bins=hist_bins, stat="density", kde=True,
                alpha=0.4, color="#E24B4A",
                label=f"od {split_month} (n={len(after):,})",
                line_kws={"linewidth": 1.8},
            )

            for i, bp_val in enumerate(bp_inner):
                ax.axvline(
                    bp_val, color="black", linestyle="--",
                    linewidth=0.8, alpha=0.5,
                    label="granice PSI" if i == 0 else None,
                )

            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel("Gęstość", fontsize=8)
            ax.set_title(feature, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        for ax in axes_flat[len(features):]:
            ax.axis("off")

        fig.suptitle(
            f"Rozkład przed / od {split_month} z granicami decyli PSI",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])   # ✅ miejsce na suptitle
        return fig

    # ==================================================
    # WERSJA Z SEGMENTACJĄ
    # ==================================================

    if segment_values is None:
        raise ValueError("Jeżeli podano segment_col, należy podać segment_values.")

    segments = [(str(v), df[segment_col] == v) for v in segment_values]
    segments.append(("Reszta", ~df[segment_col].isin(segment_values)))

    nrows_seg = len(features)
    ncols_seg = len(segments)

    fig, axes = plt.subplots(
        nrows_seg, ncols_seg,
        figsize=(figsize_per_plot[0] * ncols_seg, figsize_per_plot[1] * nrows_seg),
        squeeze=False,   # ✅ zawsze 2D — eliminuje rozjezdzanie
    )

    for row_idx, feature in enumerate(features):

        all_ = df[feature].dropna().values.astype(float)
        if len(all_) == 0:
            continue

        bp_inner  = _breakpoints_from_series(all_, bins)
        x_min     = np.percentile(all_, 0.5)
        x_max     = np.percentile(all_, 99.5)
        hist_bins = np.linspace(x_min, x_max, 50)

        for col_idx, (segment_name, segment_mask) in enumerate(segments):
            ax = axes[row_idx, col_idx]

            before = df.loc[segment_mask & (df[month_col] <  split_month), feature].dropna().values.astype(float)
            after  = df.loc[segment_mask & (df[month_col] >= split_month), feature].dropna().values.astype(float)

            sns.histplot(
                before, ax=ax, bins=hist_bins, stat="density", kde=True,
                alpha=0.4, color="#378ADD",
                label=f"przed {split_month} (n={len(before):,})",
                line_kws={"linewidth": 1.8},
            )
            sns.histplot(
                after, ax=ax, bins=hist_bins, stat="density", kde=True,
                alpha=0.4, color="#E24B4A",
                label=f"od {split_month} (n={len(after):,})",
                line_kws={"linewidth": 1.8},
            )

            for i, bp_val in enumerate(bp_inner):
                ax.axvline(
                    bp_val, color="black", linestyle="--",
                    linewidth=0.8, alpha=0.5,
                    label="granice PSI" if i == 0 else None,
                )

            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{feature} | {segment_name}", fontsize=9, fontweight="bold")
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel("Gęstość", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        # ✅ Wspólna oś Y per wiersz — segmenty porównywalne między sobą
        y_max = max(axes[row_idx, c].get_ylim()[1] for c in range(ncols_seg))
        for c in range(ncols_seg):
            axes[row_idx, c].set_ylim(0, y_max)

    fig.suptitle(
        f"Rozkład przed / od {split_month} | segmentacja: {segment_col}",
        fontsize=12, fontweight="bold",
    )
    # ✅ subplots_adjust zamiast tight_layout — precyzyjna kontrola odstępów
    fig.subplots_adjust(hspace=0.5, wspace=0.35, top=0.93, bottom=0.06)
    return fig