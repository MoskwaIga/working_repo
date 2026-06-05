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

    reference_df = df[df[month_col] < split_month]
    current_df = df[df[month_col] >= split_month]

    # ==================================================
    # WERSJA BEZ SEGMENTACJI
    # ==================================================

    if segment_col is None:

        n_plots = len(features)
        nrows = -(-n_plots // ncols)
        actual_ncols = min(ncols, n_plots)

        fig, axes = plt.subplots(
            nrows,
            actual_ncols,
            figsize=(
                figsize_per_plot[0] * actual_ncols,
                figsize_per_plot[1] * nrows,
            ),
            squeeze=False,
        )

        axes_flat = axes.flatten()

        for ax, feature in zip(axes_flat, features):

            before = (
                reference_df[feature]
                .dropna()
                .astype(float)
                .values
            )

            after = (
                current_df[feature]
                .dropna()
                .astype(float)
                .values
            )

            if len(before) == 0:
                ax.set_visible(False)
                continue

            bp_inner = _breakpoints(
                reference_df,
                feature,
                bins,
            )

            combined = np.concatenate(
                [arr for arr in [before, after] if len(arr) > 0]
            )

            x_min = np.percentile(combined, 0.5)
            x_max = np.percentile(combined, 99.5)

            hist_bins = np.linspace(x_min, x_max, 50)

            sns.histplot(
                before,
                ax=ax,
                bins=hist_bins,
                stat="density",
                kde=True,
                alpha=0.4,
                color="#378ADD",
                label=f"przed {split_month} (n={len(before):,})",
                line_kws={"linewidth": 1.8},
            )

            if len(after) > 0:
                sns.histplot(
                    after,
                    ax=ax,
                    bins=hist_bins,
                    stat="density",
                    kde=True,
                    alpha=0.4,
                    color="#E24B4A",
                    label=f"od {split_month} (n={len(after):,})",
                    line_kws={"linewidth": 1.8},
                )

            for i, bp_val in enumerate(bp_inner):

                ax.axvline(
                    bp_val,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                    label="granice PSI" if i == 0 else None,
                )

            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel("Gęstość", fontsize=8)
            ax.set_title(
                feature,
                fontsize=10,
                fontweight="bold",
            )

            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        for ax in axes_flat[len(features):]:
            ax.axis("off")

        fig.suptitle(
            f"Rozkład przed / od {split_month} z granicami PSI",
            fontsize=12,
            fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig

    # ==================================================
    # WERSJA Z SEGMENTACJĄ
    # ==================================================

    if segment_values is None:
        raise ValueError(
            "Jeżeli podano segment_col, należy podać segment_values."
        )

    segments = [
        (str(v), df[segment_col] == v)
        for v in segment_values
    ]

    segments.append(
        (
            "Reszta",
            ~df[segment_col].isin(segment_values),
        )
    )

    nrows_seg = len(features)
    ncols_seg = len(segments)

    fig, axes = plt.subplots(
        nrows_seg,
        ncols_seg,
        figsize=(
            figsize_per_plot[0] * ncols_seg,
            figsize_per_plot[1] * nrows_seg,
        ),
        squeeze=False,
    )

    for row_idx, feature in enumerate(features):

        if reference_df[feature].notna().sum() == 0:
            continue

        bp_inner = _breakpoints(
            reference_df,
            feature,
            bins,
        )

        segment_values_all = []

        for _, segment_mask in segments:

            before_seg = (
                df.loc[
                    segment_mask
                    & (df[month_col] < split_month),
                    feature,
                ]
                .dropna()
                .astype(float)
                .values
            )

            after_seg = (
                df.loc[
                    segment_mask
                    & (df[month_col] >= split_month),
                    feature,
                ]
                .dropna()
                .astype(float)
                .values
            )

            if len(before_seg):
                segment_values_all.append(before_seg)

            if len(after_seg):
                segment_values_all.append(after_seg)

        if len(segment_values_all) == 0:
            continue

        combined = np.concatenate(segment_values_all)

        x_min = np.percentile(combined, 0.5)
        x_max = np.percentile(combined, 99.5)

        hist_bins = np.linspace(x_min, x_max, 50)

        for col_idx, (segment_name, segment_mask) in enumerate(segments):

            ax = axes[row_idx, col_idx]

            before = (
                df.loc[
                    segment_mask
                    & (df[month_col] < split_month),
                    feature,
                ]
                .dropna()
                .astype(float)
                .values
            )

            after = (
                df.loc[
                    segment_mask
                    & (df[month_col] >= split_month),
                    feature,
                ]
                .dropna()
                .astype(float)
                .values
            )

            if len(before) > 0:
                sns.histplot(
                    before,
                    ax=ax,
                    bins=hist_bins,
                    stat="density",
                    kde=True,
                    alpha=0.4,
                    color="#378ADD",
                    label=f"przed {split_month} (n={len(before):,})",
                    line_kws={"linewidth": 1.8},
                )

            if len(after) > 0:
                sns.histplot(
                    after,
                    ax=ax,
                    bins=hist_bins,
                    stat="density",
                    kde=True,
                    alpha=0.4,
                    color="#E24B4A",
                    label=f"od {split_month} (n={len(after):,})",
                    line_kws={"linewidth": 1.8},
                )

            for i, bp_val in enumerate(bp_inner):

                ax.axvline(
                    bp_val,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                    label="granice PSI" if i == 0 else None,
                )

            ax.set_xlim(x_min, x_max)

            ax.set_title(
                f"{feature} | {segment_name}",
                fontsize=9,
                fontweight="bold",
            )

            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel("Gęstość", fontsize=8)

            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        y_max = max(
            axes[row_idx, c].get_ylim()[1]
            for c in range(ncols_seg)
        )

        for c in range(ncols_seg):
            axes[row_idx, c].set_ylim(0, y_max)

    fig.suptitle(
        f"Rozkład przed / od {split_month} | segmentacja: {segment_col}",
        fontsize=12,
        fontweight="bold",
    )

    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.35,
        top=0.93,
        bottom=0.06,
    )

    return fig