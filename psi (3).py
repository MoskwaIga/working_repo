def plot_feature_target_distribution(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    bins: int = 10,
    figsize: tuple = (8, 4),
):
    import numpy as np
    import matplotlib.pyplot as plt

    feature_cols = [f for f in feature_cols if f in df.columns]

    if not feature_cols:
        raise ValueError("Brak poprawnych feature columns")

    for feature in feature_cols:

        df0 = df[df[target_col] == 0][feature].dropna()
        df1 = df[df[target_col] == 1][feature].dropna()

        all_ = df[feature].dropna().values.astype(float)

        if len(all_) == 0:
            continue

        bp = _breakpoints_from_series(all_, bins)

        x_min = np.percentile(all_, 0.5)
        x_max = np.percentile(all_, 99.5)

        hist_bins = np.linspace(x_min, x_max, 50)

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(
            df0,
            bins=hist_bins,
            density=True,
            alpha=0.5,
            label=f"target = 0 (n={len(df0):,})",
        )

        ax.hist(
            df1,
            bins=hist_bins,
            density=True,
            alpha=0.5,
            label=f"target = 1 (n={len(df1):,})",
        )

        for i, b in enumerate(bp):
            ax.axvline(
                b,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label="PSI bins" if i == 0 else None,
            )

        ax.set_title(f"{feature}: target 0 vs 1")
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.set_xlim(x_min, x_max)
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()