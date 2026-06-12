def plot_eliminated_by_source(
    features_to_drop: list,
    all_features: list,
    prefix_sep: str = "_",
    figsize: tuple = (9, 5),
    title: str = "Eliminacja zmiennych według źródła",
):
    """
    Stacked bar chart showing eliminated (red) vs kept (green) features per source.

    Source is identified by the feature name prefix before the first separator,
    e.g. "km_wiek" → source "km", "br_dochod" → source "br".

    Parameters
    ----------
    features_to_drop : list
        Features marked for removal at a given selection stage.
    all_features : list
        Full list of features entering this selection stage.
    prefix_sep : str, default "_"
        Separator used to extract the source prefix from feature names.
    figsize : tuple, default (9, 5)
        Figure size in inches.
    title : str
        Plot title.
    """
    import matplotlib.pyplot as plt

    if not features_to_drop:
        raise ValueError("Lista features_to_drop jest pusta.")
    if not all_features:
        raise ValueError("Lista all_features jest pusta.")

    def get_source(f):
        return f.split(prefix_sep)[0] if prefix_sep in f else "(brak prefiksu)"

    drop_set = set(features_to_drop)

    # Zliczenie per źródło
    sources_all      = pd.Series([get_source(f) for f in all_features])
    sources_dropped  = pd.Series([get_source(f) for f in features_to_drop])

    total_per_source   = sources_all.value_counts()
    dropped_per_source = sources_dropped.value_counts()

    # Wyrównanie indeksów — każde źródło musi mieć obie wartości
    sources_order  = total_per_source.sort_values(ascending=False).index
    total_counts   = total_per_source.reindex(sources_order, fill_value=0)
    dropped_counts = dropped_per_source.reindex(sources_order, fill_value=0)
    kept_counts    = total_counts - dropped_counts

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(sources_order))

    # Słupki stackowane: najpierw kept (zielony), potem dropped (czerwony) na wierzchu
    bars_kept = ax.bar(
        x, kept_counts.values,
        color="#1D9E75", alpha=0.85, edgecolor="white", linewidth=0.5,
        label="zachowane",
    )
    bars_drop = ax.bar(
        x, dropped_counts.values,
        bottom=kept_counts.values,
        color="#E24B4A", alpha=0.85, edgecolor="white", linewidth=0.5,
        label="usunięte",
    )

    # Etykiety wartości
    for i, (kept, dropped, total) in enumerate(
        zip(kept_counts.values, dropped_counts.values, total_counts.values)
    ):
        if kept > 0:
            ax.text(i, kept / 2, str(int(kept)),
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if dropped > 0:
            ax.text(i, kept + dropped / 2, str(int(dropped)),
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(i, total + 0.2, str(int(total)),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(sources_order, fontsize=10)
    ax.set_ylabel("Liczba zmiennych", fontsize=10)
    ax.set_xlabel("Źródło (prefiks)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, total_counts.values.max() * 1.18)

    fig.tight_layout()
    return fig