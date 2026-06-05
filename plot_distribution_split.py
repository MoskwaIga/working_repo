import numpy as np
import pandas as pd


def plot_distribution_split(
    df: pd.DataFrame,
    features: list,
    split_month: str,
    month_col: str,
    bins: int = 10,
    ncols: int = 2,
    figsize_per_plot: tuple = (8, 4),
    segment_col: str | None = None,
    segment_values: list | None = None,
    three_periods: bool = False,
):
    """
    Histogram rozkładu (seaborn + KDE) dla featerów z podziałem na okresy.

    Parametry
    ----------
    df              : cały dataset
    features        : lista featerów do narysowania
    split_month     : miesiąc podziału np. "2023-07"
    month_col       : kolumna z miesiącem
    bins            : liczba decyli PSI (domyślnie 10)
    ncols           : liczba kolumn siatki (tylko bez segmentacji)
    figsize_per_plot: (szerokość, wysokość) jednego wykresu
    segment_col     : opcjonalna kolumna segmentacji
    segment_values  : wartości segmentu (wymagane jeśli segment_col podany)
    three_periods   : jeśli True → 3 okresy: przed / split_month / po
                      jeśli False → 2 okresy: przed / od split_month

    Legenda
    -------
    Kliknięcie na element legendy ukrywa/pokazuje dany rozkład.
    Wymaga %matplotlib widget (lub %matplotlib notebook) w Jupyter.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("Żadna z podanych zmiennych nie istnieje w df.")

    # --- Definicje okresów ---
    if three_periods:
        period_defs = [
            (f"przed {split_month}", df[month_col] < split_month),
            (split_month,            df[month_col] == split_month),
            (f"po {split_month}",    df[month_col] > split_month),
        ]
    else:
        period_defs = [
            (f"przed {split_month}", df[month_col] < split_month),
            (f"od {split_month}",    df[month_col] >= split_month),
        ]

    COLORS = ["#378ADD", "#E24B4A", "#1D9E75"]

    # --- Breakpointy PSI z całej populacji ---
    def get_breakpoints(feature):
        all_vals = df[feature].dropna().values.astype(float)
        bp = np.nanquantile(all_vals, np.linspace(0, 1, bins + 1))
        return np.unique(bp)[1:-1]

    # --- Rysowanie jednego subplotu ---
    def draw_ax(ax, feature, data_mask=None):
        """
        Rysuje histogram dla jednego featurea.
        data_mask: opcjonalny boolean mask dla segmentacji.
        Zwraca listę grup artystów per okres (do interaktywnej legendy).
        """
        source = df if data_mask is None else df[data_mask]
        all_vals = source[feature].dropna().values.astype(float)

        if len(all_vals) == 0:
            ax.set_visible(False)
            return []

        bp_inner  = get_breakpoints(feature)
        x_min     = np.percentile(all_vals, 0.5)
        x_max     = np.percentile(all_vals, 99.5)

        artist_groups = []

        for (label, period_mask), color in zip(period_defs, COLORS):
            combined_mask = period_mask if data_mask is None else (data_mask & period_mask)
            vals = df.loc[combined_mask, feature].dropna().values.astype(float)
            vals = vals[(vals >= x_min) & (vals <= x_max)]

            if len(vals) == 0:
                artist_groups.append([])
                continue

            # Śledź ile mamy artystów przed rysowaniem
            n_patches_before     = len(ax.patches)
            n_lines_before       = len(ax.lines)
            n_collections_before = len(ax.collections)

            sns.histplot(
                vals,
                ax=ax,
                stat="density",
                kde=True,
                color=color,
                alpha=0.35,
                label=f"{label} (n={len(vals):,})",
                bins=30,
                line_kws={"linewidth": 1.8},
            )

            # Zbierz nowe artysty dodane przez tę serię
            new_artists = (
                list(ax.patches[n_patches_before:])
                + list(ax.lines[n_lines_before:])
                + list(ax.collections[n_collections_before:])
            )
            artist_groups.append(new_artists)

        # Granice decyli PSI
        for i, bp_val in enumerate(bp_inner):
            ax.axvline(
                bp_val,
                color="black", linestyle="--", linewidth=0.8, alpha=0.45,
                label="granice PSI" if i == 0 else None,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("Gęstość", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        return artist_groups

    # --- Interaktywna legenda ---
    def make_legend_interactive(ax, fig, artist_groups):
        """
        Kliknięcie na element legendy ukrywa/pokazuje odpowiedni rozkład.
        Wymaga: %matplotlib widget w Jupyter.
        """
        leg          = ax.legend(fontsize=7, loc="upper right")
        leg_handles  = leg.legend_handles

        # Dopasuj handle → group (pomijamy PSI lines — brak w artist_groups)
        interactive_handles = leg_handles[:len(artist_groups)]

        for h in interactive_handles:
            h.set_picker(True)
            if hasattr(h, "set_pickradius"):
                h.set_pickradius(8)

        handle_to_group = {
            id(h): g
            for h, g in zip(interactive_handles, artist_groups)
        }
        handle_to_leghandle = {id(h): h for h in interactive_handles}

        def on_pick(event):
            h = event.artist
            key = id(h)
            if key not in handle_to_group:
                return
            group = handle_to_group[key]
            if not group:
                return
            visible = not group[0].get_visible()
            for artist in group:
                artist.set_visible(visible)
            handle_to_leghandle[key].set_alpha(1.0 if visible else 0.25)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

    # ==================================================================
    # BEZ SEGMENTACJI
    # ==================================================================
    if segment_col is None:

        n_plots      = len(features)
        nrows        = -(-n_plots // ncols)
        actual_ncols = min(ncols, n_plots)

        fig, axes = plt.subplots(
            nrows, actual_ncols,
            figsize=(figsize_per_plot[0] * actual_ncols, figsize_per_plot[1] * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for ax, feature in zip(axes_flat, features):
            groups = draw_ax(ax, feature)
            ax.set_title(feature, fontsize=10, fontweight="bold")
            make_legend_interactive(ax, fig, groups)

        for ax in axes_flat[len(features):]:
            ax.axis("off")

        fig.suptitle(
            f"Rozkład przed / od {split_month}  |  granice decyli PSI",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # ==================================================================
    # Z SEGMENTACJĄ
    # ==================================================================
    if segment_values is None:
        raise ValueError("Jeżeli podano segment_col, należy podać segment_values.")

    segments = [(str(v), df[segment_col] == v) for v in segment_values]
    segments.append(("Reszta", ~df[segment_col].isin(segment_values)))

    nrows_seg = len(features)
    ncols_seg = len(segments)

    fig, axes = plt.subplots(
        nrows_seg, ncols_seg,
        figsize=(figsize_per_plot[0] * ncols_seg, figsize_per_plot[1] * nrows_seg),
        squeeze=False,                        # zawsze 2D — eliminuje rozjezdzanie
    )

    for row_idx, feature in enumerate(features):
        for col_idx, (seg_name, seg_mask) in enumerate(segments):
            ax = axes[row_idx, col_idx]
            groups = draw_ax(ax, feature, data_mask=seg_mask)
            ax.set_title(f"{feature}  |  {seg_name}", fontsize=9, fontweight="bold")
            make_legend_interactive(ax, fig, groups)

    fig.suptitle(
        f"Rozkład przed / od {split_month}  |  segmentacja: {segment_col}",
        fontsize=12, fontweight="bold",
    )

    # hspace / wspace zamiast tight_layout — precyzyjna kontrola odstępów
    fig.subplots_adjust(
        hspace=0.55,   # odstęp pionowy między wierszami
        wspace=0.35,   # odstęp poziomy między kolumnami
        top=0.93,      # miejsce na suptitle
        bottom=0.06,
        left=0.06,
        right=0.97,
    )

    return fig
