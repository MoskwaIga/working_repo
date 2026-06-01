import ipywidgets as widgets
from IPython.display import display

def explore_psi_thresholds(final_df: pd.DataFrame):
    """
    Interaktywny explorer progów PSI.
    final_df — wynik feature_selection_by_psi() z kolumnami:
               max_psi_vs_all, max_psi_vs_first, target_psi
    """
    import matplotlib.pyplot as plt

    n_total = len(final_df)

    slider_all = widgets.FloatSlider(
        value=0.20, min=0.0, max=1.0, step=0.01,
        description="psi_vs_all:", style={"description_width": "120px"},
        layout=widgets.Layout(width="500px")
    )
    slider_first = widgets.FloatSlider(
        value=0.20, min=0.0, max=1.0, step=0.01,
        description="psi_vs_first:", style={"description_width": "120px"},
        layout=widgets.Layout(width="500px")
    )
    slider_target = widgets.FloatSlider(
        value=0.10, min=0.0, max=1.0, step=0.01,
        description="target_psi:", style={"description_width": "120px"},
        layout=widgets.Layout(width="500px")
    )

    output = widgets.Output()

    def update(change):
        t_all    = slider_all.value
        t_first  = slider_first.value
        t_target = slider_target.value

        mask_all    = final_df["max_psi_vs_all"]   < t_all
        mask_first  = final_df["max_psi_vs_first"] < t_first
        mask_target = final_df["target_psi"]        > t_target

        n_all    = mask_all.sum()
        n_first  = mask_first.sum()
        n_target = mask_target.sum()
        n_all_first        = (mask_all & mask_first).sum()
        n_all_first_target = (mask_all & mask_first & mask_target).sum()

        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            # --- Wykres 1: Waterfall — ile odpada na każdym warunku ---
            labels = [
                "wszystkie",
                f"psi_vs_all\n< {t_all:.2f}",
                f"+ psi_vs_first\n< {t_first:.2f}",
                f"+ target\n> {t_target:.2f}",
            ]
            values = [n_total, n_all, n_all_first, n_all_first_target]
            colors = ["#378ADD", "#1D9E75", "#1D9E75", "#1D9E75"]

            bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
            for bar, val in zip(bars, values):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold"
                )
            axes[0].set_ylabel("Liczba zmiennych")
            axes[0].set_title("Ile zmiennych przeżywa każdy filtr")
            axes[0].set_ylim(0, n_total * 1.15)
            axes[0].grid(axis="y", alpha=0.3)

            # --- Wykres 2: Rozkład PSI dla przeżywających i odpadających ---
            kept     = final_df[mask_all & mask_first & mask_target]["max_psi_vs_all"]
            dropped  = final_df[~(mask_all & mask_first & mask_target)]["max_psi_vs_all"]

            axes[1].hist(dropped, bins=30, color="#E24B4A", alpha=0.6, label=f"odrzucone ({len(dropped)})")
            axes[1].hist(kept,    bins=30, color="#1D9E75", alpha=0.7, label=f"zachowane ({len(kept)})")
            axes[1].axvline(t_all, color="#378ADD", linestyle="--", linewidth=1.5, label=f"próg {t_all:.2f}")
            axes[1].set_xlabel("max_psi_vs_all")
            axes[1].set_ylabel("Liczba zmiennych")
            axes[1].set_title("Rozkład PSI — zachowane vs odrzucone")
            axes[1].legend(fontsize=9)
            axes[1].grid(axis="y", alpha=0.3)

            fig.tight_layout()
            plt.show()

    slider_all.observe(update,   names="value")
    slider_first.observe(update, names="value")
    slider_target.observe(update, names="value")

    display(widgets.VBox([slider_all, slider_first, slider_target, output]))
    update(None)