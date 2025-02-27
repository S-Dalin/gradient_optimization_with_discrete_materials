import numpy as np
import matplotlib.pyplot as plt
import os


def plot_best_geometries_by_weights(
    best_geometries_by_weight,
    wavelengths_original,
    wavelengths_new,
    weights,
    target_lambda_index,
):
    num_weights = len(weights)
    num_geometries = 5  # Number of best geometries per weight

    fig, axes = plt.subplots(
        num_geometries, num_weights, figsize=(35, 30), sharex=True, sharey=True
    )
    os.makedirs("save_figures", exist_ok=True)

    # Loop through each weight
    for col_idx, weight in enumerate(weights):
        best_geometries = best_geometries_by_weight[weight].reset_index(drop=True)

        # Loop through the best 5 geometries for the current weight
        for row_idx, row in best_geometries.iterrows():
            ax = axes[row_idx, col_idx]
            ax.plot(
                wavelengths_new,
                np.array(row["mie_Qfwd"]).flatten(),
                label="Qfwd",
                marker="o",
                color="#1f77b4ff",
                linestyle="-",
                linewidth=4,
            )
            ax.plot(
                wavelengths_new,
                np.array(row["mie_Qback"]).flatten(),
                label="Qback",
                marker="o",
                color="#a00000ff",
                linestyle="-",
                linewidth=4,
            )

            target_lambda = wavelengths_original[target_lambda_index]

            # Find the closest wavelength in wavelengths_new to the target_lambda
            closest_idx = np.abs(wavelengths_new - target_lambda).argmin()
            closest_wavelength = wavelengths_new[closest_idx]

            # Draw a vertical line at the closest wavelength in the new spectrum
            ax.axvline(
                x=closest_wavelength,
                color="black",
                linestyle="--",
                linewidth=4,
                alpha=0.6,
                label=f"λ = {closest_wavelength:.2f} nm",
            )

            # Draw horizontal lines for Qfwd and Qback at the closest wavelength in the new spectrum
            qfwd_value = np.array(row["mie_Qfwd"]).flatten()[closest_idx]
            qback_value = np.array(row["mie_Qback"]).flatten()[closest_idx]
            ax.set_title(
                f"Qfwd: {qfwd_value:.1f}, Qback: {qback_value:.1f}", fontsize=24
            )

            annotation_text = (
                f"Core: {row['mat_core']} (r$_c$= {row['r_core']:.1f} nm)\n"
                f"Shell : {row['mat_shell']} (t$_s$= {row['r_shell']:.1f} nm)"
            )
            ax.annotate(
                annotation_text,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                fontsize=26,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                    facecolor="white",
                    alpha=0.8,
                ),
                verticalalignment="top",
                horizontalalignment="left",
            )

            # Set labels, title, and legend
            if row_idx == 0:
                ax.set_title(
                    f"Weight: {weight} \n Qfwd: {qfwd_value:.1f}, Qback: {qback_value:.1f}",
                    fontsize=26,
                )
                ax.legend(fontsize=20, loc="upper right", frameon=True)
            if row_idx == num_geometries - 1:
                ax.set_xlabel("Wavelength (nm)", fontsize=26)
            if col_idx == 0:
                ax.set_ylabel(f"Q_scat", fontsize=26)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=20)

    plt.tight_layout()
    plt.show()
