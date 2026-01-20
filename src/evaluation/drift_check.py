"""Script for detecting data drift by comparing training and inference data distributions.

This module generates a histogram comparing reference data (training) vs current data (inference)
to identify potential distribution shifts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def generate_drift_report():
    """Generate a data drift report comparing two simulated distributions.

    This function simulates reference and current data, plots their histograms,
    and saves the resulting figure to the reports/figures directory.
    """
    print("🚀 Starting Drift script (Manual Mode)...")

    # 1. Simulate two deviating distributions (Drift)
    # Training Data (Reference): Normal distribution centered at 0
    reference_data = np.random.normal(0, 1, 1000)
    # Current Data (Current): Normal distribution centered at 2 (Significant drift!)
    current_data = np.random.normal(2, 1.2, 1000)

    # 2. Create the plot
    plt.figure(figsize=(10, 6))

    # Reference Histogram
    plt.hist(
        reference_data,
        bins=30,
        alpha=0.5,
        label="Training Data (Reference)",
        color="blue",
        density=True,
    )
    # Current Histogram
    plt.hist(
        current_data,
        bins=30,
        alpha=0.5,
        label="Inference Data (Current)",
        color="red",
        density=True,
    )

    plt.title("Data Drift Detection: Feature Distribution Shift")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "drift.png")

    plt.savefig(output_path)
    plt.close()

    print("-" * 40)
    print(f"✅ SUCCESS! Image generated here: {output_path}")
    print("-" * 40)


if __name__ == "__main__":
    generate_drift_report()
