from argparse import ArgumentParser
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import seaborn as sns

def derivative(combined_vtss):

    # 1. Generate the same continuous curve Pandas uses for plotting
    kde = gaussian_kde(combined_vtss.dropna())

    # Create a dense grid of X-values to evaluate the curve
    x_grid = np.linspace(combined_vtss.min(), combined_vtss.max(), 1000)
    y_grid = kde(x_grid)

    # 2. Find the peaks (Local Maxima where derivative goes from + to -)
    # find_peaks looks for points higher than their immediate neighbors
    peaks, _ = find_peaks(y_grid)

    # 3. Find the valleys (Local Minima where derivative goes from - to +)
    # We invert the curve (-y_grid) so valleys turn into peaks
    valleys, _ = find_peaks(-y_grid)

    # 4. Extract the exact (X, Y) coordinates where derivative is 0
    zero_derivative_x = x_grid[np.concatenate([peaks, valleys])]
    zero_derivative_y = y_grid[np.concatenate([peaks, valleys])]

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    # Plot the baseline density curve
    plt.plot(x_grid, y_grid, color="green", linewidth=2, label="Density Curve")

    # Mark the zero-derivative points with red dots
    plt.scatter(
        zero_derivative_x, 
        zero_derivative_y, 
        color="red", 
        s=50, 
        zorder=5, 
        label="Derivative = 0"
    )

    # Print out the coordinate values to your terminal
    for x, y in zip(zero_derivative_x, zero_derivative_y):
        print(f"Zero Derivative Point found at: X = {x:.4f}, Y = {y:.4f}")

    plt.title("Density Plot with Zero Derivative Points")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("vtss derivative plot")
def ccdf(combined_vtss):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Absolute Count Scale (How many clips survive the threshold)
    sns.ecdfplot(
        x=combined_vtss,
        stat="count", 
        complementary=True,  # This forces the integration from high x to low x
        ax=axes[0]
    )
    axes[0].set_title("Data Retention Count (Cutoff Threshold Analyst)")
    axes[0].set_xlabel("VTSS Threshold Cutoff (Keep everything ≥ x)")
    axes[0].set_ylabel("Total Number of Clips Retained")
    axes[0].grid(True, alpha=0.3)

    # 2. Normalized Proportion Scale (What % of each label survives)
    sns.ecdfplot(
        x=combined_vtss,
        stat="proportion", 
        complementary=True, 
        ax=axes[1]
    )
    axes[1].set_title("Data Retention Proportion per Label")
    axes[1].set_xlabel("VTSS Threshold Cutoff (Keep everything ≥ x)")
    axes[1].set_ylabel("Proportion of Class Retained")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ccdf")

def score_density_f(combined_vtss):
    # Plot the density
    combined_vtss.plot.density(color="green", linewidth=2)
    cdf = combined_vtss.value_counts(normalize=True).sort_index().cumsum()
    plt.title("Density plot of Video Training Suitability Score")
    plt.xlabel("VTSS (higher is better, 0 is synonymous to 3/5 in paper)")
    plt.savefig("Density plot of vtss")

def main(csv_filepath):
    pattern = csv_filepath + "/sub*.csv"
    filepaths = glob.glob(pattern)

    if not filepaths:
        print("No CSV files found matching the pattern.")
        return

    all_vtss_series = []
    
    # we only keep 1 csv-> df worth of memory in RAM to avoid OOM
    length = 0
    for fp in filepaths:
        # Read the unique file path 'fp' from the glob loop
        df = pd.read_csv(fp)
        length+=len(df)
        if "vtss" in df.columns:
            all_vtss_series.append(df["vtss"])
        else:
            print(f"Warning: 'vtss' column missing in {fp}")

    if not all_vtss_series:
        print("No 'vtss' data found to plot.")
        print(f"{length} rows")
        return

    # Combine all collected series into a single master Series
    combined_vtss = pd.concat(all_vtss_series, ignore_index=True)
    # ccdf(combined_vtss)
    score_density_f(combined_vtss)
    print(f"Saved data from {length} rows")






if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--csv_filepath", default=None, type=str)
    args = argparser.parse_args()
    main(args.csv_filepath)