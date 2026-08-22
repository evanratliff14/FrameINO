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
import cv2

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
        all_vtss_series.append(df)

    if not all_vtss_series:
        print("No 'vtss' data found to plot.")
        print(f"{length} rows")
        return

    # Combine all collected series into a single master Series
    combined_vtss = pd.concat(all_vtss_series, ignore_index=True)

    instruction_prompt = [
        "The video has text overlays, watermarks, artificial borders, or multiple views?",
        "The video is of real-life?",
        "Does video suffer from motion blur, camera jittering, or sudden viewpoint shift?",
        "Foreground occlusion examples: fog, heavy rain, or a person passing too close to the camera. Does video contain significant foreground occlusion?",
        "Does the scene contain any vehicles, animals, humans, objects, tools, or other objects suitable for moving around the scene artificially?",
        "Does the scene contain sexual, violent/gory, political, or any other 'Not-Safe-For-Work' content?",
        "Does the scene contain an object that leaves the frame of view at any time?",
        "Does the scene contain an object that enters the frame of view at any time?"
    ]
    
    # ccdf(combined_vtss)
    # corr_matrix = df[instruction_prompt].corr(numeric_only=True)

    # # 3. Create a heatmap visualization
    # plt.figure(figsize=(30, 30))
    # sns.heatmap(
    #     corr_matrix, 
    #     annot=True,          # Show the correlation numbers inside the boxes
    #     cmap='coolwarm',     # Red for positive, blue for negative correlation
    #     fmt='.1f',           # Limit decimals to 2 places
    #     vmin=-1, vmax=1,     # Fix the scale boundaries to standard correlation limits
    #     linewidths=0.5       # Add small dividers between cells
    # )
    # plt.tight_layout() # Prevents labels from getting cut off at the edges
    # for i,p in enumerate(instruction_prompt):
    #     score_density_f(combined_vtss[p])

    #     plt.title(f"Density plot of \"{" ".join(p.split(" ")[:6])}...\"")

    #     plt.savefig(f"{i}_density.jpg", dpi=300)
    #     plt.close()

    # shot_change = df[(df[instruction_prompt[6]] > 0.5) & (len(df["SceneCut_AutoShot"]) >1)]
    # sc_len = len(shot_change)
    # percent_fail = sc_len/length

    # nsfw = df[df[instruction_prompt[5]] < 0.8]
    # nsfw.to_csv("nsfw.csv")
    # df= df[(df[instruction_prompt[4]] < 0.2) | (df[instruction_prompt[5]] < 0.8) | (df[instruction_prompt[3]] < 0.5) | (df[instruction_prompt[2]] < 0.1) | (df[instruction_prompt[1]] < 0.5) | (df[instruction_prompt[0]] < 0.5)]             
    combined_vtss['vlm_score'] = combined_vtss[instruction_prompt[:5]].sum(axis=1)

    score_density_f(df['vlm_score']) 
    plt.title(f"Density plot of 'vlm_score'")
    plt.savefig(f"vlm_score.jpg", dpi=300)                                                        
    print(f"Saved data from {length} rows")



    def display_video_scores(df, instruction_prompt):
        for idx, row in df.iterrows():
            video_path = row.get('video_path')
            if not video_path or not cv2.os.path.exists(str(video_path)):
                print(f"Skipping row {idx}: Invalid or missing video path '{video_path}'")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                continue

            # Prepare text lines to render
            lines = [f"Total VLM Score: {row['vlm_score']:.2f}", "--- Subscores ---"]
            for p in instruction_prompt:
                score = row.get(p, np.nan)
                short_prompt = " ".join(p.split()[:5]) + "..."
                lines.append(f"{short_prompt}: {score:.2f}" if pd.notnull(score) else f"{short_prompt}: N/A")

            exit_requested = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video playback
                    continue

                # Draw semi-transparent HUD overlay
                overlay = frame.copy()
                h, w = frame.shape[:2]
                cv2.rectangle(overlay, (10, 10), (min(500, w - 10), 30 + len(lines) * 22), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                # Render score text onto the frame
                y0 = 32
                for i, line in enumerate(lines):
                    color = (0, 255, 0) if i == 0 else (255, 255, 255)
                    scale = 0.55 if i == 0 else 0.45
                    cv2.putText(frame, line, (20, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

                cv2.imshow("Video Score Viewer", frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # Esc key
                    exit_requested = True
                    break
                elif key == ord('0'):  # '0' key -> next video
                    break

            cap.release()
            if exit_requested:
                break

        cv2.destroyAllWindows()

    # Run viewer on concatenated dataframe
    display_video_scores(combined_vtss, instruction_prompt)






if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--csv_filepath", default=None, type=str)
    args = argparser.parse_args()
    main(args.csv_filepath)