"""
Cut all csvs in directory to above the Video Training Suitability Score (vtss) threshold. See Koala 36M paper (Wang et. al). 
Their threshold is normalized to their subjective score 1-5 during supervised learning (threshold 2.5). Quantitive analysis shows 
2d saddle between two gaussians (d density/dscore = 0 at density ~-0.035). Personal qualitative analysis shows a measurable dropoff
between -0.04 - -0.05. See Koala 36M paper for their paper napkin math. Therefore -0.035 is a safe baseline.

If we instead normalize their threshold of 2.5 based on a paper napkin range and base estimate, we get a much higher threshold
of -0.125.  (2.5-(base=1))/(range=4) = (target_threshold - (base = -0.05))/(range = 0.1), solve for target_threshold. This reflects the 
bias of our data towards lower vtss than Panda 70M. 
"""
import argparse
import glob
import os
import pandas as pd

THRESHOLD = -0.035

def main(input_filepath, output_filepath):

    os.makedirs(output_filepath, exist_ok=True)

    pattern = os.path.join(input_filepath, "sub*.csv")
    filepaths = glob.glob(pattern)

    if not filepaths:
        print(f"No CSV files found matching pattern: {pattern}")
        return

    total_length = 0
    left_length = 0
    
    for fp in filepaths:
        df = pd.read_csv(fp)
        total_length += len(df)
        
        # Optimized vectorized boolean indexing filter
        df = df[df['vtss'] > THRESHOLD]
        left_length += len(df)
        
        # Extract filename (e.g., 'sub1.csv') and map to the new output directory
        filename = os.path.basename(fp)
        out_fp = os.path.join(output_filepath, filename)
        
        # Write out to the clean target location
        df.to_csv(out_fp, index=False)
        

    print(f"Filtering Complete. Saved data from {left_length} / {total_length} rows to '{output_filepath}'.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_filepath", type=str, required=True)
    argparser.add_argument("--output_filepath", type=str, required=True)
    args = argparser.parse_args()
    
    main(input_filepath=args.input_filepath, output_filepath=args.output_filepath)