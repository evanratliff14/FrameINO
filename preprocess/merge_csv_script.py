from argparse import ArgumentParser
from pathlib import Path
import glob
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument("--folder_a", type=str, required=True)
    parser.add_argument("--folder_b", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    dir_a, dir_b, out_dir = Path(args.folder_a), Path(args.folder_b), Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["Text_Area", "Image_Quality_Assessment", "Aesthetic", "Image_Complexity", "First_Frame_Clarity", "vtss"]

    for fp_a in glob.glob(str(dir_a / "sub*.csv")):
        path_a = Path(fp_a)
        path_b = dir_b / path_a.name
        print(f"Processing {path_a}, {path_b}")

        if not path_b.exists():
            print(f"Skipping {str(path_b)}")
            continue

        # Outer merge combines all unique columns side-by-side
        df_a, df_b = pd.read_csv(path_a), pd.read_csv(path_b)
        merged_df = pd.merge(df_a, df_b, on="video_path", how="outer", suffixes = ('', '_remove'))
        merged_df = merged_df.drop(merged_df.filter(regex='_remove$').columns, axis=1)
        
        # Deduplicate identical column names if any exist
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        cleaned_df = merged_df.dropna(subset=metrics)
        cleaned_df.to_csv(out_dir / path_a.name, index=False)
        print(f"Processed {path_a.name}: kept {len(cleaned_df)}/{len(merged_df)} rows.")

if __name__ == "__main__":
    main()