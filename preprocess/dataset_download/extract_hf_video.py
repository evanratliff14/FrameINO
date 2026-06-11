from pathlib import Path
import re
import subprocess
from collections import defaultdict

def __main__():
    DATA_DIR = Path("/scratch/uft5by/OpenVid-1M")
    TARGET_DIR = DATA_DIR / "video"

    TARGET_DIR.mkdir(parents=True, exist_ok=False)

    # Group the split files using regex
    # Matches 'OpenVid_part102' out of 'OpenVid_part102_partaa'
    split_pattern = re.compile(r"^(OpenVid_part\d+)_part[a-z]+")
    groups = defaultdict(list)

    print(f"Scanning directory: {DATA_DIR}...")

    # Use pathlib's iterdir() to look through the directory
    for item in DATA_DIR.iterdir():
        if item.is_file():
            match = split_pattern.match(item.name)
            if match:
                base_zip_name = match.group(1)  # e.g., "OpenVid_part102"
                groups[base_zip_name].append(item)

    if not groups:
        print("No matching split files found.")
        exit()

    print(f"Found {len(groups)} split file sets to reconstruct.\n")

    # Cat the pieces together into full zip files
    for base_name, split_files in groups.items():
        # Sort files by name so partaa comes before partab
        split_files.sort(key=lambda p: p.name)
        
        output_zip = DATA_DIR / f"{base_name}.zip"
        
        print(f" Rebuilding {output_zip.name} from {len(split_files)} parts...")
        
        # Convert pathlib objects to strings for the shell command
        file_strings = " ".join([str(f) for f in split_files])
        cat_cmd = f"cat {file_strings} > {output_zip}"
        
        try:
            subprocess.run(cat_cmd, shell=True, check=True)
            print(f"Created {output_zip.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing cat for {base_name}: {e}")
            continue

    # 4. Unzip everything into the 'video' folder
    print("\n Reconstructed files ready. Starting Unzip Process.")

    # Look for all zipped parts in the directory
    for zip_file in DATA_DIR.glob("OpenVid_part*.zip"):
        print(f" Extracting {zip_file.name} to {TARGET_DIR.name}/ ...")
        try:
            # -j junk paths (keeps video directory flat)
            # -q quiet mode
            unzip_cmd = ["unzip", "-j", str(zip_file), "-d", str(TARGET_DIR)]
            subprocess.run(unzip_cmd, check=True)
            print(f"   Finished extracting {zip_file.name}")
            
            # Optional: Delete the merged zip file to save space after successful extraction
            # zip_file.unlink() 
            
        except subprocess.CalledProcessError as e:
            print(f" Error unzipping {zip_file.name}: {e}")

    print("\n Successfully catted and unzipped!")

if __name__ == "__main__":
    __main__()