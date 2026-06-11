

# Preprocess Code


> [!WARNING]
> **NOTE**
> - The code provided here is the sample pipeline we used. You will likely need to modify settings inside each file (e.g., input/output directories).
> - **Read the code carefully before executing and modify the part needed to fit your computing environment.**
> - **Run with care:** some scripts automatically `shutil.rmtree` the output folder. **Double-check paths** to avoid deleting the wrong directory.
> - This preprocessing codebase needs more improvement to be user friently (like auto weight download).
<br>
<br>

0. Create HF account and set your read access token in your environment.

hf download nkp37/OpenVid-1M --repo-type dataset --exclude "OpenVidHD/*" --local-dir scratch/uft5by/OpenVid-1M
```shell
export HF_XET_HIGH_PERFORMANCE=1 # Optional
export HF_HOME="path/to/hf_cache" # Set the location that tracks the download progress
export HF_CACHE_HUB"path/to/hf_file_cache" # If your home directory has a storage limit
ls /scratch/id/path/to/*.zip | xargs -n 1 -P 40 -I {} sh -c 'unzip -q "{}" -d "$(dirname "{}")" && rm "{}"'```

1. Provide OpenVid-1M example of the dataset download from HuggingFace (Please check the code carefully to adapt your environment!).
```shell
    # Dataset Download
    python preprocess/dataset_download/openvid_download.py --start_zip_idx 0 --end_zip_idx 186 --output_directory /scratch/uft5by/datasets

    # Initial CSV Prepare   
    python preprocess/dataset_download/csv_prepare_openvid.py
```
<br>
<br>

2. Check the validity of the videos. Put your cpu # cores as *n*
```shell
  python preprocess/filter_basic.py --num_workersn n 
```
<br>
<br>


3. Scene Cut by AutoShot
```shell
    python preprocess/scoring_scene_cut_autoshot.py
```
<br>

After we have the score, we need to use the following to sort and remove:
```shell
    python preprocess/make_delete_lists_scene_cut.py
```
<br>
<br>


4. Check the Image quality of videos, by randomly smaple frames from the video:
```shell
    python preprocess/scoring_img.py
```

After we have the score, we need to use the following to sort and remove:
```shell
    python preprocess/make_delete_lists_img_scoring.py
```
<br>
<br>


5. Panoptic Segmentation (Needs to use Oneformer environment, Check [here](https://github.com/SHI-Labs/OneFormer) )
```shell
    conda activate oneformer    # Check their environment
    python preprocess/filter_panoptic_multi.py
```
<br>
<br>


6.  Camera Pose Estimation and Filter (The environment might has slight difference than ours, we recommend to use thiers [here](https://github.com/henry123-boy/SpaTrackerV2))
```shell
    python preprocess/track_camera_pose_spatracker2.py
```
<br>

After we have the camera pose estimation, we need to use the following to sort and remove:
```shell
    python preprocess/make_delete_lists_camera.py
```
<br>
<br>



7. Caption (Will download Qwen2.5-VL-32B-Instruct automatically)
```shell
    python preprocess/caption_qwen_multi.py
```
<br>
<br>



8. Motion Tracking (Stage 1 Training Ready)
```shell
    python preprocess/track_regular_motion_cycle.py
```
<br>

After we have the Track Motion information, we need to use the following to sort and remove:
```shell
    python preprocess/make_delete_lists_motion.py       
```
<br>
<br>


9. Track for Frame In-N-Out cases (Stage 2 Training Ready)
```shell
    python preprocess/track_FrameINO.py     # ID Reference images will also be the output
```
<br>
<br>







