# Preprocess ScanNet++ for Quicksplat training

- Render the depth maps
- Export the colmap point cloud to ply file and crop the point cloud based on the mesh
- Provide the transforms.json file for each scene


# Render depth
```bash
python -m preprocess.copy_data_and_render_depth \
    --data_root <SCANNETPP_ROOT>/data \
    --scene_file  <OUTPUT_DATA_ROOT>/splits/test_scene_ids.txt \
    --output_root <OUTPUT_DATA_ROOT>
```



# Prepare the data for training
The script will preprocess the ScanNet++ COLMAP point clouds and GT point clouds from the mesh, and voxel downsample them to the specified voxel size.

```bash
python -m preprocess.prepare_spp_data \
    --data_root <SCANNETPP_ROOT>/data \
    --scene_file  <OUTPUT_DATA_ROOT>/splits/test_scene_ids.txt \
    --output_root <OUTPUT_DATA_ROOT> \
    --voxel_size 0.02
```


# Download the data splits & transforms.json files used for testing

The training and validation splits in Quicksplat are the subset of ScanNet++ train split. The test split is the subset of the ScanNet++ val split. We filter out the scenes that have incomplete meshes (e.g., missing walls) to have better GT for training and evaluation. We also filter out the scenes that are too large to train.


# The overall directory structure after preprocessing
```
<OUTPUT_DATA_ROOT>/
├── splits/                     # Data splits for training, validation, and testing
├── colmap/                     # COLMAP point clouds and depth maps
├── mesh_simplified/            # GT point clouds from mesh simplified (for training)
├── mesh/                       # GT mesh (for evaluation)
├── data/
│   ├── scene1/
│   ├── scene2/
│   ├── scene3/
│       ├── depth/              # Rendered depth maps
│       ├── images/             # Undistorted images
│       ├── transforms.json     # Camera poses
```