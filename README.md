# Onshape 2D-3D Correspondence

This project extracts 2D drawing views and 3D mesh data from Onshape documents and computes the Gromov-Wasserstein correspondences between them using the POT library.

In your terminal, run the command below to get `drawingData.json` along with corresponding plots. Note that the 3D URLs refer to those from `PART_STUDIO`.

```python getDocInfo.py <DRAWING_URL> <3D_WORKSPACE_URL> <3D_VERSION_URL>```
Create a `data` folder to store the `.json` info. This is the input for our GW Implementation.
```python gw.py metadata_{src_viewid}.json metadata_{tar_viewid}```
## Files

- `getDocInfo.py`: Downloads geometry from Onshape (tessellated 3D meshes and 2D drawing views).
- `gw.py`: Computes GW correspondences with *Euclidean* Metrics.
- `plotDrawing.py`: Plots drawing views.


## Requirements

The dependencies used are:

```
conda install -c conda-forge python-dotenv matplotlib scipy
pip install numpy pot gdist
```

For getting the API keys to enter in your .env file with names `ONSHAPE_ACCESS_KEY` & `ONSHAPE_SECRET_KEY`, 
along with authentication steps, refer to [Onshape API Quickstart](https://onshape-public.github.io/docs/api-intro/quickstart/).
