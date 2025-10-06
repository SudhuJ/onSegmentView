# Onshape 2D-3D Correspondence

`GW is yet to be completed as of now. Drawing plots and json data work for line segments.`

This project extracts 2D drawing views and 3D mesh data from Onshape documents and computes the Gromov-Wasserstein correspondences between them using the POT library.

In your terminal, run `python getDocInfo.py <DRAWING_URL> <3D_WORKSPACE_URL> <3D_VERSION_URL>` to get `drawingData.json` along with corresponding plots. Note that the latter links refer to `PART_STUDIO` tab.

## Files

- `getDocInfo.py`: Downloads geometry from Onshape (tessellated 3D meshes and 2D drawing views).
- `plotDrawing.py`: Plots drawing views.
- `gwCalc.py`: Computes GW correspondences with Geodesic/Euclidean Metrics.

## Requirements

The dependencies used are:
```
conda install -c conda-forge python-dotenv matplotlib scipy
pip install numpy pot gdist
```

For getting the API keys to enter in your .env file with names `ONSHAPE_ACCESS_KEY` & `ONSHAPE_SECRET_KEY`, 
along with authentication steps, refer to [Onshape API Quickstart](https://onshape-public.github.io/docs/api-intro/quickstart/).