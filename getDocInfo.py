import sys
import os
import json
import re
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt
from plotDrawing import plot_geometry_from_json

load_dotenv()
ACCESS_KEY = os.getenv("ONSHAPE_ACCESS_KEY")
SECRET_KEY = os.getenv("ONSHAPE_SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    print("ERROR: Missing Onshape API credentials. Set them in your .env." \
    "For API Key Generation, See: https://onshape-public.github.io/docs/api-intro/quickstart/")
    sys.exit(1)

BASE_URL = "https://cad.onshape.com"

os.makedirs("../data", exist_ok=True)

def extract_ids_from_url(url):
    match = re.search(r"/documents/([^/]+)/([wvm])/([^/]+)/e/([^/?#]+)", url)
    if not match:
        raise ValueError("Invalid Onshape URL format.")
    return {
        "documentId": match.group(1),
        "wvm": match.group(2),
        "wvmId": match.group(3),
        "elementId": match.group(4)
    }

def get_drawing_views(meta):
    url = f"{BASE_URL}/api/drawings/d/{meta['documentId']}/{meta['wvm']}/{meta['wvmId']}/e/{meta['elementId']}/views"
    response = requests.get(url, auth=HTTPBasicAuth(ACCESS_KEY, SECRET_KEY))
    response.raise_for_status()
    return response.json()

def get_view_geometry(meta, viewid):
    url = f"{BASE_URL}/api/drawings/d/{meta['documentId']}/{meta['wvm']}/{meta['wvmId']}/e/{meta['elementId']}/views/{viewid}/jsongeometry"
    response = requests.get(url, auth=HTTPBasicAuth(ACCESS_KEY, SECRET_KEY))
    response.raise_for_status()
    return response.json()

def get_partstudio_geometry(meta, mode="workspace"):
    base = f"{BASE_URL}/api/partstudios/d/{meta['documentId']}/{meta['wvm']}/{meta['wvmId']}/e/{meta['elementId']}"
    edges_url = base + "/tessellatededges"
    faces_url = base + "/tessellatedfaces"

    edges = requests.get(edges_url, auth=HTTPBasicAuth(ACCESS_KEY, SECRET_KEY))
    faces = requests.get(faces_url, auth=HTTPBasicAuth(ACCESS_KEY, SECRET_KEY))
    edges.raise_for_status()
    faces.raise_for_status()

    with open(f"../data/tessellated_edges_{mode}.json", "w") as f:
        json.dump(edges.json(), f, indent=2)
    with open(f"../data/tessellated_faces_{mode}.json", "w") as f:
        json.dump(faces.json(), f, indent=2)
    print(f"[OK] Saved tessellation data for {mode}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python getDocInfo.py <drawing_url> <workspace_url> <version_url>")
        sys.exit(1)

    try:
        drawing_meta = extract_ids_from_url(sys.argv[1])
        workspace_meta = extract_ids_from_url(sys.argv[2])
        version_meta = extract_ids_from_url(sys.argv[3])
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Save metadata
    for name, meta in zip(["drawing", "workspace", "version"], [drawing_meta, workspace_meta, version_meta]):
        with open(f"../data/metadata_{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[OK] Saved metadata_{name}.json")

    # Fetch 3D geometry for workspace and version
    get_partstudio_geometry(workspace_meta, mode="workspace")
    get_partstudio_geometry(version_meta, mode="version")

    # Fetch drawing views and geometry
    print(f"[INFO] Getting views for drawing element: {drawing_meta['elementId']}")
    views_data = get_drawing_views(drawing_meta)
    views = views_data.get("items", [])
    if not views:
        print("[WARN] No views found.")
        return

    # Geometry Plotting Code.
    figs = []
    for view in views:
        view_id = view.get("viewId")
        if not view_id:
            continue
        print(f"[INFO] Fetching geometry for view {view_id}")
        geometry = get_view_geometry(drawing_meta, view_id)
        filename = f"../data/view_geometry_{view_id}.json"
        with open(filename, "w") as f:
            json.dump(geometry, f, indent=2)
        fig = plot_geometry_from_json(filename)
        figs.append(fig)

    # plt.show()

if __name__ == "__main__":
    main()
