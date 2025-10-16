import sys
import os
import json
import re
import requests
import math
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

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

    with open(f"data/tessellated_edges_{mode}.json", "w") as f:
        json.dump(edges.json(), f, indent=2)
    with open(f"data/tessellated_faces_{mode}.json", "w") as f:
        json.dump(faces.json(), f, indent=2)
    print(f"[OK] Saved tessellation data for {mode}")

def plot_geometry_from_json(file_path, ax=None):
    """
    Plots geometry from a JSON file and adds segment indices.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if "bodyData" not in data:
        print(f"[WARN] No bodyData in {file_path}")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    ax.set_title(os.path.basename(file_path))
    ax.set_aspect("equal")
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Use enumerate to get an index for each geometry item
    for idx, item in enumerate(data["bodyData"]):
        geo = item.get("data", {})
        geo_type = item.get("type", "").lower()
        
        # Define the style for the index labels, copied from gw.py
        label_style = dict(
            fontsize=8, 
            ha='center', 
            va='center', 
            color='black', 
            zorder=2, 
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7)
        )

        if geo_type == "line":
            start = geo["start"]
            end = geo["end"]
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black')
            
            # Add index at the midpoint of the line
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, str(idx), **label_style)

        elif geo_type == "circle":
            center = geo["center"]
            radius = geo["radius"]
            circle = Circle(center, radius, fill=False, edgecolor='blue')
            ax.add_patch(circle)
            
            # Add index at the center of the circle
            ax.text(center[0], center[1], str(idx), **label_style)

        elif geo_type == "arc":
            center = geo["center"]
            radius = geo["radius"]
            start_angle_rad = geo["startAngle"]
            end_angle_rad = geo["endAngle"]
            
            # Matplotlib's Arc uses degrees
            start_angle_deg = math.degrees(start_angle_rad)
            end_angle_deg = math.degrees(end_angle_rad)
            
            arc = Arc(center, 2 * radius, 2 * radius, angle=0,
                      theta1=start_angle_deg, theta2=end_angle_deg, edgecolor='green')
            ax.add_patch(arc)

            # --- MODIFICATION START ---
            # Add index at the midpoint of the arc
            avg_angle_rad = (start_angle_rad + end_angle_rad) / 2
            mid_x = center[0] + radius * math.cos(avg_angle_rad)
            mid_y = center[1] + radius * math.sin(avg_angle_rad)
            ax.text(mid_x, mid_y, str(idx), **label_style)
            # --- MODIFICATION END ---

        else:
            print(f"[NOTE] Unsupported geometry type: {geo_type}")
            
    return fig
    
    
def main():
    if len(sys.argv) != 4:
        print("Usage: python getview.py <drawing_url> <workspace_url> <version_url>")
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
        with open(f"data/metadata_{name}.json", "w") as f:
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
        filename = f"data/view_geometry_{view_id}.json"
        with open(filename, "w") as f:
            json.dump(geometry, f, indent=2)
        fig = plot_geometry_from_json(filename)
        figs.append(fig)

    plt.show()

if __name__ == "__main__":
    main()
