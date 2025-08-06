import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

def plot_geometry_from_json(file_path, ax=None):
    with open(file_path, "r") as f:
        data = json.load(f)

    if "bodyData" not in data:
        print(f"[WARN] No bodyData in {file_path}")
        return

    # Only create a new figure if no axes are passed
    fig, ax = plt.subplots()
    ax.set_title(file_path)
    ax.set_aspect("equal")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.grid(True)

    for item in data["bodyData"]:
        geo = item.get("data", {})
        geo_type = item.get("type", "").lower()

        if geo_type == "line":
            start = geo["start"]
            end = geo["end"]
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black')

        elif geo_type == "circle":
            center = geo["center"]
            radius = geo["radius"]
            circle = Circle(center, radius, fill=False, edgecolor='blue')
            ax.add_patch(circle)

        elif geo_type == "arc":
            center = geo["center"]
            radius = geo["radius"]
            start_angle = math.degrees(geo["startAngle"])
            end_angle = math.degrees(geo["endAngle"])
            arc = Arc(center, 2 * radius, 2 * radius, angle=0,
                      theta1=start_angle, theta2=end_angle, edgecolor='green')
            ax.add_patch(arc)

        else:
            print(f"[NOTE] Unsupported geometry type: {geo_type}")
    return fig
    