import json
import numpy as np
import matplotlib.pyplot as plt
from pot import gromov
from scipy.spatial import distance_matrix

def load_points_from_geometry(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    points = []
    for item in data.get("bodyData", []):
        geo = item.get("data", {})
        if item["type"] == "line":
            points.append(geo["start"])
            points.append(geo["end"])
        elif item["type"] == "circle":
            points.append(geo["center"])
        elif item["type"] == "arc":
            points.append(geo["center"])
        # You can add more types if needed

    return np.array(points)


def compute_distance_matrix(points):
    return distance_matrix(points, points)


def compute_gw_correspondence(C1, C2, p=None, q=None, epsilon=1e-3):
    """
    C1, C2: intra-domain distance matrices
    p, q: probability distributions (default: uniform)
    """
    if p is None:
        p = np.ones((C1.shape[0],)) / C1.shape[0]
    if q is None:
        q = np.ones((C2.shape[0],)) / C2.shape[0]

    T, log = gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', log=True)
    return T, log


def plot_correspondence(points1, points2, T, threshold=0.05):
    fig, ax = plt.subplots()
    ax.scatter(*points1.T, label="View 1", marker='o')
    ax.scatter(*points2.T, label="View 2", marker='x')

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i, j] > threshold:
                ax.plot([points1[i][0], points2[j][0]], [points1[i][1], points2[j][1]], 'k-', alpha=T[i, j])

    ax.legend()
    ax.set_aspect('equal')
    plt.title("GW Correspondence (thresholded)")
    plt.show()


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python ot.py view1.json view2.json")
        return

    f1, f2 = sys.argv[1], sys.argv[2]
    points1 = load_points_from_geometry(f1)
    points2 = load_points_from_geometry(f2)

    if len(points1) == 0 or len(points2) == 0:
        print("[ERROR] No points extracted.")
        return

    C1 = compute_distance_matrix(points1)
    C2 = compute_distance_matrix(points2)

    T, log = compute_gw_correspondence(C1, C2)

    print(f"[INFO] GW loss: {log['gw_dist']}")
    plot_correspondence(points1, points2, T)


if __name__ == "__main__":
    main()
