import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import ot
from scipy.spatial.distance import pdist, squareform, cdist

# --- Helper functions (extract_segment_features, plot_segment) remain the same ---

def extract_segment_features(json_data):
    """
    Represents each segment (line/arc) by its midpoint.
    Returns:
        - A NumPy array of midpoints.
        - A list of the original segment data for plotting.
    """
    midpoints = []
    segments = []
    for item in json_data.get("bodyData", []):
        if not item.get("visible", True):
            continue
        data = item.get("data", {})
        start = data.get("start")
        end = data.get("end")
        if start and end:
            midpoint = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
            midpoints.append(midpoint)
            segments.append(item)
    return np.array(midpoints), segments

def get_segment_lengths(segments_data):
    """
    Calculates the length of each segment. For arcs, it uses the chord length.
    """
    lengths = []
    for item in segments_data:
        data = item.get("data", {})
        start = data.get("start")
        end = data.get("end")
        if start and end:
            # Calculate Euclidean distance between start and end points (chord length for arcs)
            length = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
            lengths.append(length)
    return np.array(lengths)

def plot_segment(ax, segment_item, color='k', linewidth=1.0, shift_x=0.0):
    """Plots a single segment (line or arc) on a given axis, with an optional shift."""
    item_type = segment_item.get("type")
    data = segment_item.get("data", {})
    if item_type == 'line':
        start, end = data.get("start"), data.get("end")
        if start and end:
            ax.plot([start[0] + shift_x, end[0] + shift_x], [start[1], end[1]], color=color, linewidth=linewidth, zorder=1)
    elif item_type == 'arc':
        center, radius, start_angle, end_angle = data.get("center"), data.get("radius"), data.get("startAngle"), data.get("endAngle")
        if all(v is not None for v in [center, radius, start_angle, end_angle]):
            shifted_center = (center[0] + shift_x, center[1])
            ax.add_patch(Arc(shifted_center, 2 * radius, 2 * radius, angle=0, theta1=np.rad2deg(start_angle), theta2=np.rad2deg(end_angle), color=color, linewidth=linewidth, zorder=1))

def main():
    if len(sys.argv) != 3:
        print("Usage: python gw.py <path/to/view_geometry_1.json> <path/to/view_geometry_2.json>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    
    # 1. Load Data and Extract Segment Features
    geo1 = json.load(open(file1, 'r'))
    geo2 = json.load(open(file2, 'r'))
    midpoints1, segments1 = extract_segment_features(geo1)
    midpoints2, segments2 = extract_segment_features(geo2)

    if midpoints1.shape[0] < 2 or midpoints2.shape[0] < 2:
        print("Error: Not enough segments found to perform matching.")
        sys.exit(1)

    print(f"Found {len(segments1)} segments in View 1.")
    print(f"Found {len(segments2)} segments in View 2.")

    # --- 2. Compute Cost Matrices ---
    C1 = squareform(pdist(midpoints1, 'euclidean'))
    C2 = squareform(pdist(midpoints2, 'euclidean'))
    M = cdist(midpoints1, midpoints2, 'euclidean')
    C1 /= C1.max() if C1.max() > 0 else 1
    C2 /= C2.max() if C2.max() > 0 else 1
    M /= M.max() if M.max() > 0 else 1

    # 3. Non-Uniform Initialization based on Segment Length
    print("\nInitializing distributions 'p' and 'q' based on segment lengths.")
    lengths1 = get_segment_lengths(segments1)
    lengths2 = get_segment_lengths(segments2)
    p = lengths1 / lengths1.sum() if lengths1.sum() > 0 else ot.unif(len(lengths1))
    q = lengths2 / lengths2.sum() if lengths2.sum() > 0 else ot.unif(len(lengths2))
    
    # 4. Fused GW
    alpha = 0.5 
    print(f"Computing Fused Gromov-Wasserstein with alpha={alpha}...")
    
    T, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=alpha, log=True)
    
    print(f"FGW computation complete. FGW distance: {log['fgw_dist']:.4f}")

    # --- 5. Visualization ---
    fig, (ax_match, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    ax_match.set_title(f'FGW Segment Matching (alpha={alpha}, length-weighted)', fontsize=16)
    
    # Calculate shift
    x_coords1 = np.array([p[0] for p in midpoints1])
    max_x1 = x_coords1.max()
    min_x2 = np.array([p[0] for p in midpoints2]).min()
    shift_x = max_x1 - min_x2 + (x_coords1.max() - x_coords1.min()) * 0.1

    # Plot grey background
    for seg in segments1: plot_segment(ax_match, seg, color='lightgrey', linewidth=1)
    for seg in segments2: plot_segment(ax_match, seg, color='lightgrey', linewidth=1, shift_x=shift_x)

    matches = np.argmax(T, axis=1)
    colors = plt.get_cmap('rainbow', len(segments1))

    for i in range(len(segments1)):
        j = matches[i]
        if j < len(segments2):
            color = colors(i / float(len(segments1)))
            plot_segment(ax_match, segments1[i], color=color, linewidth=2.5)
            ax_match.text(midpoints1[i][0], midpoints1[i][1], str(i), fontsize=8, ha='center', va='center', color='black', zorder=2, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            plot_segment(ax_match, segments2[j], color=color, linewidth=2.5, shift_x=shift_x)
            ax_match.text(midpoints2[j][0] + shift_x, midpoints2[j][1], str(j), fontsize=8, ha='center', va='center', color='black', zorder=2, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    ax_match.set_aspect('equal', adjustable='box')
    ax_match.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Heatmap
    im = ax_heatmap.imshow(T, cmap='YlOrRd', origin='lower')
    ax_heatmap.set_title('Transport Plan (T)')
    ax_heatmap.set_xlabel('View 2 Segment Index')
    ax_heatmap.set_ylabel('View 1 Segment Index')
    fig.colorbar(im, ax=ax_heatmap, shrink=0.6, label='Correspondence Strength')
    
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()