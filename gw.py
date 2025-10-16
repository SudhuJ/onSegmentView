import json
import sys
import os # Import os to get file basenames for prompts
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import ot
from scipy.spatial.distance import pdist, squareform, cdist

# --- HELPER FUNCTIONS (UNCHANGED) ---

def extract_segment_features(json_data):
    """
    Represents each segment (line/arc/circle) by its midpoint or center.
    Returns:
        - A NumPy array of points (midpoints/centers).
        - A list of the original segment data for plotting.
    """
    points = []
    segments = []
    for item in json_data.get("bodyData", []):
        if not item.get("visible", True):
            continue
        
        data = item.get("data", {})
        item_type = item.get("type", "").lower()
        point = None
        
        if item_type == "line":
            start, end = data.get("start"), data.get("end")
            if start and end:
                point = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
        elif item_type == "arc":
            start, end = data.get("start"), data.get("end")
            if start and end:
                point = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
        elif item_type == "circle":
            center = data.get("center")
            if center:
                point = center
        
        if point:
            points.append(point)
            segments.append(item)
            
    return np.array(points), segments

def get_segment_lengths(segments_data):
    """Calculates the length of each segment. For arcs/circles, it uses the chord length/diameter."""
    lengths = []
    for item in segments_data:
        data = item.get("data", {})
        length = 0
        if item.get("type") in ['line', 'arc']:
            start, end = data.get("start"), data.get("end")
            if start and end:
                length = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        elif item.get("type") == 'circle':
             length = 2 * data.get("radius", 0)
        lengths.append(length)
    return np.array(lengths)

def plot_segment(ax, segment_item, color='k', linewidth=1.0, shift_x=0.0):
    """Plots a single segment (line, arc, or circle) on a given axis."""
    item_type = segment_item.get("type")
    data = segment_item.get("data", {})
    if item_type == 'line':
        start, end = data.get("start"), data.get("end")
        if start and end:
            ax.plot([start[0] + shift_x, end[0] + shift_x], [start[1], end[1]], color=color, linewidth=linewidth, zorder=1)
    elif item_type == 'arc':
        center, r, sa, ea = data.get("center"), data.get("radius"), data.get("startAngle"), data.get("endAngle")
        if all(v is not None for v in [center, r, sa, ea]):
            sc = (center[0] + shift_x, center[1])
            ax.add_patch(Arc(sc, 2*r, 2*r, angle=0, theta1=np.rad2deg(sa), theta2=np.rad2deg(ea), color=color, linewidth=linewidth, zorder=1))
    elif item_type == 'circle':
        center, r = data.get("center"), data.get("radius")
        if all(v is not None for v in [center, r]):
            sc = (center[0] + shift_x, center[1])
            ax.add_patch(Circle(sc, r, color=color, fill=False, linewidth=linewidth, zorder=1))

def parse_indices(indices_str):
    """Parses a comma-separated string of indices into a list of integers."""
    if not indices_str.strip(): # Check if the string is empty or just whitespace
        return None
    try:
        return [int(i.strip()) for i in indices_str.split(',')]
    except ValueError:
        print(f"Error: Invalid indices format '{indices_str}'. Please use comma-separated integers (e.g., '0,2,5').")
        sys.exit(1)

def main():
    # --- MODIFICATION: Command-line arguments now only take file paths ---
    if len(sys.argv) != 3:
        print("Usage: python gw.py <data/view_geometry_1.json> <data/view_geometry_2.json>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    
    # 1. Load Data and Extract ALL Segment Features
    geo1 = json.load(open(file1, 'r'))
    geo2 = json.load(open(file2, 'r'))
    all_midpoints1, all_segments1 = extract_segment_features(geo1)
    all_midpoints2, all_segments2 = extract_segment_features(geo2)

    
    # Inform the user how many segments were found in each file
    num_segments1 = len(all_segments1)
    num_segments2 = len(all_segments2)
    print("-" * 30)
    print(f"Found {num_segments1} segments in {os.path.basename(file1)} (View 1).")
    print(f"Found {num_segments2} segments in {os.path.basename(file2)} (View 2).")
    print("-" * 30)

    # Prompt the user for input
    print("Enter the indices of segments to include from View 1.")
    indices1_str = input(f"View 1 indices (e.g., '0,5,12' or leave blank for all): ")
    
    print("\nEnter the indices of segments to include from View 2.")
    indices2_str = input(f"View 2 indices (e.g., '1,4,9' or leave blank for all): ")
    print("-" * 30)
    
    # Parse indices from user input
    indices1 = parse_indices(indices1_str)
    indices2 = parse_indices(indices2_str)

    # If no indices were provided, use all segments
    if indices1 is None:
        indices1 = list(range(len(all_segments1)))
    if indices2 is None:
        indices2 = list(range(len(all_segments2)))
        
    # Filter the data to include only the selected segments
    midpoints1 = all_midpoints1[indices1]
    segments1 = [all_segments1[i] for i in indices1]
    
    midpoints2 = all_midpoints2[indices2]
    segments2 = [all_segments2[i] for i in indices2]
    
    if midpoints1.shape[0] < 1 or midpoints2.shape[0] < 1:
        print("Error: Not enough selected segments to perform matching.")
        sys.exit(1)

    print(f"Selected {len(segments1)} out of {len(all_segments1)} segments for View 1.")
    print(f"Selected {len(segments2)} out of {len(all_segments2)} segments for View 2.")
    print("-" * 20)
    
    # 2. Compute Cost Matrices
    dist_metric = 'euclidean'
    C1 = squareform(pdist(midpoints1, dist_metric))
    C2 = squareform(pdist(midpoints2, dist_metric))
    M = cdist(midpoints1, midpoints2, dist_metric)
    
    C1 /= C1.max() if C1.max() > 0 else 1
    C2 /= C2.max() if C2.max() > 0 else 1
    M /= M.max() if M.max() > 0 else 1

    # 3. Initialize distributions
    p = ot.unif(midpoints1.shape[0])
    q = ot.unif(midpoints2.shape[0])

    # 4. Compute Fused Gromov-Wasserstein
    alpha = 0.5
    print(f"Computing Fused GW with alpha={alpha}...")
    T, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=alpha, log=True)
    
    print(f"Fused GW distance: {log['fgw_dist']:.6f}")

    # 5. Visualization (Unchanged)
    fig, (ax_match, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    ax_match.set_title(f'Partial FGW Segment Matching: (a={alpha}), (metric={dist_metric})', fontsize=16)
    
    x_coords1 = np.array([p[0] for p in all_midpoints1])
    x_coords2 = np.array([p[0] for p in all_midpoints2])
    max_x1 = x_coords1.max()
    min_x2 = x_coords2.min()
    shift_x = max_x1 - min_x2 + (x_coords1.max() - x_coords1.min()) * 0.1

    for seg in all_segments1: plot_segment(ax_match, seg, color='lightgrey', linewidth=1)
    for seg in all_segments2: plot_segment(ax_match, seg, color='lightgrey', linewidth=1, shift_x=shift_x)

    matches = np.argmax(T, axis=1)
    colors = plt.get_cmap('rainbow', len(segments1))

    for i in range(len(segments1)):
        j = matches[i]
        original_index1 = indices1[i]
        original_index2 = indices2[j]
        if j < len(segments2):
            color = colors(i / float(len(segments1)))
            plot_segment(ax_match, segments1[i], color=color, linewidth=2.5)
            plot_segment(ax_match, segments2[j], color=color, linewidth=2.5, shift_x=shift_x)
            label_style = dict(fontsize=8, ha='center', va='center', color='black', zorder=2, 
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            ax_match.text(midpoints1[i][0], midpoints1[i][1], str(original_index1), **label_style)
            ax_match.text(midpoints2[j][0] + shift_x, midpoints2[j][1], str(original_index2), **label_style)

    ax_match.set_aspect('equal', adjustable='box')
    ax_match.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    im = ax_heatmap.imshow(T, cmap='YlOrRd', origin='lower')
    ax_heatmap.set_title('Transport Plan (T)')
    ax_heatmap.set_xlabel(f'View 2 Selected Indices ({len(segments2)})')
    ax_heatmap.set_ylabel(f'View 1 Selected Indices ({len(segments1)})')
    fig.colorbar(im, ax=ax_heatmap, shrink=0.6, label='Correspondence Strength')
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    main()