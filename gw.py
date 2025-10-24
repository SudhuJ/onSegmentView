import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import ot
from scipy.spatial.distance import cdist
import trimesh
from networkx import to_scipy_sparse_array
from scipy.sparse.csgraph import dijkstra

def extract_segment_features_3d(json_data, vertex_kdtree):
    """
    Extracts 3D features for each segment, including its corresponding mesh vertex indices.

    Returns:
        A list of dictionaries, where each dictionary represents a segment.
    """
    segments = []
    for item in json_data.get("bodyData", []):
        if not item.get("visible", True) or item.get("type") not in ["line", "arc", "circle"]:
            continue
        
        data = item.get("data", {})
        item_type = item.get("type", "").lower()
        
        start_3d, end_3d = data.get("start"), data.get("end")
        center_3d = data.get("center")
        
        if (item_type in ["line", "arc"] and start_3d and end_3d) or \
           (item_type == "circle" and center_3d):
            
            midpoint_3d = np.array([(start_3d[0] + end_3d[0]) / 2.0, 
                                    (start_3d[1] + end_3d[1]) / 2.0,
                                    (start_3d[2] + end_3d[2]) / 2.0]) if start_3d else np.array(center_3d)

            _, start_v_idx = vertex_kdtree.query(start_3d) if start_3d else (None, None)
            _, end_v_idx = vertex_kdtree.query(end_3d) if end_3d else (None, None)

            length_3d = np.linalg.norm(np.array(start_3d) - np.array(end_3d)) if start_3d and end_3d else 2 * data.get("radius", 0)

            segments.append({
                'item_data': item,
                'midpoint_3d': midpoint_3d,
                'length_3d': length_3d,
                'start_v_idx': start_v_idx,
                'end_v_idx': end_v_idx,
            })
            
    return segments

def compute_geodesic_cost_matrix(segments, vv_matrix):
    """
    Computes the cost matrix between segments using your geodesic approximation.
    Formula: dist(A,B) = min(endpoint_dists) + len(A)/2 + len(B)/2
    """
    n = len(segments)
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue

            seg_a = segments[i]
            seg_b = segments[j]

            a_v1, a_v2 = seg_a['start_v_idx'], seg_a['end_v_idx']
            b_v1, b_v2 = seg_b['start_v_idx'], seg_b['end_v_idx']
            
            if a_v1 is None or a_v2 is None or b_v1 is None or b_v2 is None:
                dist = np.linalg.norm(seg_a['midpoint_3d'] - seg_b['midpoint_3d'])
                cost_matrix[i, j] = cost_matrix[j, i] = dist
                continue

            dists = [
                vv_matrix[a_v1, b_v1], vv_matrix[a_v1, b_v2],
                vv_matrix[a_v2, b_v1], vv_matrix[a_v2, b_v2]
            ]
            
            min_endpoint_dist = np.min(dists)
            final_dist = min_endpoint_dist + (seg_a['length_3d'] / 2.0) + (seg_b['length_3d'] / 2.0)
            
            cost_matrix[i, j] = cost_matrix[j, i] = final_dist
            
    return cost_matrix

def plot_segment(ax, segment_item, color='k', linewidth=1.0, shift_x=0.0):
    """Plots a single 2D segment on a given axis."""
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
    if not indices_str.strip(): return None
    try:
        return [int(i.strip()) for i in indices_str.split(',')]
    except ValueError:
        print(f"Error: Invalid indices format '{indices_str}'. Please use comma-separated integers.")
        sys.exit(1)

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python geodesic_gw_matcher.py <view1.json> <view2.json> <mesh.obj> [alpha=0.5]")
        sys.exit(1)

    file1, file2, obj_file = sys.argv[1], sys.argv[2], sys.argv[3]
    alpha = float(sys.argv[4]) if len(sys.argv) == 5 else 0.5
    
    # 1. Load 3D Mesh and Compute Vertex-Vertex Geodesic "Map"
    print(f"--- Loading 3D mesh from {obj_file} ---")
    mesh = trimesh.load_mesh(obj_file)
    graph_nx = mesh.vertex_adjacency_graph
    graph_sparse = to_scipy_sparse_array(graph_nx)
    vv_matrix = dijkstra(csgraph=graph_sparse, directed=False)
    vertex_kdtree = mesh.kdtree
    print(f"Successfully created vertex-vertex geodesic map for {len(mesh.vertices)} vertices.")
    print("-" * 30)

    # 2. Load Drawing Data and Extract 3D Segment Features
    geo1 = json.load(open(file1, 'r'))
    geo2 = json.load(open(file2, 'r'))
    all_segments1 = extract_segment_features_3d(geo1, vertex_kdtree)
    all_segments2 = extract_segment_features_3d(geo2, vertex_kdtree)

    print(f"Found {len(all_segments1)} segments in {os.path.basename(file1)} (View 1).")
    print(f"Found {len(all_segments2)} segments in {os.path.basename(file2)} (View 2).")
    print("-" * 30)

    # 3. Prompt User for Segment Selection
    indices1_str = input(f"View 1 indices (e.g., '0,5,12' or leave blank for all): ")
    indices2_str = input(f"View 2 indices (e.g., '1,4,9' or leave blank for all): ")
    print("-" * 30)
    
    indices1 = parse_indices(indices1_str) or list(range(len(all_segments1)))
    indices2 = parse_indices(indices2_str) or list(range(len(all_segments2)))
        
    segments1 = [all_segments1[i] for i in indices1]
    segments2 = [all_segments2[i] for i in indices2]
    
    if len(segments1) < 1 or len(segments2) < 1:
        print("Error: Not enough selected segments to perform matching.")
        sys.exit(1)

    print(f"Selected {len(segments1)} segments for View 1.")
    print(f"Selected {len(segments2)} segments for View 2.")
    print("-" * 20)
    
    # 4. Compute Cost Matrices
    print("Computing cost matrices with your geodesic approximation...")
    C1 = compute_geodesic_cost_matrix(segments1, vv_matrix)
    C2 = compute_geodesic_cost_matrix(segments2, vv_matrix)
    
    midpoints1_3d = np.array([s['midpoint_3d'] for s in segments1])
    midpoints2_3d = np.array([s['midpoint_3d'] for s in segments2])
    M = cdist(midpoints1_3d, midpoints2_3d, 'euclidean')
    
    C1 /= C1.max() if C1.max() > 0 else 1
    C2 /= C2.max() if C2.max() > 0 else 1
    M /= M.max() if M.max() > 0 else 1

    # 5. Compute Fused Gromov-Wasserstein
    p = ot.unif(len(segments1))
    q = ot.unif(len(segments2))
    print(f"Computing Fused GW with alpha={alpha}...")
    T, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=alpha, log=True)
    print(f"Fused GW distance: {log['fgw_dist']:.6f}")

    # 6. Visualization
    fig, (ax_match, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    ax_match.set_title(f'Partial FGW Segment Matching (Geodesic, a={alpha})', fontsize=16)
    
    # Extract 2D midpoints for plotting from the full segment list
    def get_2d_midpoint(seg_item):
        data = seg_item['data']
        if seg_item['type'] == 'circle':
            return data['center'][:2]
        else:
            start, end = data['start'], data['end']
            return [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]

    all_midpoints1_2d = np.array([get_2d_midpoint(s['item_data']) for s in all_segments1])
    all_midpoints2_2d = np.array([get_2d_midpoint(s['item_data']) for s in all_segments2])
    
    x_coords1 = all_midpoints1_2d[:, 0]
    x_coords2 = all_midpoints2_2d[:, 0]
    shift_x = x_coords1.max() - x_coords2.min() + (x_coords1.max() - x_coords1.min()) * 0.1

    for seg_info in all_segments1: plot_segment(ax_match, seg_info['item_data'], color='lightgrey', linewidth=1)
    for seg_info in all_segments2: plot_segment(ax_match, seg_info['item_data'], color='lightgrey', linewidth=1, shift_x=shift_x)

    matches = np.argmax(T, axis=1)
    colors = plt.get_cmap('rainbow', len(segments1))

    # Get the 2D midpoints for the *selected* segments
    midpoints1_2d = all_midpoints1_2d[indices1]
    midpoints2_2d = all_midpoints2_2d[indices2]

    for i in range(len(segments1)):
        j = matches[i]
        if j < len(segments2):
            color = colors(i / float(len(segments1)))
            plot_segment(ax_match, segments1[i]['item_data'], color=color, linewidth=2.5)
            plot_segment(ax_match, segments2[j]['item_data'], color=color, linewidth=2.5, shift_x=shift_x)
            
            label1 = f"{indices1[i]} ({segments1[i]['item_data'].get('deterministicId', 'N/A')})"
            label2 = f"{indices2[j]} ({segments2[j]['item_data'].get('deterministicId', 'N/A')})"
            
            label_style = dict(fontsize=8, ha='center', va='center', color='black', zorder=2, 
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            
            ax_match.text(midpoints1_2d[i][0], midpoints1_2d[i][1], label1, **label_style)
            ax_match.text(midpoints2_2d[j][0] + shift_x, midpoints2_2d[j][1], label2, **label_style)

    ax_match.set_aspect('equal', adjustable='box')
    ax_match.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    im = ax_heatmap.imshow(T, cmap='YlOrRd', origin='lower')
    ax_heatmap.set_title('Transport Plan (T)')
    ax_heatmap.set_xlabel(f'View 2 Selected Indices ({len(segments2)})')
    ax_heatmap.set_ylabel(f'View 1 Selected Indices ({len(segments1)})')
    fig.colorbar(im, ax=ax_heatmap, shrink=0.6, label='Correspondence Strength')
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    output_filename = f"gw_match_alpha_{alpha}.png"
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()




# import json
# import sys
# import os # Import os to get file basenames for prompts
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Arc
# import ot
# from scipy.spatial.distance import pdist, squareform, cdist

# # --- HELPER FUNCTIONS (UNCHANGED) ---

# def extract_segment_features(json_data):
#     """
#     Represents each segment (line/arc/circle) by its midpoint or center.
#     Returns:
#         - A NumPy array of points (midpoints/centers).
#         - A list of the original segment data for plotting.
#     """
#     points = []
#     segments = []
#     for item in json_data.get("bodyData", []):
#         if not item.get("visible", True):
#             continue
        
#         data = item.get("data", {})
#         item_type = item.get("type", "").lower()
#         point = None
        
#         if item_type == "line":
#             start, end = data.get("start"), data.get("end")
#             if start and end:
#                 point = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
#         elif item_type == "arc":
#             start, end = data.get("start"), data.get("end")
#             if start and end:
#                 point = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
#         elif item_type == "circle":
#             center = data.get("center")
#             if center:
#                 point = center
        
#         if point:
#             points.append(point)
#             # We append the whole item to retain all its data, including the ID
#             segments.append(item)
            
#     return np.array(points), segments

# def get_segment_lengths(segments_data):
#     """Calculates the length of each segment. For arcs/circles, it uses the chord length/diameter."""
#     lengths = []
#     for item in segments_data:
#         data = item.get("data", {})
#         length = 0
#         if item.get("type") in ['line', 'arc']:
#             start, end = data.get("start"), data.get("end")
#             if start and end:
#                 length = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
#         elif item.get("type") == 'circle':
#              length = 2 * data.get("radius", 0)
#         lengths.append(length)
#     return np.array(lengths)

# def plot_segment(ax, segment_item, color='k', linewidth=1.0, shift_x=0.0):
#     """Plots a single segment (line, arc, or circle) on a given axis."""
#     item_type = segment_item.get("type")
#     data = segment_item.get("data", {})
#     if item_type == 'line':
#         start, end = data.get("start"), data.get("end")
#         if start and end:
#             ax.plot([start[0] + shift_x, end[0] + shift_x], [start[1], end[1]], color=color, linewidth=linewidth, zorder=1)
#     elif item_type == 'arc':
#         center, r, sa, ea = data.get("center"), data.get("radius"), data.get("startAngle"), data.get("endAngle")
#         if all(v is not None for v in [center, r, sa, ea]):
#             sc = (center[0] + shift_x, center[1])
#             ax.add_patch(Arc(sc, 2*r, 2*r, angle=0, theta1=np.rad2deg(sa), theta2=np.rad2deg(ea), color=color, linewidth=linewidth, zorder=1))
#     elif item_type == 'circle':
#         center, r = data.get("center"), data.get("radius")
#         if all(v is not None for v in [center, r]):
#             sc = (center[0] + shift_x, center[1])
#             ax.add_patch(Circle(sc, r, color=color, fill=False, linewidth=linewidth, zorder=1))

# def parse_indices(indices_str):
#     """Parses a comma-separated string of indices into a list of integers."""
#     if not indices_str.strip(): # Check if the string is empty or just whitespace
#         return None
#     try:
#         return [int(i.strip()) for i in indices_str.split(',')]
#     except ValueError:
#         print(f"Error: Invalid indices format '{indices_str}'. Please use comma-separated integers (e.g., '0,2,5').")
#         sys.exit(1)

# def main():
#     if len(sys.argv) != 3:
#         print("Usage: python gw.py <data/view_geometry_1.json> <data/view_geometry_2.json>")
#         sys.exit(1)

#     file1, file2 = sys.argv[1], sys.argv[2]
    
#     # 1. Load Data and Extract ALL Segment Features
#     geo1 = json.load(open(file1, 'r'))
#     geo2 = json.load(open(file2, 'r'))
#     all_midpoints1, all_segments1 = extract_segment_features(geo1)
#     all_midpoints2, all_segments2 = extract_segment_features(geo2)

#     num_segments1 = len(all_segments1)
#     num_segments2 = len(all_segments2)
#     print("-" * 30)
#     print(f"Found {num_segments1} segments in {os.path.basename(file1)} (View 1).")
#     print(f"Found {num_segments2} segments in {os.path.basename(file2)} (View 2).")
#     print("-" * 30)

#     # Prompt the user for input
#     print("Enter the indices of segments to include from View 1.")
#     indices1_str = input(f"View 1 indices (e.g., '0,5,12' or leave blank for all): ")
    
#     print("\nEnter the indices of segments to include from View 2.")
#     indices2_str = input(f"View 2 indices (e.g., '1,4,9' or leave blank for all): ")
#     print("-" * 30)
    
#     # Parse indices from user input
#     indices1 = parse_indices(indices1_str)
#     indices2 = parse_indices(indices2_str)

#     # If no indices were provided, use all segments
#     if indices1 is None:
#         indices1 = list(range(len(all_segments1)))
#     if indices2 is None:
#         indices2 = list(range(len(all_segments2)))
        
#     # Filter the data to include only the selected segments
#     midpoints1 = all_midpoints1[indices1]
#     segments1 = [all_segments1[i] for i in indices1]
    
#     midpoints2 = all_midpoints2[indices2]
#     segments2 = [all_segments2[i] for i in indices2]
    
#     if midpoints1.shape[0] < 1 or midpoints2.shape[0] < 1:
#         print("Error: Not enough selected segments to perform matching.")
#         sys.exit(1)

#     print(f"Selected {len(segments1)} out of {len(all_segments1)} segments for View 1.")
#     print(f"Selected {len(segments2)} out of {len(all_segments2)} segments for View 2.")
#     print("-" * 20)
    
#     # 2. Compute Cost Matrices
#     dist_metric = 'euclidean'
#     C1 = squareform(pdist(midpoints1, dist_metric))
#     C2 = squareform(pdist(midpoints2, dist_metric))
#     M = cdist(midpoints1, midpoints2, dist_metric)
    
#     C1 /= C1.max() if C1.max() > 0 else 1
#     C2 /= C2.max() if C2.max() > 0 else 1
#     M /= M.max() if M.max() > 0 else 1

#     # 3. Initialize distributions
#     p = ot.unif(midpoints1.shape[0])
#     q = ot.unif(midpoints2.shape[0])

#     # 4. Compute Fused Gromov-Wasserstein
#     alpha = 0.5
#     print(f"Computing Fused GW with alpha={alpha}...")
#     T, log = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha=alpha, log=True)
    
#     print(f"Fused GW distance: {log['fgw_dist']:.6f}")

#     # 5. Visualization
#     fig, (ax_match, ax_heatmap) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
#     ax_match.set_title(f'Partial FGW Segment Matching: (a={alpha}), (metric={dist_metric})', fontsize=16)
    
#     x_coords1 = np.array([p[0] for p in all_midpoints1])
#     x_coords2 = np.array([p[0] for p in all_midpoints2])
#     max_x1 = x_coords1.max()
#     min_x2 = x_coords2.min()
#     shift_x = max_x1 - min_x2 + (x_coords1.max() - x_coords1.min()) * 0.1

#     for seg in all_segments1: plot_segment(ax_match, seg, color='lightgrey', linewidth=1)
#     for seg in all_segments2: plot_segment(ax_match, seg, color='lightgrey', linewidth=1, shift_x=shift_x)

#     matches = np.argmax(T, axis=1)
#     colors = plt.get_cmap('rainbow', len(segments1))

#     for i in range(len(segments1)):
#         j = matches[i]
#         original_index1 = indices1[i]
#         original_index2 = indices2[j]
#         if j < len(segments2):
#             color = colors(i / float(len(segments1)))
#             plot_segment(ax_match, segments1[i], color=color, linewidth=2.5)
#             plot_segment(ax_match, segments2[j], color=color, linewidth=2.5, shift_x=shift_x)
            
#             # --- MODIFICATION: Add segment ID to the label ---
#             segment_id1 = segments1[i].get('deterministicId', 'N/A')
#             segment_id2 = segments2[j].get('deterministicId', 'N/A')
#             label1 = f"{original_index1} ({segment_id1})"
#             label2 = f"{original_index2} ({segment_id2})"
            
#             label_style = dict(fontsize=8, ha='center', va='center', color='black', zorder=2, 
#                                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            
#             ax_match.text(midpoints1[i][0], midpoints1[i][1], label1, **label_style)
#             ax_match.text(midpoints2[j][0] + shift_x, midpoints2[j][1], label2, **label_style)

#     ax_match.set_aspect('equal', adjustable='box')
#     ax_match.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

#     im = ax_heatmap.imshow(T, cmap='YlOrRd', origin='lower')
#     ax_heatmap.set_title('Transport Plan (T)')
#     ax_heatmap.set_xlabel(f'View 2 Selected Indices ({len(segments2)})')
#     ax_heatmap.set_ylabel(f'View 1 Selected Indices ({len(segments1)})')
#     fig.colorbar(im, ax=ax_heatmap, shrink=0.6, label='Correspondence Strength')
#     plt.tight_layout(pad=3.0)
#     plt.show()

# if __name__ == "__main__":
#     main()