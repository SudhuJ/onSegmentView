import json
import sys
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import math

# ======================================================================
# You can integrate the functions below directly into your gw.py script
# ======================================================================

def _euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def compute_midpoint_geodesic_matrix(faces_filepath, edges_filepath, view_geometry_filepath):
    """
    Computes a pairwise geodesic distance matrix between the midpoints of segments
    visible in a 2D drawing view.

    The process involves:
    1. Building a full graph representation of the 3D model's tessellated surface.
    2. Identifying the 3D edges and their midpoints corresponding to the 2D view.
    3. Calculating all-pairs shortest paths between the endpoints of these relevant edges.
    4. Using these endpoint distances to find the true shortest path between each pair of midpoints.

    Args:
        faces_filepath (str): Path to the 3D tessellated faces JSON.
        edges_filepath (str): Path to the 3D tessellated edges JSON.
        view_geometry_filepath (str): Path to the 2D view geometry JSON.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The (k x k) geodesic distance matrix for the k midpoints.
            - list: A list of the k [x, y, z] coordinates for those midpoints.
        Returns (None, None) if an error occurs.
    """
    print("--- Starting Midpoint Geodesic Distance Calculation ---")
    
    # 1: Load data files
    try:
        with open(faces_filepath, 'r') as f: faces_data = json.load(f)
        with open(edges_filepath, 'r') as f: edges_data = json.load(f)
        with open(view_geometry_filepath, 'r') as f: view_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data files: {e}", file=sys.stderr)
        return None, None

    # 2: Build the full 3D mesh graph from vertices
    print("Building full 3D mesh graph...")
    vertex_map, vertex_list = {}, []
    for body in faces_data:
        for face in body.get('faces', []):
            for facet in face.get('facets', []):
                for v_coords in facet.get('vertices', []):
                    v_tuple = tuple(v_coords)
                    if v_tuple not in vertex_map:
                        vertex_map[v_tuple] = len(vertex_list)
                        vertex_list.append(list(v_coords))

    num_vertices = len(vertex_list)
    if num_vertices == 0:
        print("Error: No vertices in faces file.", file=sys.stderr)
        return None, None

    rows, cols, weights = [], [], []
    for body in faces_data:
        for face in body.get('faces', []):
            for facet in face.get('facets', []):
                if len(facet['vertices']) == 3:
                    v_indices = [vertex_map[tuple(v)] for v in facet['vertices']]
                    for i in range(3):
                        start, end = v_indices[i], v_indices[(i + 1) % 3]
                        dist = _euclidean_distance(vertex_list[start], vertex_list[end])
                        rows.extend([start, end]); cols.extend([end, start]); weights.extend([dist, dist])
    
    graph_matrix = csr_matrix((weights, (rows, cols)), shape=(num_vertices, num_vertices))
    print(f"Graph built with {num_vertices} total vertices.")

    # 3: Identify midpoints and their edge endpoints from the 2D view
    print("Identifying target segments and midpoints from 2D view...")
    edge_id_to_vertices = {e['id']: [tuple(v) for v in e['vertices']]
                           for body in edges_data for e in body.get('edges', [])
                           if e.get('id') and len(e.get('vertices', [])) == 2}

    target_segments = []
    # **FIXED**: Changed from .get("edgeId") to .get("deterministicId") to match your new JSON format.
    for item in view_data.get("bodyData", []):
        if (edge_id := item.get("deterministicId")) in edge_id_to_vertices:
            p1_coords, p2_coords = edge_id_to_vertices[edge_id]
            if p1_coords in vertex_map and p2_coords in vertex_map:
                midpoint = [(p1_coords[i] + p2_coords[i]) / 2 for i in range(3)]
                p1_idx, p2_idx = vertex_map[p1_coords], vertex_map[p2_coords]
                target_segments.append({'midpoint': midpoint, 'endpoints': (p1_idx, p2_idx)})

    if not target_segments:
        print("Error: No corresponding 3D segments found for the 2D view.", file=sys.stderr)
        print("Check if 'deterministicId' in the view file matches 'id' in the edges file.")
        return None, None
    
    midpoints = [seg['midpoint'] for seg in target_segments]
    print(f"Found {len(midpoints)} segments corresponding to the 2D view.")

    # Step 4: Compute geodesic distances between all required ENDPOINTS
    endpoint_indices = sorted(list({idx for seg in target_segments for idx in seg['endpoints']}))
    
    print(f"Calculating geodesic distances between {len(endpoint_indices)} unique endpoints...")
    endpoint_dist_matrix = dijkstra(csgraph=graph_matrix, directed=False, indices=endpoint_indices)
    
    # Create a map from original vertex index to its index in the endpoint_dist_matrix
    endpoint_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(endpoint_indices)}

    # Step 5: Compute the final midpoint-to-midpoint distance matrix
    print("Assembling final midpoint-to-midpoint distance matrix...")
    num_midpoints = len(midpoints)
    final_midpoint_matrix = np.zeros((num_midpoints, num_midpoints))

    for i in range(num_midpoints):
        for j in range(i, num_midpoints):
            if i == j: continue

            seg_i = target_segments[i]
            seg_j = target_segments[j]

            Ai_idx, Bi_idx = seg_i['endpoints']
            Aj_idx, Bj_idx = seg_j['endpoints']

            dist_Mi_Ai = _euclidean_distance(seg_i['midpoint'], vertex_list[Ai_idx])
            dist_Mi_Bi = _euclidean_distance(seg_i['midpoint'], vertex_list[Bi_idx])
            
            dist_Mj_Aj = _euclidean_distance(seg_j['midpoint'], vertex_list[Aj_idx])
            dist_Mj_Bj = _euclidean_distance(seg_j['midpoint'], vertex_list[Bj_idx])

            geo_Ai_Aj = endpoint_dist_matrix[endpoint_idx_map[Ai_idx], endpoint_idx_map[Aj_idx]]
            geo_Ai_Bj = endpoint_dist_matrix[endpoint_idx_map[Ai_idx], endpoint_idx_map[Bj_idx]]
            geo_Bi_Aj = endpoint_dist_matrix[endpoint_idx_map[Bi_idx], endpoint_idx_map[Aj_idx]]
            geo_Bi_Bj = endpoint_dist_matrix[endpoint_idx_map[Bi_idx], endpoint_idx_map[Bj_idx]]

            path1 = dist_Mi_Ai + geo_Ai_Aj + dist_Mj_Aj
            path2 = dist_Mi_Ai + geo_Ai_Bj + dist_Mj_Bj
            path3 = dist_Mi_Bi + geo_Bi_Aj + dist_Mj_Aj
            path4 = dist_Mi_Bi + geo_Bi_Bj + dist_Mj_Bj
            
            min_dist = min(path1, path2, path3, path4)
            final_midpoint_matrix[i, j] = final_midpoint_matrix[j, i] = min_dist

    print("--- Midpoint Geodesic Distance Calculation Complete ---")
    return final_midpoint_matrix, midpoints

# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python gw_with_geodesic.py <faces.json> <edges.json> <view_geometry.json>")
        sys.exit(1)

    faces_file, edges_file, view_file = sys.argv[1], sys.argv[2], sys.argv[3]

    C_geodesic, midpoints_3d = compute_midpoint_geodesic_matrix(faces_file, edges_file, view_file)

    if C_geodesic is not None:
        print("\n--- Results ---")
        print(f"Successfully computed geodesic cost matrix.")
        print(f"Shape of the cost matrix (C1 or C2): {C_geodesic.shape}")
        print(f"Number of 3D midpoints: {len(midpoints_3d)}")

