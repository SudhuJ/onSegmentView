import json
import sys
import os

def convert_tessellation_to_obj(faces_filepath, output_filepath):
    """
    Parses an Onshape tessellated faces JSON file and converts it into a .obj file.

    Args:
        faces_filepath (str): The path to the input 'tessellated_faces.json' file.
        output_filepath (str): The path where the output '.obj' file will be saved.
    """
    try:
        with open(faces_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{faces_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{faces_filepath}'.")
        return

    # --- Step 1: Extract all unique vertices ---
    vertex_map = {}
    vertex_list = []
    
    for body in data:
        for face in body.get('faces', []):
            for facet in face.get('facets', []):
                for vertex_coords in facet.get('vertices', []):
                    vertex_tuple = tuple(vertex_coords)
                    if vertex_tuple not in vertex_map:
                        vertex_map[vertex_tuple] = len(vertex_list)
                        vertex_list.append(vertex_tuple)

    # --- Step 2: Create a list of faces using vertex indices ---
    face_indices_list = []
    for body in data:
        for face in body.get('faces', []):
            for facet in face.get('facets', []):
                if len(facet.get('vertices', [])) == 3:
                    v1_idx = vertex_map[tuple(facet['vertices'][0])]
                    v2_idx = vertex_map[tuple(facet['vertices'][1])]
                    v3_idx = vertex_map[tuple(facet['vertices'][2])]
                    face_indices_list.append((v1_idx, v2_idx, v3_idx))

    # --- Step 3: Write the data to a .obj file ---
    with open(output_filepath, 'w') as f:
        f.write("# OBJ file generated from Onshape tessellation data\n")
        f.write(f"# Found {len(vertex_list)} unique vertices and {len(face_indices_list)} faces.\n\n")

        for vertex in vertex_list:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
        f.write("\n")

        for face_indices in face_indices_list:
            f.write(f"f {face_indices[0] + 1} {face_indices[1] + 1} {face_indices[2] + 1}\n")

    print(f"Successfully converted data to '{output_filepath}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python meshmaker.py <input_json_file> [output_obj_file]")
        sys.exit(1)

    input_json_file = sys.argv[1]

    # If an output file is not specified, create a default name
    if len(sys.argv) > 2:
        output_obj_file = sys.argv[2]
    else:
        # Replaces the extension and puts it in the same directory
        base = os.path.splitext(input_json_file)[0]
        output_obj_file = base + '.obj'

    convert_tessellation_to_obj(input_json_file, output_obj_file)
