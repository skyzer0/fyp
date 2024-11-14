import os
import trimesh
import numpy as np

def scale_stl(mesh_data, min_size, max_size):
    if mesh_data is None:
        raise ValueError("Failed to load STL file.")
    
    current_size = mesh_data.bounding_box.bounds[1] - mesh_data.bounding_box.bounds[0]
    scale_factor = np.min((max_size / current_size, min_size / current_size))

    scaled_mesh = mesh_data.copy()
    scaled_mesh.apply_scale(scale_factor)
    return scaled_mesh

def reset_orientation_and_position(mesh_data):
    mesh_data.apply_translation(-mesh_data.centroid)
    mesh_data.apply_transform(trimesh.transformations.rotation_matrix(np.radians(0), [1, 0, 0]))
    mesh_data.apply_transform(trimesh.transformations.rotation_matrix(np.radians(0), [0, 1, 0]))
    mesh_data.apply_transform(trimesh.transformations.rotation_matrix(np.radians(0), [0, 0, 1]))

def get_next_available_index(directory, prefix="object", extension=".stl"):
    existing_files = os.listdir(directory)
    existing_indices = [int(f[len(prefix):-len(extension)]) for f in existing_files if f.startswith(prefix) and f.endswith(extension)]
    existing_indices.sort()
    for idx, number in enumerate(existing_indices, start=1):
        if idx != number:
            return idx
    return len(existing_indices) + 1

new_min_size = 20
new_max_size = 20
output_dir = '/home/simtech/Downloads/Collected_ur5e/objects/resize_obj'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

for i in range(1, 401):
    file_path = f'/home/simtech/Downloads/Collected_ur5e/objects/{i}.stl'
    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        continue

    try:
        mesh = trimesh.load(file_path)
        resized_mesh = scale_stl(mesh, new_min_size, new_max_size)
        reset_orientation_and_position(resized_mesh)

        next_index = get_next_available_index(output_dir)
        output_file = os.path.join(output_dir, f'object{next_index}.stl')
        resized_mesh.export(output_file)
        print(f"Processed and saved: {output_file}")
    except Exception as e:
        print(f"Error processing file {i}.stl: {e}")
