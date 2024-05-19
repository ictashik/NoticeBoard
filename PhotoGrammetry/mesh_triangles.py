from pxr import Usd, UsdGeom
import numpy as np

def compute_mesh_volume(vertices, indices):
    # Calculate the signed volume of a tetrahedron given vertices and face indices
    volume = 0
    for face in indices:
        v0, v1, v2 = [vertices[i] for i in face]
        # Compute the volume using the scalar triple product (volume of parallelepiped)
        tetra_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
        volume += tetra_volume
    return abs(volume)

def load_usdz_and_compute_volume(filepath):
    stage = Usd.Stage.Open(filepath)
    print(f"Loaded USDZ file: {filepath}")
    print(f"Default up axis: {stage.GetMetadata('upAxis')}")
    print(f"Default meters per unit: {stage.GetMetadata('metersPerUnit')}")
    current_scale = stage.GetMetadata('metersPerUnit') or 1.0  # Default to 1 if not specified

    total_volume = 0

    # Iterate over all Mesh prims
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            faceVertexIndices = mesh.GetFaceVertexIndicesAttr().Get()
            vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

            # Scale points by current_scale
            scaled_points = np.array(points) * current_scale

            # Reshape indices according to vertex counts (assuming all faces are triangles)
            indices = []
            start_idx = 0
            for count in vertex_counts:
                indices.append(faceVertexIndices[start_idx:start_idx + count])
                start_idx += count

            # Calculate the volume of this mesh
            volume = compute_mesh_volume(scaled_points, indices)
            total_volume += volume
            print(f"Volume of {prim.GetPath()}: {volume}")

    return total_volume

# Example usage
file_path = 'tv_retro.usdz'
total_volume = load_usdz_and_compute_volume(file_path)
print(f"Total Volume: {total_volume} cubic meters")