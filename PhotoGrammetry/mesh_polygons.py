from pxr import Usd, UsdGeom
import numpy as np

def compute_mesh_volume(vertices, indices):
    # Calculate the signed volume of a tetrahedron given vertices and face indices
    volume = 0
    for face in indices:
        if len(face) > 3:
            # Triangulating the polygon
            for i in range(1, len(face) - 1):
                v0, v1, v2 = vertices[face[0]], vertices[face[i]], vertices[face[i + 1]]
                tetra_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
                volume += tetra_volume
        else:
            v0, v1, v2 = [vertices[i] for i in face]
            tetra_volume = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += tetra_volume
    return abs(volume)

def print_metadata(stage):
    # Print metadata for the USD stage, focusing on common and useful keys
    metadata_keys = ["upAxis", "metersPerUnit", "startTimeCode", "endTimeCode", "timeCodesPerSecond"]
    print("Metadata for the USD stage:")
    for key in metadata_keys:
        if stage.HasMetadata(key):
            value = stage.GetMetadata(key)
            print(f"{key}: {value}")

def load_usdz_and_compute_volume(filepath):
    stage = Usd.Stage.Open(filepath)

    # Print metadata
    print_metadata(stage)

    current_scale = stage.GetMetadata('metersPerUnit') or 1.0
    print(f"Current metersPerUnit: {current_scale}")

    total_volume = 0

    # Iterate over all Mesh prims
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            faceVertexIndices = mesh.GetFaceVertexIndicesAttr().Get()
            vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

            scaled_points = np.array(points) * current_scale
            indices = []
            start_idx = 0
            for count in vertex_counts:
                indices.append(faceVertexIndices[start_idx:start_idx + count])
                start_idx += count

            volume = compute_mesh_volume(scaled_points, indices)
            total_volume += volume
            print(f"Volume of {prim.GetPath()}: {volume}")

    return total_volume

# Example usage
file_path = 'tv_retro.usdz'
total_volume = load_usdz_and_compute_volume(file_path)
print(f"Total Volume: {total_volume} cubic meters")