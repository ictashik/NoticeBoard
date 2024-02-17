import sys 
import tensorflow as tf
import tensorflow_hub as hub 
import matplotlib.pyplot as plt 
import numpy as np 
import tarfile 
import os 
import pandas as pd 
import matplotlib.patches as mpatches
from PIL import Image 
import requests 
import torch 
from transformers import DPTForDepthEstimation, DPTFeatureExtractor 
import open3d as o3d
import trimesh
from tqdm import tqdm
def computeVolume():
    # Check if filename is passed
    if len(sys.argv) < 2:
        print("Please provide a filename as a command line argument.")
        return

    filename = sys.argv[1]
    print("Filename:", filename)

    # Rest of your function using filename

    model_filename = 'seefood_segmenter_mobile_food_segmenter_V1_1.tar.gz'
    extracted_folder_path = 'extracted_model'


    if not os.path.exists(extracted_folder_path):
        with tarfile.open(model_filename, 'r:gz') as tar:
            tar.extractall(path=extracted_folder_path)
        print("Model extracted")

    # Load the image
    image_path = filename
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.image.resize(image, [513, 513])
    image = image / 255.0  # Normalize to [0, 1]
    print("Image loaded")

    # Check if the image is 3-channel RGB
    if image.shape[-1] != 3:
        print("Make sure your image is RGB.")

    # Expand dimensions for batch
    image_batch = tf.expand_dims(image, 0)

    # Load the local model with specified output keys
    m = hub.KerasLayer(extracted_folder_path, signature_outputs_as_dict=True)
    print("Model loaded")

    # Use the model
    results = m(image_batch)
    print("Model used")

    segmentation_probs = results['food_group_segmenter:semantic_probabilities'][0]
    segmentation_mask = results['food_group_segmenter:semantic_predictions'][0]

    # Define the label classes to remove (adjust as needed)
    classes_to_remove = [0, 23, 24]  # Example: Remove classes 2, 4, and 6

    # Create a mask to remove the specified classes
    mask_to_remove = np.isin(segmentation_mask, classes_to_remove)

    # Apply the mask to remove the corresponding regions from the original image
    image_without_classes = image * (1 - mask_to_remove[..., tf.newaxis])  # Set to black (or any desired background color)
    print("Image processed")

    # Save the modified image without specified classes
    output_image_path = 'modified_image.png'  # Specify the desired output path and filename
    tf.keras.preprocessing.image.save_img(output_image_path, image_without_classes.numpy())  # Save the modified image
    print("Modified image saved")

    # Load the depth estimation model
    torch.device('cpu')
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=False)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    print("Depth estimation model loaded")

    # Load the modified image
    image = Image.open('modified_image.png')
    print("Modified image loaded")

    # Prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    print("Depth estimation completed")

    # Save the depth image
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save("MT_Depth_UnMasked.png")
    print("Depth image saved")

    # Load the depth image
    image_path = 'MT_Depth_UnMasked.png'
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.image.resize(image, [513, 513])
    new_image = image / 255.0  # Normalize to [0, 1]
    print("Depth image loaded")

    # Apply the mask to the new image to remove the background
    new_image_without_background = new_image * (1 - mask_to_remove[..., tf.newaxis])  # Set to black (or any desired background color)
    print("New image processed")

    # Save the new image without the background
    output_new_image_path = 'MT_Depth_Masked.png'  # Specify the output path and filename
    tf.keras.preprocessing.image.save_img(output_new_image_path, new_image_without_background.numpy())  # Save the new image without the background
    print("New image saved")

    # Create a point cloud from the depth image
    depthim = new_image_without_background
    points_data = np.dstack(np.mgrid[0:depthim.shape[0], 0:depthim.shape[1]]) 

    # Ensure depthim is a 3-dimensional array
    if len(depthim.shape) == 2:
        depthim = np.expand_dims(depthim, axis=2)

    points_data = np.append(points_data, depthim, axis=2)
    points_data = points_data.reshape(-1, 3)

    # Create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_data)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print("Point cloud created")

    # Create the triangle mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    print("Triangle mesh created")

    # Convert Open3D mesh to trimesh mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices, faces)
    print("Trimesh mesh created")

    # Fix the normals
    trimesh_mesh.fix_normals()
    print("Normals fixed")

    # Compute volume
    volume = trimesh_mesh.volume
    print("Volume computed")

    trimesh_mesh.show()
    

    return volume

if __name__ == "__main__":
    with tqdm(total=100, desc="Computing Volume") as pbar:
        k = computeVolume()
        pbar.update(100)
    print("Volume computed: ",k)
