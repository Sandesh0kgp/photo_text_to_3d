# app.py
import streamlit as st
import numpy as np
from PIL import Image
import open3d as o3d
from rembg import remove
import trimesh
import pyrender
import os

# --------------------------
# Common Functions
# --------------------------
def save_obj(mesh, filename):
    mesh.export(filename)

def show_3d_preview(mesh):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

# --------------------------
# Image to 3D Pipeline
# --------------------------
def image_to_3d(uploaded_image, remove_bg=True):
    # Process image
    img = Image.open(uploaded_image)
    
    if remove_bg:
        img = remove(img)
        img = np.array(img)
    
    # Create point cloud (simplified)
    color_raw = o3d.io.read_image(uploaded_image)
    depth_raw = o3d.geometry.Image(np.zeros_like(img))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    
    # Create mesh
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    # Poisson reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    return mesh

# --------------------------
# Text to 3D Pipeline (Simplified)
# --------------------------
def text_to_3d(text):
    # Placeholder - Real implementation needs DreamFusion/Latent-NeRF
    # Generate simple primitive based on text
    if "vase" in text.lower():
        mesh = trimesh.creation.cylinder(radius=0.5, height=2)
    elif "cube" in text.lower():
        mesh = trimesh.creation.box()
    else:
        mesh = trimesh.creation.icosphere()
    
    return mesh

# --------------------------
# Streamlit UI
# --------------------------
st.title("3D Generator App 🎨→📦")

input_type = st.radio("Choose Input Type:", ["Image", "Text"])

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    remove_bg = st.checkbox("Remove Background", value=True)
    
    if uploaded_file:
        with st.spinner("Creating 3D model..."):
            # Process image
            mesh = image_to_3d(uploaded_file, remove_bg)
            
            # Save and show
            save_obj(mesh, "output.obj")
            
            st.success("Done!")
            st.subheader("3D Preview")
            show_3d_preview(mesh)
            
            with open("output.obj", "rb") as f:
                st.download_button("Download OBJ", f, file_name="model.obj")

elif input_type == "Text":
    text_input = st.text_input("Enter text prompt (e.g., 'A red vase')")
    if text_input:
        with st.spinner("Generating 3D from text..."):
            mesh = text_to_3d(text_input)
            
            # Save and show
            save_obj(mesh, "output.obj")
            
            st.success("Done!")
            st.subheader("3D Preview")
            show_3d_preview(mesh)
            
            with open("output.obj", "rb") as f:
                st.download_button("Download OBJ", f, file_name="text_model.obj")
