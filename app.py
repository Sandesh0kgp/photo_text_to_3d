# app.py
import streamlit as st
import numpy as np
from PIL import Image
from rembg import remove
import trimesh
import pyrender
import tempfile
import os
import sys
if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
    raise RuntimeError("Python 3.10 required")


# --------------------------
# Configuration
# --------------------------
st.set_page_config(page_title="3D Generator", layout="wide")

# --------------------------
# Common Functions
# --------------------------
def save_mesh(mesh, filename):
    mesh.export(filename)

def show_3d_preview(mesh):
    try:
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    except Exception as e:
        st.error(f"3D preview failed: {str(e)}")

# --------------------------
# Image to 3D Pipeline
# --------------------------
def image_to_mesh(uploaded_image, remove_bg=True):
    try:
        # Process image
        img = Image.open(uploaded_image)
        
        if remove_bg:
            img = remove(img)
            img = np.array(img)
        
        # Create basic mesh (placeholder - replace with actual 3D reconstruction)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1]
        ])
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3]
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    except Exception as e:
        st.error(f"Image processing failed: {str(e)}")
        return None

# --------------------------
# Text to 3D Pipeline
# --------------------------
def text_to_mesh(text):
    try:
        # Generate primitive based on text
        text = text.lower()
        
        if "vase" in text:
            mesh = trimesh.creation.cylinder(height=2, radius=0.5)
        elif "cube" in text:
            mesh = trimesh.creation.box()
        elif "sphere" in text:
            mesh = trimesh.creation.icosphere()
        else:
            mesh = trimesh.creation.torus()
            
        return mesh
    
    except Exception as e:
        st.error(f"Text processing failed: {str(e)}")
        return None

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.title("📷➡️🪄 3D Model Generator")
    
    input_type = st.radio("Select input type:", 
                         ["Image Upload", "Text Prompt"], 
                         horizontal=True)
    
    if input_type == "Image Upload":
        st.subheader("Image to 3D Model")
        uploaded_file = st.file_uploader("Upload an image", 
                                       type=["png", "jpg", "jpeg"])
        remove_bg = st.checkbox("Remove background", value=True)
        
        if uploaded_file and st.button("Generate 3D Model"):
            with st.spinner("🔮 Creating magic..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    mesh = image_to_mesh(tmp_file.name, remove_bg)
                
                if mesh:
                    st.success("🎉 3D Model Generated!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("3D Preview")
                        show_3d_preview(mesh)
                    
                    with col2:
                        st.subheader("Download")
                        with tempfile.NamedTemporaryFile(suffix=".obj") as tmp_obj:
                            save_mesh(mesh, tmp_obj.name)
                            with open(tmp_obj.name, "rb") as f:
                                st.download_button(
                                    label="Download OBJ File",
                                    data=f,
                                    file_name="model.obj",
                                    mime="application/octet-stream"
                                )
    
    else:
        st.subheader("Text to 3D Model")
        text_input = st.text_input("Enter your text prompt:", 
                                 placeholder="e.g., 'A decorative vase'")
        
        if text_input and st.button("Generate 3D Model"):
            with st.spinner("🔮 Conjuring 3D model..."):
                mesh = text_to_mesh(text_input)
                
                if mesh:
                    st.success("🎉 3D Model Generated!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("3D Preview")
                        show_3d_preview(mesh)
                    
                    with col2:
                        st.subheader("Download")
                        with tempfile.NamedTemporaryFile(suffix=".obj") as tmp_obj:
                            save_mesh(mesh, tmp_obj.name)
                            with open(tmp_obj.name, "rb") as f:
                                st.download_button(
                                    label="Download OBJ File",
                                    data=f,
                                    file_name="text_model.obj",
                                    mime="application/octet-stream"
                                )

if __name__ == "__main__":
    main()
