import streamlit as st
try:
    from rembg import remove
except ImportError as e:
    st.error(f"Failed to import rembg. Ensure libGL.so.1 is installed. Error: {str(e)}")
    st.stop()
import torch
import cv2
import numpy as np
from PIL import Image
import io
import open3d as o3d
from huggingface_hub import hf_hub_download, login
import trimesh
import os
import tempfile
import sys

# Check Python version compatibility
if sys.version_info < (3, 8) or sys.version_info >= (3, 13):
    st.error("This app requires Python 3.8 to 3.12. Current version: " + sys.version)
    st.stop()

# App title and description
st.title("Photo/Text to Simple 3D Model Generator")
st.write("Upload a photo of a single object or enter a short text prompt to generate a basic 3D model.")

# Hugging Face API token input for text-to-3D
hf_token = st.text_input("Enter your Hugging Face API token (required for text-to-3D):", type="password")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        st.success("Hugging Face API token accepted.")
    except Exception as e:
        st.error(f"Invalid Hugging Face API token: {str(e)}")
else:
    st.warning("Hugging Face API token is required for text-to-3D generation.")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image (.jpg, .png)", type=["jpg", "png"])

# Text input for prompt
text_prompt = st.text_input("Or enter a short text prompt (e.g., 'A small toy car'):")

# Image processing functions
def load_image(uploaded_file):
    """Load and convert uploaded image to RGB format."""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def image_to_numpy(image):
    """Convert PIL image to NumPy array."""
    return np.array(image)

def remove_background(image):
    """Remove background from image using rembg."""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        output_bytes = remove(image_bytes)
        bg_removed = Image.open(io.BytesIO(output_bytes)).convert("RGB")
        return bg_removed
    except Exception as e:
        st.error(f"Error removing background: {str(e)}")
        return None

def load_midas_model():
    """Load MiDaS model for depth estimation."""
    try:
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform
        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas.to(device)
        return midas, transform, device
    except Exception as e:
        st.error(f"Error loading MiDaS model: {str(e)}")
        return None, None, None

def estimate_depth(image, midas, transform, device):
    """Estimate depth map from image using MiDaS."""
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy()
        return depth
    except Exception as e:
        st.error(f"Error estimating depth: {str(e)}")
        return None

def depth_to_point_cloud(depth, image):
    """Convert depth map and image to colored point cloud using vectorized operations."""
    try:
        h, w = depth.shape
        fx = fy = 1  # Focal lengths (relative scale)
        cx, cy = w / 2, h / 2  # Principal point
        img = np.array(image) / 255.0  # Normalize RGB to [0, 1]

        # Create meshgrid for pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        # Stack coordinates and colors
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = img.reshape(-1, 3)
        points_rgb = np.hstack([points, colors])

        return points_rgb
    except Exception as e:
        st.error(f"Error creating point cloud: {str(e)}")
        return None

def create_point_cloud(points_rgb):
    """Create Open3D point cloud from points and colors."""
    try:
        xyz = points_rgb[:, :3]
        rgb = points_rgb[:, 3:]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        return pcd
    except Exception as e:
        st.error(f"Error creating point cloud: {str(e)}")
        return None

def estimate_normals(pcd):
    """Estimate normals for point cloud."""
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        return pcd
    except Exception as e:
        st.error(f"Error estimating normals: {str(e)}")
        return None

def reconstruct_mesh(pcd):
    """Reconstruct mesh from point cloud using Poisson reconstruction."""
    try:
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        return mesh, density
    except Exception as e:
        st.error(f"Error reconstructing mesh: {str(e)}")
        return None, None

def remove_low_density_vertices(mesh, density, threshold=0.01):
    """Remove low-density vertices from mesh."""
    try:
        density = np.asarray(density)
        vertices_to_keep = density > np.quantile(density, threshold)
        mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
        return mesh
    except Exception as e:
        st.error(f"Error removing low-density vertices: {str(e)}")
        return None

def export_mesh(mesh, file_path="output_model.obj"):
    """Export mesh to file."""
    try:
        o3d.io.write_triangle_mesh(file_path, mesh)
        return file_path
    except Exception as e:
        st.error(f"Error exporting mesh: {str(e)}")
        return None

def render_mesh_image(mesh):
    """Render mesh as a static image using Open3D offscreen rendering."""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=False)
        vis.add_geometry(mesh)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        return np.asarray(image)
    except Exception as e:
        st.error(f"Error rendering mesh image: {str(e)}")
        return None

def visualize_mesh_streamlit(mesh):
    """Visualize mesh in Streamlit by rendering a static image."""
    try:
        image = render_mesh_image(mesh)
        if image is not None:
            st.image(image, caption="3D Model Preview", use_column_width=True)
            st.write("Download the model to view in a 3D viewer.")
    except Exception as e:
        st.error(f"Error visualizing mesh: {str(e)}")

# Text-to-3D functions
def load_lgm_model(ckpt_path):
    """Load LGM model for text-to-3D generation (placeholder)."""
    try:
        if not torch.cuda.is_available():
            st.warning("No GPU detected. LGM model will run on CPU, which may be slow.")
        from diffusers import StableDiffusionPipeline  # Placeholder; replace with LGM-specific pipeline
        model = StableDiffusionPipeline.from_pretrained(
            "ashawkey/LGM",
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        # Load custom checkpoint (adjust based on LGM repo)
        model.load_lora_weights(ckpt_path)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model
    except Exception as e:
        st.error(f"Failed to load LGM model: {str(e)}")
        return None

def text_to_3d(prompt, model):
    """Generate 3D model from text prompt using LGM (placeholder)."""
    try:
        # Placeholder; replace with LGM-specific inference
        result = model(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]  # Assuming LGM outputs an image or mesh
        # Convert to mesh (adjust based on LGM output format)
        mesh = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        st.warning("Placeholder mesh generated. Update with LGM inference code.")
        return mesh
    except Exception as e:
        st.error(f"Failed to generate 3D model from text: {str(e)}")
        return None

def trimesh_to_open3d(trimesh_mesh):
    """Convert trimesh mesh to Open3D mesh."""
    try:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh
    except Exception as e:
        st.error(f"Error converting trimesh to Open3D: {str(e)}")
        return None

# Main workflow
if uploaded_image is not None:
    st.success("Image uploaded. Proceeding with image-to-3D processing.")
    image = load_image(uploaded_image)
    if image:
        bg_removed = remove_background(image)
        if bg_removed:
            midas, transform, device = load_midas_model()
            if midas and transform:
                depth = estimate_depth(bg_removed, midas, transform, device)
                if depth is not None:
                    points_rgb = depth_to_point_cloud(depth, bg_removed)
                    if points_rgb is not None:
                        pcd = create_point_cloud(points_rgb)
                        if pcd:
                            pcd = estimate_normals(pcd)
                            if pcd:
                                mesh, density = reconstruct_mesh(pcd)
                                if mesh and density is not None:
                                    mesh = remove_low_density_vertices(mesh, density)
                                    if mesh:
                                        file_path = export_mesh(mesh)
                                        if file_path:
                                            visualize_mesh_streamlit(mesh)
                                            with open(file_path, "rb") as file:
                                                st.download_button(
                                                    label="Download 3D Model (.obj)",
                                                    data=file,
                                                    file_name="output_model.obj",
                                                    mime="model/obj"
                                                )

elif text_prompt.strip() != "":
    if not hf_token:
        st.error("Hugging Face API token is required for text-to-3D generation.")
    else:
        st.success("Text prompt entered. Proceeding with text-to-3D processing.")
        try:
            ckpt_path = hf_hub_download(repo_id="ashawkey/LGM", filename="model_fp16_fixrot.safetensors")
            lgm_model = load_lgm_model(ckpt_path)
            if lgm_model:
                mesh_trimesh = text_to_3d(text_prompt, lgm_model)
                if mesh_trimesh:
                    mesh_o3d = trimesh_to_open3d(mesh_trimesh)
                    if mesh_o3d:
                        file_path = export_mesh(mesh_o3d)
                        if file_path:
                            visualize_mesh_streamlit(mesh_o3d)
                            with open(file_path, "rb") as file:
                                st.download_button(
                                    label="Download 3D Model (.obj)",
                                    data=file,
                                    file_name="output_model.obj",
                                    mime="model/obj"
                                )
        except Exception as e:
            st.error(f"Error in text-to-3D processing: {str(e)}")

else:
    st.info("Please upload an image or enter a text prompt to continue.")
