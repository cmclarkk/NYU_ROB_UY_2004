import os
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pil_image_to_albedo_texture(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an albedo texture."""
    albedo_texture = np.asarray(image)
    if albedo_texture.ndim == 2:
        # If the texture is grayscale, we need to convert it to RGB since
        # Rerun expects a 3-channel texture.
        # See: https://github.com/rerun-io/rerun/issues/4878
        albedo_texture = np.stack([albedo_texture] * 3, axis=-1)
    return albedo_texture

def scene_to_trimeshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    """
    Convert a trimesh.Scene to a list of trimesh.Trimesh.

    Skips objects that are not an instance of trimesh.Trimesh.
    """
    trimeshes = []
    scene_dump = scene.dump()
    geometries = [scene_dump] if not isinstance(scene_dump, list) else scene_dump
    for geometry in geometries:
        if isinstance(geometry, trimesh.Trimesh):
            trimeshes.append(geometry)
        elif isinstance(geometry, trimesh.Scene):
            trimeshes.extend(scene_to_trimeshes(geometry))
    return trimeshes

def GenerateRandomColors(n):
    cmap = plt.get_cmap('tab10')  # You can change this to other palettes
    colors = [cmap(i / n) for i in range(n)]    
    rgb_colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
    return rgb_colors

class Visualizer:
    def __init__(self, app_name="RerunVisualizer", log_time_label='logtime', spawn=True, port=9876):
        # Initialize Rerun session
        if spawn == False:
            rr.init(app_name, spawn=False)
            rr.connect(f'127.0.0.1:{port}')
        else:
            rr.init(app_name, spawn=True)
        self.log_time_label = log_time_label
    
    def logPoints(self, points, colors=None, radii=None, log_path='/points', log_time=None):
        ps = []
        cs = []
        rs = []
        if colors is None:
            colors = [[0, 255, 0]] * points.shape[0]
        if radii is None:
            radii = [0.002] * points.shape[0]

        rr.log(log_path, rr.Points3D(points, colors = colors, radii=radii))   
        if log_time is not None:
            rr.set_time_seconds(self.log_time_label, log_time)
    
    def logMeshFile(self, mesh_file_path, world_T_mesh, log_path='/mesh', log_time=None, alpha=0.5):
        mesh_or_scene = trimesh.load_mesh(mesh_file_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            meshes = scene_to_trimeshes(mesh_or_scene)
        else:
            meshes = [mesh_or_scene]
        
        for i, mesh in enumerate(meshes):
            vertex_colors = albedo_texture = vertex_texcoords = None
            # If the mesh has vertex colors, use them. Otherwise, use the texture if it exists.
            if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
                vertex_colors = mesh.visual.vertex_colors
            elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                trimesh_material = mesh.visual.material

                if mesh.visual.uv is not None:
                    vertex_texcoords = mesh.visual.uv
                    # Trimesh uses the OpenGL convention for UV coordinates, so we need to flip the V coordinate
                    # since Rerun uses the Vulkan/Metal/DX12/WebGPU convention.
                    vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

                if isinstance(trimesh_material, trimesh.visual.material.PBRMaterial):
                    if trimesh_material.baseColorTexture is not None:
                        albedo_texture = pil_image_to_albedo_texture(
                            trimesh_material.baseColorTexture
                        )
                    elif trimesh_material.baseColorFactor is not None:
                        vertex_colors = trimesh_material.baseColorFactor
                elif isinstance(trimesh_material, trimesh.visual.material.SimpleMaterial):
                    if trimesh_material.image is not None:
                        albedo_texture = pil_image_to_albedo_texture(trimesh_material.image)
                    else:
                        vertex_colors = mesh.visual.to_color().vertex_colors
            vertex_colors[:, -1] = alpha
            rr.log(
                f"{log_path}/{i}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    albedo_texture=albedo_texture,
                    vertex_texcoords=vertex_texcoords,
                ),
            )
            # Transoform the mesh into its world pose
            xyzw = R.from_matrix(world_T_mesh[0:3,0:3]).as_quat()
            quat = rr.Quaternion.identity()
            quat.xyzw = xyzw
            rr.log(f"{log_path}/{i}", rr.Transform3D(translation=world_T_mesh[:3,-1].squeeze(), rotation=quat))
        if log_time is not None:
            rr.set_time_seconds(self.log_time_label, log_time)
    
    def logCoordinateFrame(self, world_T_frame, log_path, axis_length=0.2, log_time=None):
        rr.log(log_path, rr.ViewCoordinates.LEFT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            f"{log_path}",
            rr.Arrows3D(
                vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii = [axis_length/30, axis_length/30, axis_length/30]
            ),
        )
        xyzw = R.from_matrix(world_T_frame[0:3,0:3]).as_quat()
        quat = rr.Quaternion.identity()
        quat.xyzw = xyzw
        rr.log(log_path, rr.Transform3D(translation=world_T_frame[:3,-1].squeeze(), rotation=quat))
        if log_time is not None:
            rr.set_time_seconds(self.log_time_label, log_time)