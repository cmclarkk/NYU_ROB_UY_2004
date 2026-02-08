import os
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from .parsers.urdf import URDFParser  
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


class RerunVisualizer:
    def __init__(self, app_name="RerunVisualizer", log_time_label='logtime', spawn=True, port=9876):
        # Initialize Rerun session
        if spawn == False:
            rr.init(app_name, spawn=False)
            rr.connect(f'127.0.0.1:{port}')
        else:
            rr.init(app_name, spawn=True)
        self.log_time_label = log_time_label
    
    def logMPPI(self, traj_batch):
        rr.log("strip", rr.LineStrips3D(traj_batch.tolist(), colors=[255,255,0,100]))

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
    
    def logTree(self, tree, pointcloud, goal, robot_radius=0.02, goal_radius=0.06):
        node_points = []
        for node in tree:
                if node[1]==True:
                        node_points.append(node[2])
        node_points = np.vstack(node_points)
        print(node_points.shape, pointcloud.shape)

        colors = [[255,0,0], [0,255,0], [255,255,0]]
        radii = [0.002, robot_radius, goal_radius]
        self.draw_point_clouds([pointcloud, node_points, goal.reshape(1,3)], colors, radii)

        lines = []
        for node in tree:
                if node[1]:
                        parent_idx=node[0]
                state = node[2]
                parent_state = tree[parent_idx][2]
                lines.append([[parent_state[0], parent_state[1], parent_state[2]], [state[0], state[1], parent_state[2]]])

        rr.log(
        "segments",
        rr.LineStrips3D(
                np.array(
                lines
                )
        ),
        )
    
    def logEllipsoids(self, ellipsoid_poses, 
                            ellipsoid_scales, 
                            ellipsoid_colors=None,
                            log_path='/ellipsoids',
                            log_time=None):
        for i in range(ellipsoid_poses.shape[0]):
            ellipsoid_path = f"{log_path}/ellipoid_{i}"
            center = np.array([0.0, 0.0, 0.0])
            rr.log(
                ellipsoid_path,
                rr.Ellipsoids3D(
                    centers=[center],
                    half_sizes=[ellipsoid_scales[i]],
                    colors=[[255, 0, 0] if ellipsoid_colors is None else ellipsoid_colors[i]],
                ),
            )
            xyzw = R.from_matrix(ellipsoid_poses[i][0:3,0:3]).as_quat()
            quat = rr.Quaternion.identity()
            quat.xyzw = xyzw
            rr.log(ellipsoid_path, rr.Transform3D(translation=ellipsoid_poses[i][:3,-1].squeeze(), rotation=quat))
        
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


class RerunRobotVisualizer:
    def __init__(self, 
                 robot_name: str, 
                 robot_description_file:str, 
                 robot_assets_path: str, 
                 logger_name: str):
        self.robot_name = robot_name
        self.robot_info = URDFParser(os.path.join(robot_assets_path, robot_description_file))
        self.logger_name = logger_name
        self.visualizer = RerunVisualizer(app_name=logger_name)
        self.mesh_assets = []
        self.pcd_assets = []
        # Log the meshfile corresponding to each link
        self.collision_geoms = {}
        for link_name in self.robot_info.links_info.keys():
            visuals = self.robot_info.links_info[link_name]['visuals']
            if len(visuals) == 0:
                continue
            for n,visual in enumerate(visuals):
                mesh_file = visual['file_path']
                if mesh_file.endswith('.ply'):
                    continue
                mesh_file_path = os.path.join(robot_assets_path, mesh_file)
                self.visualizer.logMeshFile(mesh_file_path, np.eye(4), f'{self.robot_name}/{link_name}/visual_{n}')
        # Store the ellipsoidal collision primitives            
        for link_name in self.robot_info.links_info.keys():
            collisions = self.robot_info.links_info[link_name]['collisions']
            if len(collisions) > 0:
                for i, collision in enumerate(collisions):
                    if collision['type'] == 'ellipsoid':
                        offset = collision['collision_offset']
                        scale = collision['scale']
                        self.collision_geoms[f'{self.robot_name}/{link_name}/collision_{i}'] = (offset, scale)
                        # Log the ellipsoidal collision primitives
                        self.visualizer.logEllipsoids(np.eye(4).reshape(1,4,4), np.array(scale).reshape(1,3), log_path=f'{self.robot_name}/{link_name}/collision_{i}')

    def updateRobot(self, T, q):
        self.robot_info.computeForwardKinematics(q)
        for link_name in self.robot_info.links_info.keys():
            pose = self.robot_info.links_info[link_name]['link_pose']
            self.visualizer.logCoordinateFrame(pose, f'{self.robot_name}/{link_name}', axis_length=0.03)
            pose = T @ pose
            xyzw = R.from_matrix(pose[0:3,0:3]).as_quat()
            quat = rr.Quaternion.identity()
            quat.xyzw = xyzw
            # rr.log(f'{self.robot_name}/{link_name}', rr.Transform3D(translation=pose[:3,-1].squeeze(), rotation=quat))
            
    def updateScene(self, points, radii = 0.001, color = [1., 1., 1.]):
        self.visualizer.logPoints(points, radii=radii, colors=color, log_path=f'{self.robot_name}/scene_points')

    def registerMeshAsset(self, asset_mesh_path, asset_pose):
        asset_name = asset_mesh_path.split('/')[-1].split('.')[0]
        if asset_name not in self.mesh_assets:
            self.visualizer.logMeshFile(asset_mesh_path, asset_pose, f'{self.robot_name}/mesh_assets/{asset_name}')
            xyzw = R.from_matrix(asset_pose[0:3,0:3]).as_quat()
            quat = rr.Quaternion.identity()
            quat.xyzw = xyzw
            rr.log(f'{self.robot_name}/mesh_assets/{asset_name}', rr.Transform3D(translation=asset_pose[:3,-1].squeeze(), rotation=quat))
            self.mesh_assets.append(asset_name)
        else:
            raise ValueError(f'Asset {asset_name} already registered')

    def updateMeshAsset(self, asset_name, asset_pose):
        if asset_name not in self.mesh_assets:
            raise ValueError(f'Asset {asset_name} not registered')
        xyzw = R.from_matrix(asset_pose[0:3,0:3]).as_quat()
        quat = rr.Quaternion.identity()
        quat.xyzw = xyzw
        rr.log(f'{self.robot_name}/mesh_assets/{asset_name}', rr.Transform3D(translation=asset_pose[:3,-1].squeeze(), rotation=quat))

    def registerPointCloudAsset(self, asset_points, asset_pose, asset_name, radii = 0.001, color = [1., 1., 1.]):
        if asset_name not in self.pcd_assets:
            self.visualizer.logPoints(asset_points, radii=radii, colors=color, log_path=f'{self.robot_name}/pcd_assets/{asset_name}')
            xyzw = R.from_matrix(asset_pose[0:3,0:3]).as_quat()
            quat = rr.Quaternion.identity()
            quat.xyzw = xyzw
            rr.log(f'{self.robot_name}/pcd_assets/{asset_name}', rr.Transform3D(translation=asset_pose[:3,-1].squeeze(), rotation=quat))
            self.pcd_assets.append(asset_name)
        else:
            raise ValueError(f'Asset {asset_name} already registered')
        
    def updatePointCloudAsset(self, asset_name, asset_pose):
        if asset_name not in self.pcd_assets:
            raise ValueError(f'Asset {asset_name} not registered')
        xyzw = R.from_matrix(asset_pose[0:3,0:3]).as_quat()
        quat = rr.Quaternion.identity()
        quat.xyzw = xyzw
        rr.log(f'{self.robot_name}/pcd_assets/{asset_name}', rr.Transform3D(translation=asset_pose[:3,-1].squeeze(), rotation=quat))