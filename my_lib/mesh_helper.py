import copy
import numpy as np
import trimesh

try:
    import open3d as o3d
except:
    pass

from .configs import *


class MeshHelper:

    @staticmethod
    def create_triangle_soup(vs: np.ndarray, fs: np.ndarray):
        new_vs = vs[fs].reshape(-1, 3)
        new_fs = np.arange(len(new_vs)).reshape(-1, 3)
        return new_vs, new_fs

    @staticmethod
    def create_face_colored_mesh(m: trimesh.Trimesh, labels: np.ndarray = None):
        vs, fs = MeshHelper.create_triangle_soup(np.asarray(m.vertices), np.asarray(m.faces))

        # O3D MESH
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vs),
                                      o3d.utility.Vector3iVector(fs))
        m.compute_vertex_normals()
        if labels is not None:
            v_colors = np.full((len(vs), 3), 0.7, dtype=np.float32)
            for l in np.unique(labels):
                if l == 0:
                    continue
                if l in color_map:
                    idx = np.where(labels == l)[0]
                    v_colors[fs[idx].reshape(-1)] = np.array(color_map[l]) / 255.

            m.vertex_colors = o3d.utility.Vector3dVector(v_colors)
        return m

    @staticmethod
    def visualize(m: trimesh.Trimesh, labels: np.ndarray = None):
        m = MeshHelper.create_face_colored_mesh(m, labels)
        o3d.visualization.draw_geometries([m])

    @staticmethod
    def visualize_switch(m1: trimesh.Trimesh, labels1: np.ndarray = None, m2: trimesh.Trimesh = None,
                         labels2: np.ndarray = None):
        if m2 is None:
            m2 = m1
        m1 = MeshHelper.create_face_colored_mesh(m1, labels1)
        m2 = MeshHelper.create_face_colored_mesh(m2, labels2)

        meshes = [m1, m2]
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        i = 0
        vis.add_geometry(meshes[i])
        vis.update_geometry(meshes[i])

        def switch(vis: o3d.visualization.VisualizerWithKeyCallback):
            nonlocal i, meshes
            vis.remove_geometry(meshes[i], False)
            i = 1 - i
            vis.add_geometry(meshes[i], False)
            vis.update_geometry(meshes[i])

        vis.register_key_callback(ord(" "), switch)
        vis.run()
        vis.destroy_window()
