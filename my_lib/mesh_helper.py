import copy
import numpy as np
import trimesh
import trimesh.parent
from typing import List, Tuple, Union

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
    def create_face_colored_mesh(m: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh], labels: np.ndarray = None):
        # O3D MESH
        if isinstance(m, trimesh.Trimesh):
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                          o3d.utility.Vector3iVector(m.faces))
        else:
            m = o3d.geometry.TriangleMesh(m)

        vs, fs = MeshHelper.create_triangle_soup(np.asarray(m.vertices), np.asarray(m.triangles))
        m.vertices = o3d.utility.Vector3dVector(vs)
        m.triangles = o3d.utility.Vector3iVector(fs)

        m.compute_vertex_normals()
        if labels is not None:
            v_colors = np.full((len(vs), 3), 0.7, dtype=np.float32)
            for l in np.unique(labels):
                if l == 0:
                    continue
                if l in color_map:
                    idx = np.where(labels == l)[0]
                    v_colors[fs[idx].reshape(-1)] = np.array(color_map[l]) / 255.
                else:
                    idx = np.where(labels == l)[0]
                    v_colors[fs[idx].reshape(-1)] = np.array(list(color_map.values())[l % len(color_map)]) / 255.

            m.vertex_colors = o3d.utility.Vector3dVector(v_colors)
        return m

    @staticmethod
    def visualize(m: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh], labels: np.ndarray = None):
        m = MeshHelper.create_face_colored_mesh(m, labels)
        o3d.visualization.draw_geometries([m])

    @staticmethod
    def visualize_faces(m: trimesh.Trimesh, fids: np.ndarray = None):
        ls = np.zeros(len(m.faces), dtype=int)
        if fids is not None and len(fids) > 0:
            ls[fids] = 1
        m = MeshHelper.create_face_colored_mesh(m, ls)
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

    @staticmethod
    def visualize_geos(geos: List[Union[trimesh.parent.Geometry3D, o3d.geometry.Geometry3D]]):
        o3d_geos = []
        for g in geos:
            if isinstance(g, trimesh.Trimesh):
                m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g.vertices),
                                              o3d.utility.Vector3iVector(g.faces))
                m.compute_vertex_normals()
                o3d_geos.append(m)
            elif isinstance(g, trimesh.PointCloud):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(g.vertices)))
                o3d_geos.append(pcd)
            elif isinstance(g, trimesh.path.Path3D):
                m = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(g.vertices)),
                                         o3d.utility.Vector2iVector(np.array(g.vertex_nodes)))
                o3d_geos.append(m)
            elif isinstance(g, o3d.geometry.Geometry3D):
                o3d_geos.append(g)
            else:
                raise ValueError("Unsupported geometry type: {}".format(type(g)))

        o3d.visualization.draw_geometries(o3d_geos)
