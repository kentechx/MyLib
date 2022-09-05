# Codes in this file is implemented under copyleft licences.
import numpy as np
import igl
from typing import Tuple, List


class PymeshlabHelper:

    @staticmethod
    def laplacian_smooth(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray = None, stepsmoothnum: int = 3,
                         cotangentweight: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param m: mesh
        :param vids: target vertex indices to smooth.
        :param stepsmoothnum:
        :param cotangenent: True or False
        :return:
        """
        import pymeshlab
        if vids is not None:
            v_scalar_array = np.zeros(len(vs), dtype=int)
            v_scalar_array[vids] = 1
        else:
            v_scalar_array = np.ones(len(vs), dtype=int)
        mesh = pymeshlab.Mesh(vs, fs, v_scalar_array=v_scalar_array)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        ms.compute_selection_by_condition_per_vertex(condselect='q>0')
        ms.compute_selection_transfer_vertex_to_face()
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum, cotangentweight=cotangentweight, selected=True)

        m = ms.current_mesh()
        vs = m.vertex_matrix()
        fs = m.face_matrix()

        return vs, fs

    @staticmethod
    def laplacian_smooth_boundary_preserving(vs: np.ndarray, fs: np.ndarray, stepsmoothnum: int,
                                             cotangentweight: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        import pymeshlab
        mesh = pymeshlab.Mesh(vs, fs)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)

        ms.compute_selection_from_mesh_border()
        ms.apply_selection_inverse(invfaces=True, invverts=True)
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum, cotangentweight=cotangentweight, selected=True)

        m = ms.current_mesh()
        vs = m.vertex_matrix()
        fs = m.face_matrix()
        return vs, fs

    @staticmethod
    def taubin_smooth(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray = None, lambda_=0.5, mu=-0.53,
                      stepsmoothnum: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        import pymeshlab
        if vids is not None:
            v_scalar_array = np.zeros(len(vs), dtype=int)
            v_scalar_array[vids] = 1
        else:
            v_scalar_array = np.ones(len(vs), dtype=int)
        mesh = pymeshlab.Mesh(vs, fs, v_scalar_array=v_scalar_array)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        ms.compute_selection_by_condition_per_vertex(condselect='q>0')
        ms.compute_selection_transfer_vertex_to_face()
        ms.apply_coord_taubin_smoothing(lambda_=lambda_, mu=mu, stepsmoothnum=stepsmoothnum, selected=True)

        m = ms.current_mesh()
        vs = m.vertex_matrix()
        fs = m.face_matrix()
        return vs, fs

    @staticmethod
    def close_holes_pymeshlab(vs: np.ndarray,
                              fs: np.ndarray,
                              maxholesize=30,
                              selfintersection: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        import pymeshlab
        mesh = pymeshlab.Mesh(vs, fs)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)
        # ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes(maxholesize=maxholesize, selfintersection=selfintersection)

        m = ms.current_mesh()
        vs = m.vertex_matrix()
        fs = m.face_matrix()
        return vs, fs


class CGAL_Helper:

    @staticmethod
    def to_cgal_polygon_mesh(vs: np.ndarray, fs: np.ndarray):
        '''trimesh2cgal.
        :param mesh: trimesh.Trimesh
        '''
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_modifier, Polyhedron_3
        from CGAL.CGAL_Kernel import Point_3

        num_vertices = len(vs)
        num_faces = len(fs)

        m = Polyhedron_modifier()

        m.begin_surface(num_vertices, num_faces)
        for i in range(num_vertices):
            m.add_vertex(Point_3(vs[i][0], vs[i][1], vs[i][2]))
        for i in range(num_faces):
            m.begin_facet()
            m.add_vertex_to_facet(int(fs[i][0]))
            m.add_vertex_to_facet(int(fs[i][1]))
            m.add_vertex_to_facet(int(fs[i][2]))
            m.end_facet()

        P = Polyhedron_3()
        P.delegate(m)
        return P

    @staticmethod
    def from_cgal_polygon_mesh(P) -> Tuple[np.ndarray, np.ndarray]:
        # 获得vertices
        vertices = []
        for point in P.points():
            vertice = (point.x(), point.y(), point.z())
            vertices.append(vertice)

        vertices_tuple = tuple(vertices)
        # 构建dict,用于后面的查找。用list index查找太慢了
        dic_vertice_index = {vertice: index for index, vertice in enumerate(vertices_tuple)}
        # faces
        faces = []
        for facet in P.facets():
            face = []
            start = facet.facet_begin()
            stop = facet.facet_begin().deepcopy()

            while True:
                halfedge_facet = start.next()
                point_halfedge = halfedge_facet.vertex().point()
                vertice_point = (point_halfedge.x(), point_halfedge.y(), point_halfedge.z())
                index_vertice = dic_vertice_index[vertice_point]
                face.append(index_vertice)
                if (start.__eq__(stop)):
                    break
            faces.append(face)
        return np.asarray(vertices_tuple), np.asarray(faces)

    @staticmethod
    def fill_holes(vs: np.ndarray, fs: np.ndarray, density_control_factor=None) -> Tuple[np.ndarray, np.ndarray]:
        from CGAL import CGAL_Polygon_mesh_processing
        P = CGAL_Helper.to_cgal_polygon_mesh(vs, fs)
        for h in P.halfedges():
            if h.is_border():
                outf = []
                outv = []
                if density_control_factor:
                    CGAL_Polygon_mesh_processing.triangulate_and_refine_hole(P, h, outf, outv, density_control_factor)
                else:
                    CGAL_Polygon_mesh_processing.triangulate_and_refine_hole(P, h, outf, outv)

        return CGAL_Helper.from_cgal_polygon_mesh(P)


class PygmshHelper:

    @staticmethod
    def close_holes(in_vs: np.ndarray, in_fs: np.ndarray, boundary_loops: List[int] = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Close holes indicated by boundary loops by generate surfaces. The added vertices and faces will be appended
        into the original vertices and faces.
        :return:
            out_vs:
            out_fs:
        """
        import pygmsh
        if boundary_loops is None:
            boundary_loops = igl.all_boundary_loop(in_fs)

        if len(boundary_loops) == 0:
            return in_vs, in_vs

        # generate surfaces
        out_vs = np.copy(in_vs)
        out_fs = np.copy(in_fs)
        for b in boundary_loops:
            _vs = in_vs[b]
            with pygmsh.geo.Geometry() as geom:
                geom.add_polygon(_vs, mesh_size=10000.)  # keep the boundary loop identity
                mesh = geom.generate_mesh()

            assert len(b) == len(igl.boundary_loop(np.asarray(mesh.cells_dict['triangle'], dtype='i4')))

            # replace vids
            _fs = mesh.cells_dict['triangle'][:, [0, 2, 1]]  # flip faces
            b_locs = [np.where(_fs == i) for i in range(len(b))]  # the first n vertices
            added_v_loc = np.where(_fs >= len(b))
            for i, loc in enumerate(b_locs):
                _fs[loc] = b[i]

            # append vertices & faces
            if len(mesh.points) > len(b):
                assert len(added_v_loc[0]) > 0
                _fs[added_v_loc] += len(out_vs) - len(b)
                out_vs = np.r_[out_vs, mesh.points[len(b):]]

            out_fs = np.r_[out_fs, _fs]

        return out_vs, out_fs
