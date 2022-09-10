"""
@author: shenkaidi
@date: 20220901
"""

import numpy as np
import trimesh
import igl
import scipy
import scipy.sparse
import scipy.sparse.linalg
from typing import Tuple, List, Union

_epsilon = 1e-7


def o3d_to_trimesh(o3d_m, process=True) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(o3d_m.vertices), faces=np.asarray(o3d_m.triangles), process=process)


def grid_triangulation(vids: np.ndarray) -> np.ndarray:
    """
    Do grid triangulation for a set of vertices. For each grid cell, we use two triangles to represent it as follows:
    0---1---...---m-1
    |\  |
    | \ |
    |__\|
    m  m+1  ... m+m-1
    In the cell above, we triangulate it as (0, m, m+1) and (0, m+1, 1).
    :param vids: vertex ids with shape (n, m), where n is the number of rows, m is the number of columns.
    :return:
        faces: np.ndarray, int, (2 * (n-1) * (m-1), 3)
    """
    idx = np.arange(vids.shape[0] * vids.shape[1]).reshape((-1, vids.shape[1]))
    faces = np.zeros(((idx.shape[0] - 1) * (idx.shape[1] - 1), 6), dtype=int)

    faces[:, 0] = idx[:-1, :-1].reshape(-1)
    faces[:, 1] = faces[:, 0] + idx.shape[1]
    faces[:, 2] = faces[:, 1] + 1
    faces[:, 3] = faces[:, 0]
    faces[:, 4] = faces[:, 3] + idx.shape[1] + 1
    faces[:, 5] = faces[:, 3] + 1
    faces[:] = vids.reshape(-1)[faces]
    faces = faces.reshape((-1, 3))

    return faces


def laplacian_smooth(vs: np.ndarray, fs: np.ndarray, lambs: Union[np.ndarray, float] = 1., n_iter: int = 1,
                     cot: bool = True, implicit: bool = False, boundary_preserve=True) -> np.ndarray:
    """
    out_vs = (I + lambs * L)^n_iter * vs, where L is the normalized laplacian matrix (negative diagonal).
    :param lambs: Lambdas in the range [0., 1.]. The vertices will not change if the corresponding lambdas are 0.
    :param cot: If using the cotangent weights.
    :param implicit: If using implicit laplacian smooth. If true, solve (I - lambs * L)^n_iter * out_vs = vs,
            where L is the normalized laplacian matrix (negative diagonal).
    """
    if cot:
        L = igl.cotmatrix(vs, fs)
        L = -scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L
    else:
        L = igl.adjacency_matrix(fs)
        L = scipy.sparse.diags(-L.sum(1).A1) + L
        L = -scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L

    if boundary_preserve:
        b = igl.all_boundary_loop(fs)
        if len(b) > 0:
            b = np.concatenate(b)
            if isinstance(lambs, float):
                lambs = np.ones(len(vs)) * lambs
            lambs[b] = 0.

    return solve_laplacian_smooth(vs, L, lambs, n_iter, implicit)


def laplacian_smooth_selected(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, lamb: float = 1., n_iter: int = 1,
                              cot: bool = True, implicit: bool = False, boundary_preserve=True) -> np.ndarray:
    """
    Do laplacian smoothing on the selected vertices.
    """
    lambs = np.zeros(len(vs))
    lambs[vids] = lamb

    vid_selected = np.zeros(len(vs), dtype=bool)
    vid_selected[vids] = 1
    sub_fids = np.where(np.any(vid_selected[fs], axis=1))[0]
    sub_vs, sub_fs, IM, sub_vids = igl.remove_unreferenced(vs, fs[sub_fids])

    out_vs = vs.copy()
    out_vs[sub_vids] = laplacian_smooth(sub_vs, sub_fs, lambs[sub_vids], n_iter, cot, implicit, boundary_preserve)

    return out_vs


def smooth_boundary(vs: np.ndarray, fs: np.ndarray, vids_included: List = None, lamb: float = 1., n_iter: int = 1,
                    implicit: bool = False) -> np.ndarray:
    """
    Smooth mesh boundaries.
    :param vids_included: indicate which boundary are included in the smoothing.
    :return:
        out_vs: out vertices
    """
    if vids_included is not None:
        vids_included = np.asarray(vids_included)
    out_vs = vs.copy()
    vv_adj = igl.adjacency_list(fs)
    for boundary in igl.all_boundary_loop(fs):
        if vids_included is not None and not np.any(np.in1d(boundary, vids_included)):
            continue

        pts = out_vs[boundary]
        _vids = list(range(len(pts)))
        if boundary[-1] in vv_adj[0]:
            _vids += [0]

        if len(_vids) > 3:
            out_vs[boundary] = smooth_line(pts, _vids, lamb, n_iter, implicit)

    return out_vs


def smooth_line(points: np.ndarray, pid_seq: List[int], lambs: Union[np.ndarray, float] = 1., n_iter: int = 1,
                implicit: bool = False) -> np.ndarray:
    """
    :param points: point coordinates, (n, 3), where n is the number of points.
    :param pid_seq: point id sequence
    :return:
        out_points: np.ndarray, (n, 3)
    """
    row = np.concatenate([pid_seq[:-1], pid_seq[1:]])
    col = np.concatenate([pid_seq[1:], pid_seq[:-1]])
    data = np.full(len(row), -0.5)
    L = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(points), len(points)))
    L = L + scipy.sparse.eye(len(points))
    L = -scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L

    if pid_seq[-1] != pid_seq[0]:
        # if the point sequence is not a loop, then fix the end points
        if isinstance(lambs, float):
            lambs = np.full(len(points), lambs)
        lambs[pid_seq[0]] = 0.
        lambs[pid_seq[-1]] = 0.
    return solve_laplacian_smooth(points, L, lambs, n_iter, implicit)


def solve_laplacian_smooth(v_attrs: np.ndarray, L: scipy.sparse.csr_matrix,
                           lambs: Union[np.ndarray, float] = 1.,
                           n_iter: int = 1,
                           implicit: bool = False) -> np.ndarray:
    """
    :param v_attrs: vertex attributes, (n, _), where n is the number of vertices.
    :param L: laplacian matrix, (n, n), negative diagonal.
    :param lambs: per vertex smoothing factor.
    :return:
    """
    if isinstance(lambs, float):
        # np csc matrix
        lambs = lambs * np.ones(len(v_attrs))

    assert len(v_attrs) == len(lambs)

    if implicit:
        lambs = scipy.sparse.diags(lambs)
        AA = scipy.sparse.eye(len(v_attrs)) - lambs @ L

    for _ in range(n_iter):
        if not implicit:
            # explicit
            v_attrs = v_attrs + lambs[:, None] * (L @ v_attrs)
        else:
            v_attrs = scipy.sparse.linalg.spsolve(AA, v_attrs)

    return v_attrs


def remove_low_valence_faces(vs: np.ndarray, fs: np.ndarray, remove_unreferenced: bool = True) -> [np.ndarray,
                                                                                                   np.ndarray]:
    """
    Remove faces whose valences <= 1, i.e. at least two edges of the face are border edges.
    :return:
        out_vs:
        out_fs:
    """
    # remove faces whose all vertices are on the boundary
    out_vs = vs.copy()
    out_fs = fs.copy()
    is_boundary = np.zeros(len(out_vs), dtype=np.bool)
    b = igl.all_boundary_loop(out_fs)
    if len(b) == 0:
        return out_vs, out_fs

    b_vids = np.concatenate(b)
    if len(b_vids) == 0:
        return out_vs, out_fs

    is_boundary[b_vids] = True
    invalid_fids = np.where(np.all(is_boundary[out_fs], axis=1))[0]
    if len(invalid_fids) > 0:
        out_fs = out_fs[np.setdiff1d(np.arange(len(out_fs)), invalid_fids)]

    if remove_unreferenced:
        out_vs, out_fs, _, _ = igl.remove_unreferenced(out_vs, out_fs)

    return out_vs, out_fs


def remove_non_manifold(vs: np.ndarray, fs: np.ndarray):
    import open3d as o3d
    vs = vs.copy()
    fs = fs.copy()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vs),
                                  o3d.utility.Vector3iVector(fs))

    # m: o3d.geometry.TriangleMesh = m.remove_duplicated_vertices()
    # m: o3d.geometry.TriangleMesh = m.remove_duplicated_triangles()
    # m: o3d.geometry.TriangleMesh = m.remove_degenerate_triangles()
    m: o3d.geometry.TriangleMesh = m.remove_non_manifold_edges()
    m: o3d.geometry.TriangleMesh = m.remove_unreferenced_vertices()

    vids = m.get_non_manifold_vertices()
    for _ in range(20):
        if len(vids) == 0:
            break

        m.remove_vertices_by_index(vids)
        vids = m.get_non_manifold_vertices()

    # orient triangles
    out_fs, c = igl.bfs_orient(np.asarray(m.triangles))
    return np.asarray(m.vertices), out_fs


def deform_arap_igl(vertices: np.ndarray, faces: np.ndarray, handle_id: np.ndarray, handle_co: np.ndarray):
    """
    Deform a mesh via ARAP(as rigid as possible) implemented by libigl.
    :param vertices: np.ndarray, float, (_, 3)
    :param faces: np.ndarray, int, (_, 3)
    :param handle_id: np.ndarray, int, (_, 3)
    :param handle_co: np.ndarray, float, (_, 3). The target coordinates of handles.
    :return: out_vertices: same type and shape as `vertices`
    """
    arap = igl.ARAP(vertices, faces, 3, handle_id)
    out_vertices = arap.solve(handle_co, vertices)
    return out_vertices
