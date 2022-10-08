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
from typing import Tuple, List, Union, Set

_epsilon = 1e-16


def o3d_to_trimesh(o3d_m, process=True) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.asarray(o3d_m.vertices), faces=np.asarray(o3d_m.triangles), process=process)


def get_vv_adj_list(fs, nv: int = None) -> List[List[int]]:
    """
    To avoid memory leak in igl.adjacency_list, but 3x slower.
    """
    import open3d as o3d
    if nv is None:
        nv = fs.max() + 1
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.zeros((nv, 3))),
                                  o3d.utility.Vector3iVector(fs)).compute_adjacency_list()
    vv_adj = [list(adj) for adj in m.adjacency_list]
    return vv_adj


def get_vertex_neighborhood(fs: np.ndarray, vids: np.ndarray, order: int = 1):
    """
    Given vertex ids, get the neighborhood of the vertices. The neighborhood does not include the vertex itself.
    """
    vv_adj = get_vv_adj_list(fs)
    visited = np.zeros(len(vv_adj), dtype=bool)
    visited[vids] = True
    nei_vids = np.unique(np.concatenate([vv_adj[i] for i in vids]))
    for _ in range(1, order):
        cur_vids = nei_vids[~visited[nei_vids]]
        if len(cur_vids) == 0:
            break
        nei_vids = np.union1d(np.concatenate([vv_adj[i] for i in cur_vids]), nei_vids)
        visited[cur_vids] = True

    nei_vids = np.setdiff1d(nei_vids, vids)
    return nei_vids


def get_fids_from_vids(fs: np.ndarray, vids: np.ndarray):
    selected = np.zeros(fs.shape[0], dtype=bool)
    selected[vids] = True
    return np.where(np.all(selected[fs], axis=1))[0]


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


def fan_triangulation(vids: np.ndarray) -> np.ndarray:
    """
        3  2  1
        *  *  *
        \  |  /
         \ | /
           *
           0
    faces: [[0, 1, 2], [0, 2, 3], ...]
    :param vids: (n, ), int, the first of which is the start vertex.
    """
    svid = vids[0]
    fids = np.zeros((len(vids) - 2, 3), dtype=int)
    fids[:, 0] = svid
    fids[:, 1] = vids[1:-1]
    fids[:, 2] = vids[2:]
    return fids


def close_holes(vs: np.ndarray, fs: np.ndarray, hole_len_thr: float = 10000.) -> np.ndarray:
    """
    Close holes whose length is less than a given threshold.
    :param edge_len_thr:
    :return:
        out_fs: output faces
    """
    out_fs = fs.copy()
    while True:
        updated = False
        for b in igl.all_boundary_loop(out_fs):
            hole_edge_len = np.linalg.norm(vs[np.roll(b, -1)] - vs[b], axis=1).sum()
            if len(b) >= 3 and hole_edge_len <= hole_len_thr:
                out_fs = close_hole(vs, out_fs, b)
                updated = True

        if not updated:
            break

    return out_fs


def close_hole(vs: np.ndarray, fs: np.ndarray, hole_vids) -> np.ndarray:
    """
    :param hole_vids: the vid sequence
    :return:
        out_fs:
    """
    hole_vids = np.array(hole_vids)
    if len(hole_vids) < 3:
        return fs.copy()

    if len(hole_vids) == 3:
        # fill one triangle
        out_fs = np.concatenate([fs, hole_vids[::-1][None]], axis=0)
        return out_fs

    # heuristically divide the hole
    queue = [hole_vids[::-1]]
    out_fs = []
    while len(queue) > 0:
        cur_vids = queue.pop(0)
        if len(cur_vids) == 3:
            out_fs.append(cur_vids)
            continue

        # current hole
        hole_edge_len = np.linalg.norm(vs[np.roll(cur_vids, -1)] - vs[cur_vids], axis=1)
        hole_len = np.sum(hole_edge_len)
        min_concave_degree = np.inf
        tar_i, tar_j = -1, -1
        for i in range(len(cur_vids)):
            eu_dists = np.linalg.norm(vs[cur_vids[i]] - vs[cur_vids], axis=1)
            geo_dists = np.roll(np.roll(hole_edge_len, -i).cumsum(), i)
            geo_dists = np.roll(np.minimum(geo_dists, hole_len - geo_dists), 1)
            concave_degree = eu_dists / (geo_dists ** 2 + _epsilon)

            _idx = 1
            j = np.argsort(concave_degree)[_idx]
            while min((j + len(cur_vids) - i) % len(cur_vids), (i + len(cur_vids) - j) % len(cur_vids)) <= 1:
                _idx += 1
                j = np.argsort(concave_degree)[_idx]

            if concave_degree[j] < min_concave_degree:
                min_concave_degree = concave_degree[j]
                tar_i, tar_j = min(i, j), max(i, j)

        queue.append(cur_vids[tar_i:tar_j + 1])
        queue.append(np.concatenate([cur_vids[tar_j:], cur_vids[:tar_i + 1]]))

    out_fs = np.concatenate([fs, np.array(out_fs)], axis=0)
    return out_fs


def uniform_laplacian_matrix(fs: np.ndarray, nv: int, normalize: bool = False, k: int = 1) -> scipy.sparse.csr_matrix:
    """
    L = D - A. (positive diagonal)
    """
    A = igl.adjacency_matrix(fs)
    A.resize((nv, nv))
    L = scipy.sparse.diags(A.sum(1).A1) - A
    if normalize:
        L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L

    if k > 1:
        L0 = L.copy()
        for _ in range(k - 1):
            L = L @ L0
        if normalize:
            L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L
    return L


def cot_laplacian_matrix(vs: np.ndarray, fs: np.ndarray, normalize: bool = False,
                         k: int = 1) -> scipy.sparse.csr_matrix:
    L = scipy.sparse.csr_matrix((vs.shape[0], vs.shape[0]))
    L[:] = -igl.cotmatrix(vs, fs)
    if np.any(np.isnan(L.data)):
        L.data[np.isnan(L.data)] = 0.
    if np.any(np.isinf(L.data)):
        L.data[np.isinf(L.data)] = 1e6
    if not np.allclose(L.sum(1), 0.):
        L = L - scipy.sparse.diags(L.sum(1).A1)

    if normalize:
        L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L

    if k > 1:
        L0 = L.copy()
        for _ in range(k - 1):
            L = L @ L0
        if normalize:
            L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L
    return L


def laplacian_smooth(vs: np.ndarray, fs: np.ndarray, lambs: Union[np.ndarray, float] = 1., n_iter: int = 1,
                     cot: bool = True, implicit: bool = False, boundary_preserve=True, k: int = 1) -> np.ndarray:
    """
    out_vs = (I + lambs * L)^n_iter * vs, where L is the normalized laplacian matrix (negative diagonal).
    :param lambs: Lambdas in the range [0., 1.]. The vertices will not change if the corresponding lambdas are 0.
    :param cot: If using the cotangent weights.
    :param implicit: If using implicit laplacian smooth. If true, solve (I - lambs * L)^n_iter * out_vs = vs,
            where L is the normalized laplacian matrix (negative diagonal).
    :param boundary_preserve: If preserving the boundary vertices.
    :param k: The order of the laplacian.
    """
    if cot:
        L = -cot_laplacian_matrix(vs, fs, normalize=True, k=k)
    else:
        L = -uniform_laplacian_matrix(fs, len(vs), normalize=True, k=k)

    if boundary_preserve:
        b = igl.all_boundary_loop(fs)
        if len(b) > 0:
            b = np.concatenate(b)
            if isinstance(lambs, float):
                lambs = np.ones(len(vs)) * lambs
            lambs[b] = 0.

    return solve_laplacian_smooth(vs, L, lambs, n_iter, implicit)


def laplacian_smooth_selected(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, lamb: float = 1., n_iter: int = 1,
                              cot: bool = True, implicit: bool = False, boundary_preserve=True,
                              boundary_smoothing=True, k: int = 1) -> np.ndarray:
    """
    Do laplacian smoothing on the selected vertices.
    """
    lambs = np.zeros(len(vs))
    lambs[vids] = lamb

    out_vs = vs.copy()
    if not boundary_preserve and boundary_smoothing:
        # first smoothing boundary
        out_vs = smooth_boundary_part(out_vs, fs, vids, lamb, n_iter, implicit)
        boundary_preserve = True

    if k == 1:
        vid_selected = np.zeros(len(vs), dtype=bool)
        vid_selected[vids] = 1
        sub_fids = np.where(np.any(vid_selected[fs], axis=1))[0]
    else:
        nei_vids = get_vertex_neighborhood(fs, vids, order=k)
        nei_vids = np.concatenate([nei_vids, vids])
        sub_fids = get_fids_from_vids(fs, nei_vids)
    sub_vs, sub_fs, IM, sub_vids = igl.remove_unreferenced(out_vs, fs[sub_fids])

    out_vs[sub_vids] = laplacian_smooth(sub_vs, sub_fs, lambs[sub_vids], n_iter, cot, implicit, boundary_preserve, k)

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
    vv_adj = get_vv_adj_list(fs, len(vs))
    for boundary in igl.all_boundary_loop(fs):
        if vids_included is not None and not np.any(np.in1d(boundary, vids_included)):
            continue

        pts = out_vs[boundary]
        _vids = list(range(len(pts)))
        if boundary[-1] in vv_adj[boundary[0]]:
            _vids += [0]

        if len(_vids) > 3:
            out_vs[boundary] = smooth_line(pts, _vids, lamb, n_iter, implicit)

    return out_vs


def smooth_boundary_part(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, lamb: float = 1., n_iter: int = 1,
                         implicit: bool = False) -> np.ndarray:
    lambs = np.zeros(len(vs))
    lambs[vids] = lamb
    out_vs = vs.copy()

    vv_adj = get_vv_adj_list(fs, len(vs))
    for boundary in igl.all_boundary_loop(fs):
        if not np.any(np.in1d(boundary, vids)):
            continue

        pts = out_vs[boundary]
        _vids = list(range(len(pts)))
        if boundary[-1] in vv_adj[boundary[0]]:
            _vids += [0]

        if len(_vids) > 3:
            out_vs[boundary] = smooth_line(pts, _vids, lambs[boundary], n_iter, implicit)

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


def remove_non_max_face_components(vs, fs):
    ls = igl.face_components(fs)
    cs = np.bincount(ls)
    if len(cs) == 1:
        return vs, fs

    out_vs, out_fs, _, _ = igl.remove_unreferenced(vs, fs[ls == np.argmax(cs)])
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


def filter_mhb(v_attrs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Filter with manifold harmonics basis (MHB).
    :param v_attrs:
    :param basis: (n, m), m basis vectors
    :return:
    """
    shape = v_attrs.shape
    if len(v_attrs.shape) == 1:
        v_attrs = v_attrs[:, None]  # (n, 1)

    out = np.zeros_like(v_attrs)
    for i in range(v_attrs.shape[1]):
        out[:, i] = np.sum((basis.T @ v_attrs[:, i]) * basis, axis=1)

    out = out.reshape(shape)
    return out


def manifold_harmonics_basis(vs: np.ndarray, fs: np.ndarray, cot: bool = True):
    """
    Calculate the manifold harmonics basis (MHB) of a mesh.
    :return:
        w: frequencies of bases
        v: basis vectors
    """
    if cot:
        L = cot_laplacian_matrix(vs, fs, normalize=False).toarray()
    else:
        L = uniform_laplacian_matrix(fs, len(vs), normalize=False).toarray()

    w, v = manifold_harmonics_basis_laplacian(L)
    return w, v


def manifold_harmonics_basis_laplacian(L: np.ndarray):
    w, v = np.linalg.eigh(L)
    return w, v
