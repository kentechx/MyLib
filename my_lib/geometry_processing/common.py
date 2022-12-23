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


def get_all_boundary_vids(fs):
    bs = igl.all_boundary_loop(fs)
    if len(bs) == 0:
        return np.array([], dtype=int)
    return np.concatenate(bs)


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


def get_vv_adj_list_python(fs, nv: int = None) -> List[List[int]]:
    if nv is None:
        nv = fs.max() + 1

    vv_adj = [set() for _ in range(nv)]
    for f in fs:
        vv_adj[f[0]].add(f[1])
        vv_adj[f[0]].add(f[2])
        vv_adj[f[1]].add(f[0])
        vv_adj[f[1]].add(f[2])
        vv_adj[f[2]].add(f[0])
        vv_adj[f[2]].add(f[1])
    return [list(i) for i in vv_adj]


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


def get_mollified_edge_length(vs: np.ndarray, fs: np.ndarray, mollify_factor=1e-5) -> np.ndarray:
    lin = igl.edge_lengths(vs, fs)
    if mollify_factor == 0:
        return lin
    delta = mollify_factor * np.mean(lin)
    eps = np.maximum(0, delta - lin[:, 0] - lin[:, 1] + lin[:, 2])
    eps = np.maximum(eps, delta - lin[:, 0] - lin[:, 2] + lin[:, 1])
    eps = np.maximum(eps, delta - lin[:, 1] - lin[:, 2] + lin[:, 0])
    eps = eps.max()
    lin += eps
    return lin


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
                         k: int = 1, mollify_factor=1e-5) -> scipy.sparse.csr_matrix:
    """ The off-diagnoals are $-1/2 * (\cot \alpha_{ij} + \cot \beta_{ij})$ """
    l = get_mollified_edge_length(vs, fs, mollify_factor)
    L = -igl.cotmatrix_intrinsic(l, fs).asformat('csr')

    if normalize:
        L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L

    if k > 1:
        L0 = L.copy()
        for _ in range(k - 1):
            L = L @ L0
        if normalize:
            L = scipy.sparse.diags(1. / (L.diagonal() + _epsilon)) @ L
    return L


def robust_laplacian(vs, fs, mollify_factor=1e-5, delaunay=True) -> \
        Tuple[scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
    """
    Get a laplcian with iDT (intrinsic Delaunay triangulation) and intrinsic mollification.
    Ref https://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf
    :param mollify_factor: the mollification factor.
    """
    lin = get_mollified_edge_length(vs, fs, mollify_factor)
    fin = fs
    if delaunay:
        lin, fin = igl.intrinsic_delaunay_triangulation(lin, fs)
    L = -igl.cotmatrix_intrinsic(lin, fin)
    M = igl.massmatrix_intrinsic(lin, fin, igl.MASSMATRIX_TYPE_VORONOI)
    return L, M


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
                implicit: bool = False, k=1) -> np.ndarray:
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
    L = L + scipy.sparse.diags(-L.sum(1).A1)
    if k > 1:
        L0 = L.copy()
        for _ in range(k - 1):
            L = L @ L0
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


def deform_harmonic(vs: np.ndarray, fs: np.ndarray, handle_id: np.ndarray, handle_co: np.ndarray, k=2):
    # L = -cot_laplacian_matrix(vs, fs, tol=100.)
    # M = igl.massmatrix(vs, fs).asformat('csr')
    L, M = robust_laplacian(vs, fs, delaunay=False)
    L = -L
    b = handle_id.astype('i4')
    dp_b = handle_co - vs[b]  # (n, 3)
    dp = igl.harmonic_weights_from_laplacian_and_mass(L, M, b, dp_b, k)
    return dp + vs


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


def mesh_fair_harmonic_global(vs: np.ndarray, fs: np.ndarray, bvids: np.ndarray, cot: bool = True, k: int = 1,
                              boundary_preserve: bool = True):
    """
    Do mesh fairing globally via minimizing the Dirichlet energy. This equals to solve the equation LX=0,
    where L is the laplacian matrix with order k. You can use igl.harmonic_weights to get a similar result.
    :param vs: (n, 3)
    :param fs: (m, 3)
    :param bvids: (k, ), boundary vertex ids
    :param cot: bool, whether to use cotangent Laplacian matrix
    :return:
        out_vs: (n, 3)
    """
    # if cot:
    #     L = igl.harmonic_weights_integrated(vs, fs, k=k).asformat('csr')
    # else:
    #     L = uniform_laplacian_matrix(fs, len(vs), normalize=False, k=k)

    if boundary_preserve:
        _bvids = get_all_boundary_vids(fs)
        bvids = np.unique(np.concatenate([bvids, _bvids]))

    # minimize trace(X^T*L*X), subject to X[bvids] = vs[bvids]
    # Aeq = scipy.sparse.csr_matrix((len(vs), vs.shape[0]))
    # Beq = np.zeros_like(vs)
    # _, out_vs = igl.min_quad_with_fixed(L, np.zeros_like(vs), bvids, vs[bvids], Aeq, Beq, True)

    # the following is faster than igl.min_quad_with_fixed
    if cot:
        L, M = robust_laplacian(vs, fs, delaunay=False)
        L = -L
        b = bvids.astype('i4')
        bc = vs[b]
        return igl.harmonic_weights_from_laplacian_and_mass(L, M, b, bc, k)
        # return igl.harmonic_weights(vs, fs, bvids, vs[bvids], k=k)
    else:
        return igl.harmonic_weights_uniform_laplacian(fs, bvids, vs[bvids], k=k)


def mesh_fair_harmonic_local(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, cot: bool = True, k: int = 1,
                             boundary_preserve: bool = True):
    """
    Do mesh fairing locally via minimizing the Dirichlet energy for fast computation. This will get the submesh
    given by `vids` and then do the fairing on the submesh.
    :param vids: Selected vertex ids to do fairing.
    :param k: The order of the laplacian matrix.
    :return:
    """
    out_vs = vs.copy()
    if k == 1:
        vid_selected = np.zeros(len(vs), dtype=bool)
        vid_selected[vids] = 1
        sub_fids = np.where(np.any(vid_selected[fs], axis=1))[0]
    else:
        nei_vids = get_vertex_neighborhood(fs, vids, order=k)
        nei_vids = np.concatenate([nei_vids, vids])
        sub_fids = get_fids_from_vids(fs, nei_vids)
    sub_vs, sub_fs, IM, sub_vids = igl.remove_unreferenced(out_vs, fs[sub_fids])

    bvids = IM[np.setdiff1d(sub_vids, vids)]
    if boundary_preserve:
        _bvids = get_all_boundary_vids(fs)
        bvids = np.unique(np.concatenate([bvids, IM[_bvids]]))
        bvids = np.setdiff1d(bvids, -1)
    out_vs[sub_vids] = mesh_fair_harmonic_global(sub_vs, sub_fs, bvids, cot=cot, k=k, boundary_preserve=False)
    return out_vs


def mesh_fair_laplacian_energy(vs: np.ndarray, fs: np.ndarray, vids: np.ndarray, alpha=1e-4, k=2):
    import robust_laplacian
    L, M = robust_laplacian.mesh_laplacian(vs, fs)
    Q = igl.harmonic_weights_integrated_from_laplacian_and_mass(L, M, k)

    a = np.full(len(vs), 0.)  # alpha
    a[vids] = alpha
    a = scipy.sparse.diags(a)
    out_vs = scipy.sparse.linalg.spsolve(a * Q + M - a * M, (M - a * M) @ vs)
    return out_vs


def triangulation_refine_leipa(vs: np.ndarray, fs: np.ndarray, fids: np.ndarray, density_factor: float = np.sqrt(2)):
    """
    Refine the triangles using barycentric subdivision and Delaunay triangulation.
    You should remove unreferenced vertices before the refinement.
    See "Filling holes in meshes." [Liepa 2003].

    :return:
        out_vs: (n, 3), the added vertices are appended to the end of the original vertices.
        out_fs: (m, 3), the added faces are appended to the end of the original faces.
        FI: (len(fs), ), face index mapping from the original faces to the refined faces, where
            fs[i] = out_fs[FI[i]], and FI[i]=-1 means the i-th face is deleted.
    """
    out_vs = np.copy(vs)
    out_fs = np.copy(fs)

    if fids is None or len(fids) == 0:
        return out_vs, out_fs, np.arange(len(fs))

    # initialize sigma
    l = get_mollified_edge_length(out_vs, out_fs)
    vv_adj_list = get_vv_adj_list(out_fs, len(vs))
    v_sigma = np.array([l[adj_vids].mean() for adj_vids in vv_adj_list])  # nv
    vc_sigma = v_sigma[out_fs].mean(axis=1)  # nf

    all_sel_fids = np.copy(fids)
    for _ in range(100):
        # check edge length
        s = density_factor * np.linalg.norm(
            out_vs[out_fs[all_sel_fids]].mean(1, keepdims=True) - out_vs[out_fs[all_sel_fids]], axis=-1)
        cond = np.all(np.logical_and(s > vc_sigma[all_sel_fids, None], s > v_sigma[out_fs[all_sel_fids]]), axis=1)
        sel_fids = all_sel_fids[cond]  # need to subdivide

        if len(sel_fids) == 0:
            break

        # subdivide
        out_vs, added_fs = igl.false_barycentric_subdivision(out_vs, out_fs[sel_fids])

        # delete old faces from out_fs and all_sel_fids
        out_fs[sel_fids] = -1
        all_sel_fids = np.setdiff1d(all_sel_fids, sel_fids)

        # add new vertices, faces & update selection
        out_fs = np.concatenate([out_fs, added_fs], axis=0)
        sel_fids = np.arange(len(out_fs) - len(added_fs), len(out_fs))
        all_sel_fids = np.concatenate([all_sel_fids, sel_fids], axis=0)

        # delaunay
        l = get_mollified_edge_length(out_vs, out_fs[all_sel_fids])
        _, add_fs = igl.intrinsic_delaunay_triangulation(l, out_fs[all_sel_fids])
        out_fs[all_sel_fids] = add_fs

        # update vv_adj_list
        vv_adj_list.extend([[] for _ in range(len(out_vs) - len(vv_adj_list))])
        _vv_adj_list = get_vv_adj_list(out_fs[all_sel_fids], len(out_vs))
        sel_vids = np.unique(out_fs[all_sel_fids])
        for sel_vid in sel_vids:
            vv_adj_list[sel_vid] = _vv_adj_list[sel_vid]

        # update v_sigma and vc_sigma
        v_sigma = np.concatenate([v_sigma, np.zeros(len(out_vs) - len(v_sigma))])
        l = get_mollified_edge_length(out_vs, out_fs)
        for sel_vid in sel_vids:
            if sel_vid >= len(vs):
                v_sigma[sel_vid] = l[_vv_adj_list[sel_vid]].mean()
        vc_sigma = v_sigma[out_fs].mean(axis=1)  # nf

    # update FI, remove deleted faces
    FI = np.arange(len(fs))
    FI[out_fs[:len(fs), 0] < -1] = -1
    idx = np.where(FI >= 0)[0]
    FI[idx] = np.arange(len(idx))
    out_fs = out_fs[out_fs[:, 0] >= 0]
    return out_vs, out_fs, FI
