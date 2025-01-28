# https://github.com/rgl-epfl/large-steps-pytorch/blob/master/scripts/geometry.py

import torch


def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse


def compute_face_normals(verts, faces):
    """
    Compute per-face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2]),
    ]

    c = torch.cross(v[1] - v[0], v[2] - v[0])
    n = c / torch.norm(c, dim=0)
    return n


def safe_acos(x):
    return torch.acos(x.clamp(min=-1, max=1))


def compute_vertex_normals(verts, faces, face_normals):
    """
    Compute per-vertex normals from face normals.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    face_normals : torch.Tensor
        Per-face normals
    """
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [
        verts.index_select(1, fi[0]),
        verts.index_select(1, fi[1]),
        verts.index_select(1, fi[2]),
    ]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0 * d1, 0)
        face_angle = safe_acos(torch.sum(d0 * d1, 0))
        nn = face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)


def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3
