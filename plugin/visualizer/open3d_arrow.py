import math
import numpy as np
import open3d as o3d


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    magnitude = np.sqrt(np.sum(vec ** 2)) + 0.0001
    return (magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def create_arrow(scale=10, fixed_radius=None):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2 + 0.001
    cylinder_height = scale * 0.5 + 0.001
    if fixed_radius == None:
        cone_radius = scale / 10 + 0.001
        cylinder_radius = scale / 30 + 0.001
    else:
        cone_radius = 3 * fixed_radius + 0.001
        cylinder_radius = fixed_radius + 0.001
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
                                                        cone_height=cone_height,
                                                        cylinder_radius=cylinder_radius,
                                                        cylinder_height=cylinder_height)
    return (mesh_frame)


def get_arrow(origin=[0, 0, 0], end=None, vec=None, fixed_radius=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 5
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale, fixed_radius=fixed_radius)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return (mesh)


def get_yaw_arrows(pre_box_set):
    draw_arrows = []
    for box_p in pre_box_set:
        cx, cy, cz, yaw = box_p[0], box_p[1], box_p[2], box_p[6]
        orin = [cx, cy, cz]
        end = [cx + 5 * math.cos(yaw), cy + 5 * math.sin(yaw), cz]
        draw_arrow = get_arrow(orin, end)
        draw_arrow.paint_uniform_color([0.4, 0.2, 0.9])
        draw_arrows.append(draw_arrow)
    return draw_arrows


def get_speed_arrows(pre_box_set):
    draw_arrows = []
    for box_p in pre_box_set:
        cx, cy, cz, sz, speed_x, speed_y = box_p[0], box_p[1], box_p[2], box_p[5], box_p[7], box_p[8]
        orin = [cx, cy, cz + 0.5 * sz]
        end = [cx + 5 * speed_x, cy + 5 * speed_y, cz + 0.5 * sz]
        draw_arrow = get_arrow(orin, end, fixed_radius=0.1)
        draw_arrow.paint_uniform_color([0.8, 0.2, 0.3])
        draw_arrows.append(draw_arrow)
    return draw_arrows
