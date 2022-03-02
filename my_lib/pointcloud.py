import open3d as o3d
import numpy as np

class PointCloudManipulator:
    COLORS = [[],  # reserved
              [1.0, 0.7, 0],  # yellow
              [0.9, 0.7, 0.6],  # pink
              [0.6, 0.9, 0.5],  # green
              [0.5, 0.7, 0.9],  # blue
              [0.9, 0.6, 0.9],  # purple
              [0, 0.8, 0.9],
              [0.6, 0.4, 0.4],
              [0.6, 0.7, 0.1],
              # [0.7, 0.7, 0.2],
              [0.9, 0.9, 0.3],
              [0.3, 0.6, 0.5],
              [0.8, 0.5, 1.0],
              [0.6, 1.0, 1.0],
              [1.0, 0.6, 0.6],
              [0.0, 0.8, 0.4],
              [1.0, 0.4, 0.7],
              [1.0, 0.8, 0.6]
              ]

    @staticmethod
    def visualize(points:np.ndarray, labels:np.ndarray):
        colors = np.array(PointCloudManipulator.COLORS[1:])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors[labels%len(colors)])
        o3d.visualization.draw_geometries([pcd])
