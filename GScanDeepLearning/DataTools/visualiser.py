# import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import math
from numpy.linalg import inv
import copy
from numpy.linalg import det
import sys
# from matplotlib.colors import ListedColormap
# from matplotlib.colors import LogNorm
# import matplotlib
import warnings
import time
import open3d as o3d
import scipy.io
import copy

warnings.filterwarnings('ignore')




class VisualizeMat:
    def __init__(self,lines=True):
        self.line_exist=lines


        self.z_axis_up1=1059
        self.z_axis_up2=959
        self.z_axis_up3=859

        self.z_axis_down1=0
        self.z_axis_down2=-100
        self.z_axis_down3=-200

        self.mat_values = {
            "up1": self.z_axis_up1,
            "up2": self.z_axis_up2,
            "up3": self.z_axis_up3,
            "down1": self.z_axis_down1,
            "down2": self.z_axis_down2,
            "down3": self.z_axis_down3
        }



        self.up1 = np.empty((0, 3))
        self.up2 = np.empty((0, 3))
        self.up3 = np.empty((0, 3))
        self.down1 = np.empty((0, 3))
        self.down2 = np.empty((0, 3))
        self.down3 = np.empty((0, 3))
        self.left1 = np.empty((0, 3))
        self.left2 = np.empty((0, 3))
        self.left3 = np.empty((0, 3))
        self.right1 = np.empty((0, 3))
        self.right2 = np.empty((0, 3))
        self.right3 = np.empty((0, 3))

        self.vis = o3d.visualization.Visualizer()

    # Additional methods for adding points to the arrays, visualizing, etc.

    def add_point(self, point, mat):
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        if point.shape[1] == 2:
            # Add a new column of zeros to change the shape to (n, 3)
            new_column = np.zeros((point.shape[0], 1))
            point = np.hstack((point, new_column))
            
        if mat == "up1":
            point[:,2]=self.z_axis_up1
            self.up1 = np.vstack((self.up1, point))
        elif mat == "up2":
            point[:,2]=self.z_axis_up2
            self.up2 = np.vstack((self.up2, point))
        elif mat == "up3":
            point[:,2]=self.z_axis_up3
            self.up3 = np.vstack((self.up3, point))
        elif mat == "down1":
            point[:,2]=0
            self.down1 = np.vstack((self.down1, point))
        elif mat == "down2":
            point[:,2]=self.z_axis_down2
            self.down2 = np.vstack((self.down2, point))
        elif mat == "down3":
            point[:,2]=self.z_axis_down3
            self.down3 = np.vstack((self.down3, point))
        elif mat == "left1":
            self.left1 = np.vstack((self.left1, point))
        elif mat == "left2":
            self.left2 = np.vstack((self.left2, point))
        elif mat == "left3":
            self.left3 = np.vstack((self.left3, point))
        elif mat == "right1":
            self.right1 = np.vstack((self.right1, point))
        elif mat == "right2":
            self.right2 = np.vstack((self.right2, point))
        elif mat == "right3":
            self.right3 = np.vstack((self.right3, point))
        else:
            raise ValueError(f"Unknown mat type: {mat}")

    def process_and_add_geometry(self, array, color):
        if array.shape[0] == 0:
            return  # Skip empty arrays
        line_points=[[array]]
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(array)
        points.paint_uniform_color(color)
        self.vis.add_geometry(points)


    def add_point_lines(self,array1,array2):

        if len(array1) == 0 or len(array2) == 0:
            return

        points = np.vstack((array1, array2))
        
        # Create lines connecting corresponding points in array1 and array2
        n = min(len(array1),len(array2))
        lines = [[i, i + n] for i in range(n)]
        
        # Convert to Open3D format
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        
        # Add to visualizer
        self.vis.add_geometry(line_set)

    def add_lines(self,array1,array2,mat1,mat2):
        if array1.shape[1] == 2:
            # Add a new column of zeros to change the shape to (n, 3)
            new_column = np.zeros((array1.shape[0], 1))
            array1 = np.hstack((array1, new_column))
        array1[:,2]=self.mat_values[mat1]

        if array2.shape[1] == 2:
            # Add a new column of zeros to change the shape to (n, 3)
            new_column = np.zeros((array2.shape[0], 1))
            array2 = np.hstack((array2, new_column))
        array2[:,2]=self.mat_values[mat2]



        points = np.vstack((array1, array2))
        n = min(len(array1),len(array2))
        lines = [[i, i + n] for i in range(n)]

        line_color = [[0, 0, 0] for _ in range(len(lines))]

        
        # Convert to Open3D format
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        # line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))
        line_set.colors = o3d.utility.Vector3dVector(line_color)
        
        # Add to visualizer
        self.vis.add_geometry(line_set)
        




    def create_v_mat_lines(self,z):

        points_up = [
            [0, 0, z+10],
            [0, 2000, z+10],
            [1200, 0, z+10],
            [1200, 2000, z+10],

            [0, 0, z - 10],
            [0, 2000, z - 10],
            [1200, 0, z - 10],
            [1200, 2000, z - 10],
        ]

        lines_up = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        line_color = [[0, 0, 0] for _ in range(len(lines_up))]
        line_set_up = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_up),
            lines=o3d.utility.Vector2iVector(lines_up),
        )
        line_set_up.colors = o3d.utility.Vector3dVector(line_color)
    
        self.vis.add_geometry(line_set_up)



    def visualize_points(self):

        self.vis.create_window(window_name='Open3D', width=1280, height=720)
        render_opt = self.vis.get_render_option()
        render_opt.background_color = [1, 1, 1]  # Open3D uses a range of 0 to 1 for colors

        colors = {"outer": [1, 0, 0], "middle": [0, 0, 1], "inside": [0, 1, 0],  }# Red, Blue, Green

        self.create_v_mat_lines(self.z_axis_up1)
        self.create_v_mat_lines(self.z_axis_up2)
        self.create_v_mat_lines(self.z_axis_up3)

        self.create_v_mat_lines(self.z_axis_down1)
        self.create_v_mat_lines(self.z_axis_down2)
        self.create_v_mat_lines(self.z_axis_down3)



        self.process_and_add_geometry(self.up1, colors["outer"])
        self.process_and_add_geometry(self.up2, colors["middle"])
        self.process_and_add_geometry(self.up3, colors["inside"])


        

        self.process_and_add_geometry(self.down3, colors["outer"])
        self.process_and_add_geometry(self.down2, colors["middle"])
        self.process_and_add_geometry(self.down1, colors["inside"])

        if self.line_exist:
            self.add_point_lines(self.up1,self.up2)
            self.add_point_lines(self.up2,self.up3)
            self.add_point_lines(self.up3,self.down1)
            self.add_point_lines(self.down1,self.down2)
            self.add_point_lines(self.down2,self.down3)



        self.process_and_add_geometry(self.left1, colors["outer"])
        self.process_and_add_geometry(self.left2, colors["middle"])
        self.process_and_add_geometry(self.left3, colors["inside"])
        self.process_and_add_geometry(self.right3, colors["outer"])
        self.process_and_add_geometry(self.right2, colors["middle"])
        self.process_and_add_geometry(self.right1, colors["inside"])

        # self.vis.draw_geometries(line_set)



        self.vis.run()
        self.vis.destroy_window()


    def reset_visualiser(self):
        self.up1 = np.empty((0, 3))
        self.up2 = np.empty((0, 3))
        self.up3 = np.empty((0, 3))
        self.down1 = np.empty((0, 3))
        self.down2 = np.empty((0, 3))
        self.down3 = np.empty((0, 3))
        self.left1 = np.empty((0, 3))
        self.left2 = np.empty((0, 3))
        self.left3 = np.empty((0, 3))
        self.right1 = np.empty((0, 3))
        self.right2 = np.empty((0, 3))
        self.right3 = np.empty((0, 3))

        self.vis.clear_geometries()



