import copy
import itertools

import numpy as np
from funcy import print_durations
from typing import Union
from scipy.io import loadmat
from visualiser import VisualizeMat
from scipy.spatial import cKDTree
import time
import h5py
import os
import concurrent.futures
import sys

class Coord:
    X = 0
    Y = 1


class Hodoscope:
    UP = 1
    DOWN = 2

    # TODO why gscan wants to eliminate left and right hodoscopes
    # TODO bc muons are from a normal distribution
    # LEFT = 3
    # RIGHT = 4

zCoords = [968, 868, 768, 0, -100, -200]

@print_durations()
def loadData():
    filename = "../tmp_short/CLOUDS.mat"
    data0 = loadmat(filename)

    data: np.ndarray = data0["CLOUDS"]

    eventNr_np = np.unique(data[:, 0]).astype(int)

    d = np.zeros(len(eventNr_np), dtype=object)
    empty_index = 0

    for idx, event_nr in enumerate(eventNr_np):
        event = data[data[:, 0] == event_nr]

        if len(event) != 12:
            empty_index += 1
            d = np.delete(d, idx, 0)

            continue

        # Create a new array to store filtered rows
        filtered_event = np.empty(len(event), dtype=object)

        for i, row in enumerate(event):
            filtered_event[i] = row[row != -1][2:]

        d[idx - empty_index] = filtered_event

    return d


def printCloud(cloud, alignConst=10):
    matNRs = [6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6]
    for i in range(0, len(cloud), 2):

        matNRx, matNRy = 6 - i if i <= 5 else i - 5, 5 - i if i <= 5 else i - 4  # matNRs[i], matNRs[i+1]  # (i % 6) // 2 + 1
        x1 = cloud[i]
        y1 = cloud[i + 1]
        x1 = [n for n in x1 if n != -1]
        y1 = [n for n in y1 if n != -1]

        print(f"X{matNRx} ({len(x1)})", end="")
        for x in x1:
            if str(x).endswith(".0"): x = int(x)
            print(str(x).rjust(alignConst), end="")

        print()

        print(f"Y{matNRy} ({len(y1)})", end="")
        for y in y1:
            if str(y).endswith(".0"): y = int(y)
            print(str(y).rjust(alignConst), end="")

        if i == 4:
            print("\n")
        else:
            print()


@print_durations()
def createGlobalMatCoords(cloud):
    assert len(cloud) == 12


    mats = []
    for i in range(6):
            X = cloud[2 * i]
            Y = cloud[2 * i + 1]
            Z = np.array(zCoords[i])
            
            mats.append((X, Y, Z))

    return mats


def generatePoints(mats,mat_no):
    mat_x = mats[mat_no*2]
    mat_y = mats[mat_no*2+1]


    x, y = np.meshgrid(mat_x, mat_y)
    
    # Reshape the meshgrid arrays to create pairs
    pairs = np.vstack([x.ravel(), y.ravel()]).T



    return np.array(pairs)

def generatePairs(mat1,mat2):
    result=np.array(list(itertools.product(mat1, mat2)))

    return result


def linear_exterpolation(point1, point2, z_coord1,z_coord2,z_coord_ext):

    pairs=generatePairs(point1,point2)



    up_points=pairs[:,0]
    down_points=pairs[:,1]



    z1 = z_coord1
    z2 = z_coord2
    
    # k = (x - x1) / (x2 - x1)
    k = (z_coord_ext - z1) / (z2 - z1)

    # y = y1 + k * (y2 - y1)
    point3 = up_points + k * (down_points - up_points)


    return point3


def search(large_array, small_array, threshold):

    result = []
    for small_coord in small_array:
        # Calculate the distance from the current small coordinate to all large coordinates
        distances = np.linalg.norm(large_array - small_coord, axis=1)
        
        # Check if there are any distances within the threshold
        if np.any(distances <= threshold):
            result.append(small_coord)
    
    return np.array(result)

def calculate_threshold(P1, P2, threshold,angle_coeff):
    """
    Calculate the dynamic threshold based on the average coordinates of two points.

    Args:
    P1 (np.array): Coordinates (x, y) of the first point.
    P2 (np.array): Coordinates (x, y) of the second point.
    threshold (float): Base threshold value.

    Returns:
    float: Adjusted threshold based on the angle.
    """
    # Average coordinates
    avg_P1 = np.mean(P1, axis=0)
    avg_P2 = np.mean(P2, axis=0)

    # Calculate horizontal and vertical distances
    dx = avg_P2[0] - avg_P1[0]
    dy = avg_P1[1] - avg_P2[1]

    # Calculate the combined horizontal distance
    d = np.sqrt(dx**2 + dy**2)

    # Calculate the angle using arctan and convert to degrees
    theta = np.arctan(d / 100)

    # Calculate dynamic threshold
    adjusted_threshold = threshold + threshold * np.cos(theta)*angle_coeff
    # print("Angle: ",np.degrees(theta))
    # print("Threshold: ",threshold)
    # print("Adjusted: ",adjusted_threshold)
    
    return adjusted_threshold


def hit_getter(anchor,fallback,threshold,filter_type):
    return 0


def process_cloud(index, testCloud, base_threshold, angle_coeff, zCoords,folder_path,allowed_miss, miss_counter):
    try:
        
        start_time = time.time()
    
        upMatP1 = generatePoints(testCloud, 0)
        upMatP2 = generatePoints(testCloud, 1)
        upMatP3 = generatePoints(testCloud, 2)
    
        downMatP1 = generatePoints(testCloud, 3)
        downMatP2 = generatePoints(testCloud, 5)
        downMatP3 = generatePoints(testCloud, 4)
     
        threshold = calculate_threshold(upMatP1, upMatP2, base_threshold,angle_coeff)

    
         
        #First up down
        ext_upMatP3 = linear_exterpolation(upMatP1, upMatP2, zCoords[0], zCoords[1], zCoords[2])
        selected_upMatP3 = search(ext_upMatP3, upMatP3, threshold)
        if len(selected_upMatP3) == 0:
            next_mat=upMatP2
            next_z=zCoords[1]
            miss_counter+=1
        else:
            next_mat=selected_upMatP3
            next_z=zCoords[2]
    
        ext_downMatP1 = linear_exterpolation(upMatP1, next_mat, zCoords[0], next_z, zCoords[3])
        selected_downMatP1 = search(ext_downMatP1, downMatP1, threshold)
        if len(selected_downMatP1) == 0:
            next_mat=upMatP2
            next_z=zCoords[1]
            miss_counter+=1
        else:
            next_mat=selected_downMatP1
            next_z=zCoords[3]
    
    
        ext_downMatP2 = linear_exterpolation(upMatP1, next_mat, zCoords[0], next_z, zCoords[4])
        selected_downMatP2 = search(ext_downMatP2, downMatP2, threshold)
        if len(selected_downMatP2) == 0:
            next_mat=upMatP2
            next_z=zCoords[1]
            miss_counter+=1
        else:
            next_mat=selected_downMatP2
            next_z=zCoords[4]

    
        ext_downMatP3 = linear_exterpolation(upMatP1, next_mat, zCoords[0], next_z, zCoords[5])
        selected_downMatP3 = search(ext_downMatP3, downMatP3, threshold)
        if len(selected_downMatP3) == 0:
            miss_counter+=1
        else:
            next_mat=selected_downMatP3
    
            

        hit_counter=6-miss_counter
        data_dict = {
            "raw_input": testCloud,
            'hit_upMatP1': upMatP1,
            'hit_upMatP2': upMatP2,
            'hit_upMatP3': selected_upMatP3,
            'hit_downMatP1': selected_downMatP1,
            'hit_downMatP2': selected_downMatP2,
            'hit_downMatP3': selected_downMatP3
        }

        if hit_counter==2: #Updown with 1 3
            miss_counter=0

            ext_downMatP1 = linear_exterpolation(upMatP1, upMatP3, zCoords[0], zCoords[2], zCoords[3])
            selected_downMatP1 = search(ext_downMatP1, downMatP1, threshold)
            if len(selected_downMatP1) == 0:
                next_mat=upMatP3
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_downMatP1
                next_z=zCoords[3]
        
        
            ext_downMatP2 = linear_exterpolation(upMatP1, next_mat, zCoords[0], next_z, zCoords[4])
            selected_downMatP2 = search(ext_downMatP2, downMatP2, threshold)
            if len(selected_downMatP2) == 0:
                next_mat=upMatP3
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_downMatP2
                next_z=zCoords[4]
    
        
            ext_downMatP3 = linear_exterpolation(upMatP1, next_mat, zCoords[0], next_z, zCoords[5])
            selected_downMatP3 = search(ext_downMatP3, downMatP3, threshold)
            if len(selected_downMatP3) == 0:
                miss_counter+=1
            else:
                next_mat=selected_downMatP3

            data_dict = { #1 3
            "raw_input": testCloud,
            'hit_upMatP1': upMatP1,
            'hit_upMatP2': np.array([]),
            'hit_upMatP3': upMatP3,
            'hit_downMatP1': selected_downMatP1,
            'hit_downMatP2': selected_downMatP2,
            'hit_downMatP3': selected_downMatP3
        }
            hit_counter=5-miss_counter


        if hit_counter==2: #2 3
            miss_counter=0

            ext_downMatP1 = linear_exterpolation(upMatP2, upMatP3, zCoords[1], zCoords[2], zCoords[3])
            selected_downMatP1 = search(ext_downMatP1, downMatP1, threshold)
            if len(selected_downMatP1) == 0:
                next_mat=upMatP3
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_downMatP1
                next_z=zCoords[3]
        
        
            ext_downMatP2 = linear_exterpolation(upMatP2, next_mat, zCoords[1], next_z, zCoords[4])
            selected_downMatP2 = search(ext_downMatP2, downMatP2, threshold)
            if len(selected_downMatP2) == 0:
                next_mat=upMatP3
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_downMatP2
                next_z=zCoords[4]
    
        
            ext_downMatP3 = linear_exterpolation(upMatP2, next_mat, zCoords[1], next_z, zCoords[5])
            selected_downMatP3 = search(ext_downMatP3, downMatP3, threshold)
            if len(selected_downMatP3) == 0:
                miss_counter+=1
            else:
                next_mat=selected_downMatP3

            data_dict = { #2 3
            "raw_input": testCloud,
            'hit_upMatP1': np.array([]),
            'hit_upMatP2': upMatP2,
            'hit_upMatP3': upMatP3,
            'hit_downMatP1': selected_downMatP1,
            'hit_downMatP2': selected_downMatP2,
            'hit_downMatP3': selected_downMatP3
        }
            hit_counter=5-miss_counter        
        
        if hit_counter==2: #6 5
            miss_counter=0

            ext_downMatP1 = linear_exterpolation(downMatP3, downMatP2, zCoords[5], zCoords[4], zCoords[3])
            selected_downMatP1 = search(ext_downMatP1, downMatP1, threshold)
            if len(selected_downMatP1) == 0:
                next_mat=downMatP2
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_downMatP1
                next_z=zCoords[3]
        
        
            ext_upMatP3 = linear_exterpolation(downMatP3, next_mat, zCoords[5], next_z, zCoords[2])
            selected_upMatP3 = search(ext_upMatP3, upMatP3, threshold)
            if len(selected_upMatP3) == 0:
                next_mat=downMatP2
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_upMatP3
                next_z=zCoords[2]
    
        
            ext_upMatP2 = linear_exterpolation(downMatP3, next_mat, zCoords[5], next_z, zCoords[1])
            selected_upMatP2 = search(ext_upMatP2, upMatP3, threshold)
            if len(selected_upMatP2) == 0:
                next_mat=downMatP2
                next_z=zCoords[2]
                miss_counter+=1
            else:
                next_mat=selected_upMatP2
                next_z=zCoords[1]

            ext_upMatP1 = linear_exterpolation(downMatP3, next_mat, zCoords[5], next_z, zCoords[0])
            selected_upMatP1 = search(ext_upMatP1, upMatP1, threshold)
            if len(selected_upMatP1) == 0:
                miss_counter+=1
            else:
                next_mat=selected_upMatP1
                next_z=zCoords[1]


            data_dict = { #6 5
            "raw_input": testCloud,
            'hit_upMatP1': selected_upMatP1,
            'hit_upMatP2': selected_upMatP2,
            'hit_upMatP3': selected_upMatP3,
            'hit_downMatP1': selected_downMatP1,
            'hit_downMatP2': downMatP2,
            'hit_downMatP3': downMatP3
        }
            hit_counter=6-miss_counter        

        if hit_counter==2: #6 4
            miss_counter=0

            ext_upMatP3 = linear_exterpolation(downMatP3, downMatP1, zCoords[5], zCoords[3], zCoords[2])
            selected_upMatP3 = search(ext_upMatP3, upMatP3, threshold)
            if len(selected_upMatP3) == 0:
                next_mat=downMatP1
                next_z=zCoords[3]
                miss_counter+=1
            else:
                next_mat=selected_upMatP3
                next_z=zCoords[2]
    
        
            ext_upMatP2 = linear_exterpolation(downMatP3, next_mat, zCoords[5], next_z, zCoords[1])
            selected_upMatP2 = search(ext_upMatP2, upMatP3, threshold)
            if len(selected_upMatP2) == 0:
                next_mat=downMatP1
                next_z=zCoords[3]
                miss_counter+=1
            else:
                next_mat=selected_upMatP2
                next_z=zCoords[1]

            ext_upMatP1 = linear_exterpolation(downMatP3, next_mat, zCoords[5], next_z, zCoords[0])
            selected_upMatP1 = search(ext_upMatP1, upMatP1, threshold)
            if len(selected_upMatP1) == 0:
                miss_counter+=1
            else:
                next_mat=selected_upMatP1
                next_z=zCoords[1]


            data_dict = { #6 5
            "raw_input": testCloud,
            'hit_upMatP1': selected_upMatP1,
            'hit_upMatP2': selected_upMatP2,
            'hit_upMatP3': selected_upMatP3,
            'hit_downMatP1': downMatP1,
            'hit_downMatP2': np.array([]),
            'hit_downMatP3': downMatP3
        }
            hit_counter=5-miss_counter        

        if hit_counter==2: #6 4
            miss_counter=0

            ext_upMatP3 = linear_exterpolation(downMatP2, downMatP1, zCoords[4], zCoords[3], zCoords[2])
            selected_upMatP3 = search(ext_upMatP3, upMatP3, threshold)
            if len(selected_upMatP3) == 0:
                next_mat=downMatP1
                next_z=zCoords[3]
                miss_counter+=1
            else:
                next_mat=selected_upMatP3
                next_z=zCoords[2]
    
        
            ext_upMatP2 = linear_exterpolation(downMatP2, next_mat, zCoords[4], next_z, zCoords[1])
            selected_upMatP2 = search(ext_upMatP2, upMatP3, threshold)
            if len(selected_upMatP2) == 0:
                next_mat=downMatP1
                next_z=zCoords[3]
                miss_counter+=1
            else:
                next_mat=selected_upMatP2
                next_z=zCoords[1]

            ext_upMatP1 = linear_exterpolation(downMatP1, next_mat, zCoords[4], next_z, zCoords[0])
            selected_upMatP1 = search(ext_upMatP1, upMatP1, threshold)
            if len(selected_upMatP1) == 0:
                miss_counter+=1
            else:
                next_mat=selected_upMatP1
                next_z=zCoords[1]


            data_dict = { #6 5
            "raw_input": testCloud,
            'hit_upMatP1': selected_upMatP1,
            'hit_upMatP2': selected_upMatP2,
            'hit_upMatP3': selected_upMatP3,
            'hit_downMatP1': downMatP1,
            'hit_downMatP2': np.array([]),
            'hit_downMatP3': downMatP3
        }
            hit_counter=5-miss_counter        

 



        end_time = time.time()
        loop_time = end_time - start_time
        print(f"Cloud number {index} processed in: {loop_time} seconds with threshold {threshold} found {hit_counter} hit")
        np.savez_compressed(f'{folder_path}/{index}_{hit_counter}.npz', **data_dict)


    except Exception as e:
        print(f"Cloud number {index} with threshold {threshold} Error encountered: {e}")
        print(len(selected_downMatP1))

    



def main():
    clouds: np.ndarray = loadData()
    print(clouds.shape)


    base_threshold=30
    angle_coeff=6 
    folder_path = 'data_square_hit'

    allowed_miss = 1  # Set initial allowed_miss
    miss_counter = 0 
    

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_cloud, index, cloud, base_threshold, angle_coeff, zCoords,folder_path,allowed_miss, miss_counter) for index, cloud in enumerate(clouds)]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for the tas



if __name__ == "__main__":
    main()
