import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

coord_paths=glob('data_loose/*')
random.shuffle(coord_paths)
import time


split_index = int(0.8 * len(coord_paths))


train_paths = coord_paths[:split_index]
test_paths = coord_paths[split_index:]

# print(coord_paths)
# exit()

def plot_and_save(data, filename, y_lines,x_range):
    plt.figure(figsize=(10, 6))
    
    for x, y in data:
        plt.scatter(x, y, marker='o', label=f'Dataset {len(data)}')
    
    # Draw horizontal lines
    for y_line in y_lines:
        plt.axhline(y=y_line, color='r', linestyle='--', label=f'Line at y={y_line}')
    
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0,x_range)  # Set x-axis range
    # plt.ylim(y_range)

    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def process_and_save(paths, subdir):
    for coord in tqdm(paths):
        # Create the output directory based on whether it's train or test
        out_dir = coord.replace('data_loose', f'data_classification_loose/{subdir}').replace('.npz', '')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Ensure the directory exists
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except FileExistsError:
                pass

        # Load data from the .npz file
        loaded_data = np.load(coord, allow_pickle=True)



        testCloud_loaded = loaded_data['raw_input']
        # selected_upMatP1_loaded = loaded_data['sensor_upMatP1']
        # selected_upMatP2_loaded = loaded_data['sensor_upMatP2']
        # selected_upMatP3_loaded = loaded_data['sensor_upMatP3']
        # selected_downMatP1_loaded = loaded_data['sensor_downMatP1']
        # selected_downMatP2_loaded = loaded_data['sensor_downMatP2']
        # selected_downMatP3_loaded = loaded_data['sensor_downMatP3']

        zCoords = [968, 868, 768, 0, -100, -200]

        front_indices = [0, 2, 4, 6, 8, 10]
        side_indices = [1, 3, 5, 7, 9, 11]


        front_data = [(testCloud_loaded[i], np.full_like(testCloud_loaded[i], zCoords[j])) for i, j in zip(front_indices, range(6))]
        side_data = [(testCloud_loaded[i], np.full_like(testCloud_loaded[i], zCoords[j])) for i, j in zip(side_indices, range(6))]

        # Save plots

        plot_and_save(front_data, f'{out_dir}/input_front.png', zCoords, 1000)
        plot_and_save(side_data, f'{out_dir}/input_side.png', zCoords, 2000)


        # Save the arrays in a compressed .npz file
        # arrays_to_save = {
        #     'upMatP1': selected_upMatP1_loaded,
        #     'upMatP2': selected_upMatP2_loaded,
        #     'upMatP3': selected_upMatP3_loaded,
        #     'downMatP1': selected_downMatP1_loaded,
        #     'downMatP2': selected_downMatP2_loaded,
        #     'downMatP3': selected_downMatP3_loaded
        # }
        # np.savez_compressed(f'{out_dir}/labels.npz', **arrays_to_save)


process_and_save(train_paths, 'train')

# Process and save testing data
process_and_save(test_paths, 'test')


# for coord in tqdm(coord_paths):

#     out_dir=coord.replace('train_data','train_data_png_and_coords').replace('.npz','')

#     if not os.path.exists(out_dir):
#         try:
#             os.makedirs(out_dir)
#         except FileExistsError:
#             pass

#     loaded_data=np.load(coord,allow_pickle=True)

#     testCloud_loaded = loaded_data['raw_input']
#     selected_upMatP1_loaded =   loaded_data['muon_hit_upMatP1']
#     selected_upMatP2_loaded =   loaded_data['muon_hit_upMatP2']
#     selected_upMatP3_loaded =   loaded_data['muon_hit_upMatP3']
#     selected_downMatP1_loaded = loaded_data['muon_hit_downMatP1']
#     selected_downMatP2_loaded = loaded_data['muon_hit_downMatP2']
#     selected_downMatP3_loaded = loaded_data['muon_hit_downMatP3']


#     zCoords = [968, 868, 768, 0, -100, -200]
    

#     front_indices = [0, 2, 4,6,8,10]
#     side_indices = [1, 3, 5,7,9,11]
 
#     front_data = [(testCloud_loaded[i], np.full_like(testCloud_loaded[i], zCoords[j])) for i, j in zip(front_indices, range(6))]
#     side_data = [(testCloud_loaded[i], np.full_like(testCloud_loaded[i], zCoords[j])) for i, j in zip(side_indices, range(6))]


#     plot_and_save(front_data, f'{out_dir}/input_front.png', zCoords,1000)
#     plot_and_save(side_data,  f'{out_dir}/input_side.png', zCoords,2000)

#     arrays_to_save = {
#     'upMatP1': selected_upMatP1_loaded,
#     'upMatP2': selected_upMatP2_loaded,
#     'upMatP3': selected_upMatP3_loaded,
#     'downMatP1': selected_downMatP1_loaded,
#     'downMatP2': selected_downMatP2_loaded,
#     'downMatP3': selected_downMatP3_loaded}
#     np.savez_compressed(f'{out_dir}/labels.npz', **arrays_to_save)

#     labels = [
#     np.column_stack((mat, np.full_like(mat[:, 0], zCoord)))
#     for mat, zCoord in zip(
#         [
#             selected_upMatP1_loaded, 
#             selected_upMatP2_loaded, 
#             selected_upMatP3_loaded, 
#             selected_downMatP1_loaded, 
#             selected_downMatP2_loaded, 
#             selected_downMatP3_loaded
#         ], 
#         zCoords
#     )
# ]
#     combined_labels = np.vstack(labels)

#     np.save(f'{out_dir}/label.npy',combined_labels)








    # front_labels=[
    # (selected_upMatP1_loaded[:,0],np.full_like((selected_upMatP1_loaded[:,0]), zCoords[0])),
    # (selected_upMatP2_loaded[:,0],np.full_like((selected_upMatP2_loaded[:,0]), zCoords[1])),
    # (selected_upMatP3_loaded[:,0],np.full_like((selected_upMatP3_loaded[:,0]), zCoords[2])),

    # (selected_downMatP1_loaded[:,0],np.full_like((selected_downMatP1_loaded[:,0]), zCoords[3])),
    # (selected_downMatP2_loaded[:,0],np.full_like((selected_downMatP2_loaded[:,0]), zCoords[4])),
    # (selected_downMatP3_loaded[:,0],np.full_like((selected_downMatP3_loaded[:,0]), zCoords[5]))
    # ]

    # side_labels=[
    # (selected_upMatP1_loaded[:,1],np.full_like((selected_upMatP1_loaded[:,1]), zCoords[0])),
    # (selected_upMatP2_loaded[:,1],np.full_like((selected_upMatP2_loaded[:,1]), zCoords[1])),
    # (selected_upMatP3_loaded[:,1],np.full_like((selected_upMatP3_loaded[:,1]), zCoords[2])),

    # (selected_downMatP1_loaded[:,1],np.full_like((selected_downMatP1_loaded[:,1]), zCoords[3])),
    # (selected_downMatP2_loaded[:,1],np.full_like((selected_downMatP2_loaded[:,1]), zCoords[4])),
    # (selected_downMatP3_loaded[:,1],np.full_like((selected_downMatP3_loaded[:,1]), zCoords[5]))
    # ]

    # plot_and_save(front_labels, f'{out_dir}/label_front.png', zCoords,1000)
    # plot_and_save(side_labels, f'{out_dir}/label_side.png', zCoords,2000)




    


