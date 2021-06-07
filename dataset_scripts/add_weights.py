import collections
import os
import cv2
import numpy as np

import print_unique_colors


def create_weights(
    input_dir
):
    masks_paths = print_unique_colors.get_file_paths_from_folder(input_dir)
    
    weights = []
    
    label_freq = {}

    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        dim1, dim2 = mask.shape
        
        label_counter = collections.Counter(mask.flatten())

        for label, count in label_counter.items():
            label_freq[label] = label_freq.get(label, 0) +count / (dim1 * dim2)
    
    median_freq = np.median(list(label_freq.values()))
    print(median_freq, np.mean(list(label_freq.values())))
    
    label_weights = { 
        label: median_freq / freq
        for label, freq in label_freq.items()
    }

    print(label_weights)
    
    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        dim1, dim2 = mask.shape

        pixel_weights = np.ones(
            (dim1, dim2, 1),
            dtype=np.float32
        )
        
        for i in range(dim1):
            for j in range(dim2):
                pixel_weights[i][j][0] = label_weights[mask[i][j]]

        weights.append(pixel_weights)

    return weights


def obtain_weights(input_dir, output_path):
    weights = create_weights(input_dir)
    
    if not output_path.endswith('.npz'):
        output_path += '.npz'

    np.savez(
        output_path,
        np.array(weights)
    )


if __name__ == '__main__':
    baseline_dataset_dir = '../baseline_dataset'

    weights = create_weights(
        os.path.join(
            baseline_dataset_dir,
            'labeled_masks'
        )
    )
    
    weights_path = os.path.join(
        baseline_dataset_dir,
        'masks_weights.npz'
    )

    np.savez(
        weights_path,
        np.array(weights)
    )
