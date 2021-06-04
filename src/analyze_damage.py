import cv2
import sys
from collections import namedtuple
import json
import os
import numpy as np

"""
path_to_baseline_dataset_dir
focal_length
road_width
debug
"""

DEBUG = False

CROP_ATTENTION_BY_WIDTH = 0.15
MIN_VANISHING_LINE_TO_WIDTH_RATIO = 0.2
MAX_BACKGROUND_TO_ALL_RATIO = 0.2
MIN_ROAD_SURFACE_RATIO = 0.15


VanishingLine = namedtuple('VanishingLine', ['height', 'start', 'end'])
    
DamageItem = namedtuple('DamageItem', ['area', 'width', 'length', 'type'])
FAILED_DAMAGE_ITEM = DamageItem(0, 0, 0, 'failed')


def load_pydict(dir, name):
    pydict_name = os.path.join(dir, name + '.json')

    pydict = None

    with open(pydict_name, 'r') as pydict_file:
        pydict = json.loads(pydict_file.read())

    valid_pydict = []

    for k, v in pydict:
        if isinstance(k, list):
            k = tuple(k)
        
        valid_pydict.append((k, v))

    return dict(valid_pydict)


def damage_items_empty(damage_items):
    return (
        len(damage_items) == 1 and damage_items[0] is FAILED_DAMAGE_ITEM
        or not damage_items
    )


# Check if there is a lot of background_labels under (starting_height, w) coordinate
def fails_underneath_background(mask, w, starting_height, background_label):
    background_labels = 0

    height, _ = mask.shape

    for h in range(starting_height, height):
        if mask[h][w] == background_label:
            background_labels += 1

    if background_labels / (height - starting_height) > MAX_BACKGROUND_TO_ALL_RATIO:
        return True

    return False


# Check if there is enough of current surface_labels mask[starting_height][w] 
def fails_underneath_surface(mask, w, starting_height):
    surface_label = mask[starting_height][w]
    surface_labels = 0

    height, _ = mask.shape

    for h in range(starting_height, height):
        if mask[h][w] == surface_label:
            surface_labels += 1

    if surface_labels / (height - starting_height) < MIN_ROAD_SURFACE_RATIO:
        return True

    return False

# Returns VanishingLine
#   has height set to -1 if fails
def find_vanishing_line(mask, background_label):
    ERR_RESULT = VanishingLine(-1, -1, -1)
    
    height, width = mask.shape

    for h in range(height):
        start = None
        end = None

        skipping = True

        for w in range(width):
            label = mask[h][w]

            if skipping:
                if label != background_label:
                    # cant see start of line
                    start = w

                    if start < CROP_ATTENTION_BY_WIDTH * width:
                        return ERR_RESULT

                    skipping = False

                continue

            if label == background_label:
                end = w - 1
                
                if end > width - CROP_ATTENTION_BY_WIDTH * width:
                    return ERR_RESULT

                break
        
        # there were only backgroung labels
        if skipping:
            continue

        # cant see end of line
        if end is None:
            return ERR_RESULT

        if not ( 
            (end - start + 1) / width < MIN_VANISHING_LINE_TO_WIDTH_RATIO
            or fails_underneath_background(mask, start, h, background_label)
            or fails_underneath_background(mask, end, h, background_label)
            or fails_underneath_surface(mask, start, h)
            or fails_underneath_surface(mask, end, h)
        ):
            return VanishingLine(h, start, end)

    return ERR_RESULT


# Chacks if 0 <= x < sup is True
def valid_coord_component(x, sup):
    return 0 <= x < sup


# Returns list of coordinates which has label equal to road_label
#   in form of [[h1, h2, h3, ...], [w1, w2, w3, ...]]
def find_edge(mask, vanishing_line, side, road_label):
    offset_w = None
    w = None

    if side == 'left':
        offset_w = -1
        w = vanishing_line.start
    elif side == 'right':
        offset_w = +1
        w = vanishing_line.end
    else:
        raise NotImplementedError

    h = vanishing_line.height
    
    edge = [[], []]
    
    height, width = mask.shape

    while True:
        if mask[h][w] == road_label:
            edge[0].append(h)
            edge[1].append(w)

        h += 1

        if not (
            valid_coord_component(h, height)
            and valid_coord_component(w + offset_w, width)
        ):
            break

        while mask[h][w + offset_w] == road_label:
            w += offset_w

            if not valid_coord_component(w + offset_w, width):
                break

    return edge


# Approximates points with a line kx + b
#   Returns (k, b)
def least_squares_line(x, y):
    A = np.vstack([x, np.ones(len(x))]).T

    k, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return k, b


# Returns:
#   h_nearest -- nearest h coordinate to the camera
#   item_labels -- number of pixels the item with labels mask[h][w] consists of
#   pixel_width -- maximum width of item in pixels along width axis
def examine_item(
    mask,
    left_edge,
    right_edge,
    h,
    w,
    used_points,
    vanishing_height
):
    used_points[h][w] = 1

    queue = [(h, w)]
    item_label = mask[h][w]
    
    min_w = w
    max_w = w
    h_nearest = h
    
    height, width = mask.shape
    
    item_labels = 0

    def valid_w(w, h):
        return max(0, round(left_edge[0] * h + left_edge[1])) <= w <= min(width-1, round(right_edge[0] * h + right_edge[1]))

    while queue:
        h, w = queue.pop()

        if mask[h][w] != item_label:
            continue

        item_labels += 1

        min_w = min(w, min_w)
        max_w = max(w, max_w)

        h_nearest = max(h, h_nearest)
        
        if vanishing_height <= h+1 < height and valid_w(w, h+1):
            if mask[h+1][w] == item_label and used_points[h+1][w] == 0:
                used_points[h+1][w] = 1
                queue.append((h+1, w))

        if vanishing_height <= h-1 < height and valid_w(w, h-1):
            if mask[h-1][w] == item_label and used_points[h-1][w] == 0:
                used_points[h-1][w] = 1
                queue.append((h-1, w))

        if valid_w(w+1, h):
            if mask[h][w+1] == item_label and used_points[h][w+1] == 0:
                used_points[h][w+1] = 1
                queue.append((h, w+1))

        if valid_w(w-1, h):
            if mask[h][w-1] == item_label and used_points[h][w-1] == 0:
                used_points[h][w-1] = 1
                queue.append((h, w-1))

    return h_nearest, item_labels, max_w - min_w + 1


def dist(pixel_width, focal_length, road_width):
    return focal_length * road_width / pixel_width


def count_road_pixels_within_area(
    mask,
    background_label,
    left_edge,
    right_edge,
    h_nearest,
    h_farthest
):
    _, width = mask.shape
    
    road_labels = 0

    for h in range(h_farthest, h_nearest+1):
        for w in range(
            max(0, round(left_edge[0] * h + left_edge[1])), 
            min(width, round(right_edge[0] * h + right_edge[1]))
        ):
            if mask[h][w] != background_label:
                road_labels += 1

    return road_labels


# Returns item's area, width, length
def get_item_data(
    left_edge,
    right_edge,
    h_nearest,
    h_farthest,
    pixel_width,
    item_to_road_ratio,
    focal_length,
    road_width
):
    def calc_width(h):
        return (right_edge[0] - left_edge[0]) * h + right_edge[1] - left_edge[1]

    dist_to_nearest = dist(
        calc_width(h_nearest),
        focal_length,
        road_width
    )

    dist_to_farthest = dist(
        calc_width(h_farthest),
        focal_length,
        road_width
    )
    
    width = (dist_to_farthest + dist_to_nearest) / 2 * pixel_width / focal_length
    
    length = dist_to_farthest - dist_to_nearest
    
    road_area = length * road_width
    area = road_area * item_to_road_ratio

    return area, width, length


# Returns list of DamageItem's and used_points
def find_damage_items(
    mask,
    name2label,
    label2name,
    starting_height,
    focal_length,
    road_width,
    left_edge,
    right_edge
):
    damaged_surface_labels = set([
        name2label['water-puddle'],
        name2label['pothole'],
        name2label['cracks'],
        name2label['unpaved']
    ])

    height, width = mask.shape

    used_points = np.zeros(mask.shape)
    
    damage_items = []

    for h in range(starting_height, height):
        for w in range(
            max(0, round(left_edge[0] * h + left_edge[1])), 
            min(width, round(right_edge[0] * h + right_edge[1]))
        ):
            if used_points[h][w] == 1:
                continue

            if mask[h][w] not in damaged_surface_labels:
                continue

            h_farthest = h

            h_nearest, pixels_num, pixel_width = examine_item(
                mask, 
                left_edge,
                right_edge,
                h, 
                w, 
                used_points,
                starting_height
            )
            
            item_to_road_ratio = pixels_num / count_road_pixels_within_area(
                mask,
                name2label['background'],
                left_edge,
                right_edge,
                h_nearest,
                h_farthest
            )

            damage_item = DamageItem(
                *get_item_data(
                    left_edge,
                    right_edge,
                    h_nearest,
                    h_farthest,
                    pixel_width,
                    item_to_road_ratio,
                    focal_length,
                    road_width
                ),
                label2name[mask[h][w]]
            )

            damage_items.append(damage_item)

    return damage_items, used_points


def analyze_masks(
    masks_dir, 
    name2label,
    label2name,
    focal_length, 
    road_width
):
    masks_names = sorted(list(os.listdir(masks_dir)))
    
    damage_list = []

    for mask_name in masks_names:
        mask_path = os.path.join(masks_dir, mask_name)

        damage_items = analyze_mask(
            mask_path,
            name2label,
            label2name,
            focal_length,
            road_width
        )

        damage_list.append(damage_items)

    return damage_list


# Returns list of DamageItem's
def analyze_mask(
    path_to_mask, 
    name2label,
    label2name,
    focal_length, 
    road_width
):
    mask = cv2.imread(path_to_mask, cv2.IMREAD_UNCHANGED)

    vanishing_line = find_vanishing_line(mask, name2label['background'])
    
    if vanishing_line.height == -1:
        if DEBUG:
            print("Failed: bad vanishing line")
        return [FAILED_DAMAGE_ITEM]
    
    road_label = round(
        np.mean([
            mask[vanishing_line.height][w]
            for w in range(vanishing_line.start, vanishing_line.end + 1)
        ])
    )

    left_edge = find_edge(mask, vanishing_line, 'left', road_label)   
    right_edge = find_edge(mask, vanishing_line, 'right', road_label)   
    
    left_edge = least_squares_line(*left_edge)
    right_edge = least_squares_line(*right_edge)

    angle_left = np.arctan(left_edge[0]) 
    angle_right = np.arctan(right_edge[0])
    
    if (
        angle_left < -np.pi / 2.3
        or angle_left > -np.pi / 8
        or angle_right > np.pi / 2.3
        or angle_right < np.pi / 8
    ):
        if DEBUG:
            print("Failed: bad angles")
        return [FAILED_DAMAGE_ITEM]
    
    damage_items, used_points = find_damage_items(
        mask, 
        name2label,
        label2name,
        vanishing_line.height,
        focal_length,
        road_width,
        left_edge,
        right_edge
    )

    if DEBUG:
        import matplotlib.pyplot as plt
        
        height, width = mask.shape

        hs = np.array(range(height))
        
        plt.plot(left_edge[0] * hs + left_edge[1], height - hs, label='left')
        plt.plot(right_edge[0] * hs + right_edge[1], height - hs, label='right')
        # plt.plot(hs, left_edge[0] * hs + left_edge[1], label='left')
        # plt.plot(hs, right_edge[0] * hs + right_edge[1], label='right')
        plt.plot(
            list(range(vanishing_line.start, vanishing_line.end + 1)), 
            [height - vanishing_line.height] * (vanishing_line.end - vanishing_line.start + 1),
            label='vanishing line'
        )

        hs = []
        ws = []

        for h in range(height):
            for w in range(width):
                if used_points[h][w] == 1:
                    hs.append(height - h)
                    ws.append(w)

        plt.scatter(ws, hs, c='m', label='damage', s=1)
        
        plt.xlim([0, width])
        plt.ylim([0, height])
        plt.legend()

        plt.show()
    
    return damage_items


if __name__ == "__main__":
    baseline_dataset_dir = sys.argv[1]
    focal_length = int(sys.argv[2])
    road_width = int(sys.argv[3])
    DEBUG = bool(sys.argv[4])
    
    color2label = load_pydict(baseline_dataset_dir, 'color2label')
    color2name = load_pydict(baseline_dataset_dir, 'color2name')

    name2label = { color2name[color]: label for color, label in color2label.items() }
    label2name = { label: name for name, label in name2label.items() }

    relpath_to_mask = sys.argv[5]

    damage_items = analyze_mask(
        os.path.join(baseline_dataset_dir, relpath_to_mask),
        name2label,
        label2name,
        focal_length,
        road_width
    )

    for item in damage_items:
        print(item)
        print()
