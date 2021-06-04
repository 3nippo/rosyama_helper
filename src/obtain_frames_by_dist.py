import shutil
import sys
import pprint
import xml.etree.ElementTree as ET
from datetime import datetime
from geopy import distance
import os
import cv2

"""
track
video
dest_folder
"""

# Returns 
#   [((latitude, longitude), time), ...]
def parse_gps_data(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    fmt = '{{http://www.topografix.com/GPX/1/0}}{}'

    trk = root.find(fmt.format('trk'))
    
    trkseg = trk.find(fmt.format('trkseg'))
    
    points = []

    for trkpt in trkseg:
        point = (float(trkpt.attrib['lat']), float(trkpt.attrib['lon']))
        
        time = trkpt.find(fmt.format('time'))

        time = datetime.strptime(time.text, "%Y:%m:%d %H:%M:%S")

        points.append([point, time])

    return points

def cut_track_every_n_meters(track, n):
    if len(track) < 2:
        raise NotImplementedError

    cur_point, cur_time = track[0]
    
    next_point_idx = 1

    last_point = track[-1][0]

    EPS = 0.5

    def dist(a, b):
        return distance.distance(a, b).meters
    
    to_move = n

    cut_track = []

    while dist(cur_point, last_point) >= EPS or next_point_idx < len(track) - 1:
        next_point, next_time = track[next_point_idx]

        d = dist(cur_point, next_point)

        if d < to_move - EPS:
            to_move -= d
            
            cur_point = next_point
            cur_time = next_time
            next_point_idx += 1
        else:
            cut_track.append([cur_point, cur_time])
            
            if d < to_move + EPS:
                cur_point = next_point
                cur_time = next_time
                next_point_idx += 1
            else:
                lat = cur_point[0] + (next_point[0] - cur_point[0]) * to_move / d
                lon = cur_point[1] + (next_point[1] - cur_point[1]) * to_move / d

                cur_point = (lat, lon)
                cur_time = cur_time + (next_time - cur_time) * to_move / d

            to_move = n
    
    return cut_track


def transform_track_timestamps_to_seconds(track):
    base_timestamp = track[0][1]
    
    for i in range(len(track)):
        track[i][1] = (track[i][1] - base_timestamp).total_seconds()


def extract_frames_by_timestamps(video_path, dest_folder, track):
    if len(track) == 0:
        raise NotImplementedError
    
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    os.mkdir(dest_folder)

    video_stream = cv2.VideoCapture(video_path)
    
    cur_track_item_idx = 0
    
    frame_fmt = os.path.join(dest_folder, "{}.jpg")

    while video_stream.isOpened() and cur_track_item_idx < len(track):
        frame_exists, frame = video_stream.read()

        if not frame_exists:
            break

        msec_passed = video_stream.get(cv2.CAP_PROP_POS_MSEC)
        
        if msec_passed / 1000 >= track[cur_track_item_idx][1]:
            height, width, _ = frame.shape

            to_crop = (width - height) // 2

            cropped_frame = frame[:, to_crop:(width - to_crop), :]

            cv2.imwrite(frame_fmt.format(cur_track_item_idx), cropped_frame)
            cur_track_item_idx += 1


# Returns cut track
def obtain_frames_by_dist(gps_path, video_path, dest_folder, dist=7):
    track = parse_gps_data(gps_path)

    cut_track = cut_track_every_n_meters(track, dist)

    transform_track_timestamps_to_seconds(cut_track)

    extract_frames_by_timestamps(video_path, dest_folder, cut_track)

    return cut_track

if __name__ == "__main__":
    cut_track = obtain_frames_by_dist(*sys.argv[1:])

    pprint.pprint(cut_track)
