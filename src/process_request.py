import pprint
import sys
import shutil
import os
import subprocess

from src.obtain_frames_by_dist import obtain_frames_by_dist
from src.obtain_masks import obtain_masks, load_model
from src.analyze_damage import analyze_masks, load_pydict
from src.generate_report import generate_report
from src.send_email import send_email

WORK_DIR = 'work_dir'


class CantDownloadError(BaseException):
    def __init__(self, str):
        self._str = str

    def __str__(self):
        return "Download failed\n{}".format(self._str)


class BadError(BaseException):
    def __init__(self, str):
        self._str = str

    def __str__(self):
        return self._str


def download_to_work_dir(link):
    download_cmd = 'curl -s -J -L -O {}'.format(link)
    
    process = subprocess.Popen(
        download_cmd.split(),
        stderr=subprocess.PIPE,
        cwd=WORK_DIR
    )

    _, stderr = process.communicate()

    if process.returncode:
        raise CantDownloadError(stderr.decode())
    
    get_name_cmd = 'ls -t {}'.format(WORK_DIR)

    process = subprocess.Popen(
        get_name_cmd.split(),
        stdout=subprocess.PIPE
    )

    stdout, _ = process.communicate()
    
    if process.returncode:
        raise BadError("Download succeeded, but work_dir is empty")

    filename = stdout.decode().split()[0]

    filepath = os.path.join(
        WORK_DIR, 
        filename
    )

    return filepath


def extract_track(video_path):
    return 'out.gpx'


def load_name_label_dicts(dicts_dir):
    color2label = load_pydict(dicts_dir, 'color2label')
    color2name = load_pydict(dicts_dir, 'color2name')

    name2label = { color2name[color]: label for color, label in color2label.items() }
    label2name = { label: name for name, label in name2label.items() }
    
    return name2label, label2name


def process_request(
    model,
    name2label,
    label2name,
    gmail_login,
    request_data
):
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    os.mkdir(WORK_DIR)

    video_path = download_to_work_dir(request_data['video_download_link'])
    print("Video download finished")
    
    gpx_data_download_link = request_data['gpx_data_download_link']

    if gpx_data_download_link:
        track_path = download_to_work_dir(gpx_data_download_link)
        print("Track download finished")
    else:
        track_path = extract_track(video_path)
        print("Track extraction finished")
    
    frames_dir = os.path.join(WORK_DIR, 'frames')
    
    track = obtain_frames_by_dist(
        track_path,
        video_path,
        frames_dir
    )
    print("Frames obtained")

    masks_dir = os.path.join(WORK_DIR, 'masks')

    obtain_masks(
        frames_dir,
        model,
        masks_dir
    )
    print("Masks obtained")

    damage_list = analyze_masks(
        masks_dir,
        name2label,
        label2name,
        request_data['focal_length'],
        request_data['road_width']
    )
    print("Analysis completed")
    
    report_path = generate_report(
        damage_list,
        track,
        WORK_DIR,
        frames_dir,
        request_data
    )
    print("Report generated")

    send_email(
        gmail_login,
        request_data,
        report_path
    )
    print("Email sent")

    print("DONE")
    print()


if __name__ == '__main__':
    model_dir, mail, video_download_link, dicts_dir = sys.argv[1:]
    
    model = load_model(model_dir)
    
    name2label, label2name = load_name_label_dicts(dicts_dir)

    request_data = {
        'video_download_link': video_download_link,
        'focal_length': 35,
        'road_width': 6,
        'name': '',
        'gpx_data_download_link': '',
        'mail': mail
    }

    process_request(
        model, 
        name2label,
        label2name,
        request_data
    )
