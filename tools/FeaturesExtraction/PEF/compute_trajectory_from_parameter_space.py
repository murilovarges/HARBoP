# Description..: Read Openpose json files with body parts
# Date.........: 10/04/2019
# Author.......: Murilo Varges da Silva

import glob
import json
import os
import argparse
import numpy as np

''' To Remove Further
def compute_features_trajectories(args):
    for dataset in datasets:
    #if 1 == 1:
        for action in actions:
            base_dir = os.path.join(args.points_base_dir, args.input_dir, dataset, action)
            #base_dir = os.path.join(args.points_base_dir, args.input_dir, action)
            print(base_dir)
            videos = sorted(glob.glob(base_dir + '/*'))
            for video_dir in videos:
                print(video_dir)
                features_video = []
                frames = sorted(glob.glob(video_dir + '/*.json'))
                for x in range(0, len(frames), args.stride):
                    if x + args.number_frames < len(frames):
                        if args.type == 'cartesian':
                            points = 15
                        else:
                            points = 14
                        features = np.zeros(shape=(points, args.number_frames, 2))
                        prev_points_data = None
                        for y in range(x, x + args.number_frames+1):
                            if args.type == 'cartesian':
                                points_data = read_body_parts_file(frames[y])
                            else:
                                with open(frames[y], 'r') as fjson:
                                    points_data = {int(k): v for k, v in json.load(fjson).items()}

                            if prev_points_data is None:
                                prev_points_data = points_data
                            else:
                                if args.type == 'cartesian':
                                    diffs = compute_displacement_cartesian(prev_points_data, points_data )
                                else:
                                    diffs = compute_displacement_hough(prev_points_data, points_data)

                                idx = (y - 1) - x
                                for a in range(points):
                                    features[a,idx] = diffs[a]

                        # Here normalize
                        for y in range(points):
                            for k in range(2):
                                s = np.sum(abs(features[y, :, k]))
                                if s != 0:
                                    features[y, :, k] = features[y, :, k] / s
                                else:
                                    print('hey there!')
                                    print(features[y, :, k])

                        features_video.append(features.flatten())

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                file = os.path.join(features_dir, video_name + '_trajectories_features.json')
                np.savetxt(file, np.asarray(features_video), delimiter=',', fmt='%.7f')
'''


def compute_features_trajectories(args):
    points = 14
    for root, directories, filenames in os.walk(os.path.join(args.points_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            features_video = []
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                for x in range(0, len(frames), args.stride):
                    if x + args.number_frames < len(frames):
                        features = np.zeros(shape=(points, args.number_frames, 2))
                        prev_points_data = None
                        for y in range(x, x + args.number_frames+1):
                            with open(frames[y], 'r') as fjson:
                                points_data = {int(k): v for k, v in json.load(fjson).items()}

                            if prev_points_data is None:
                                prev_points_data = points_data
                            else:
                                diffs = compute_displacement_hough(prev_points_data, points_data)

                                idx = (y - 1) - x
                                for a in range(points):
                                    features[a,idx] = diffs[a]

                        # Here normalize
                        for y in range(points):
                            for k in range(2):
                                s = np.sum(abs(features[y, :, k]))
                                if s != 0:
                                    features[y, :, k] = features[y, :, k] / s

                        features_video.append(features.flatten())

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                file = os.path.join(features_dir, video_name + '_trajectories_features.json')
                np.savetxt(file, np.asarray(features_video), delimiter=',', fmt='%.7f')


def compute_displacement_hough(prev_body_parts, body_parts):
    diffs = np.zeros(shape=(14,2))
    for x in range(14):
        x1 = prev_body_parts[x][0]
        y1 = prev_body_parts[x][3]
        x2 = body_parts[x][0]
        y2 = body_parts[x][3]
        diffs[x,:] = (x2 - x1, y2 - y1)

    return diffs


def compute_displacement_cartesian(prev_body_parts, body_parts):
    diffs = np.zeros(shape=(15,2))
    for x in range(15):
        x1, y1 = get_max_prob(prev_body_parts[x])
        x2, y2 = get_max_prob(body_parts[x])
        diffs[x,:] = (x2 - x1, y2 - y1)

    return diffs


def read_body_parts_file(key_points_file):
    body_parts_int = {}

    # Read json pose points
    with open(key_points_file) as f:
        data = json.load(f)

    body_parts = data['part_candidates'][0]
    if len(body_parts) > 0:

        for key, value in body_parts.items():
            body_parts_int[int(key)] = [item for item in value]

    return body_parts_int


def get_max_prob(body_part):
    m = 0
    x = 0
    y = 0
    for p in range(0, len(body_part), 3):
        if body_part[p + 2] > m:
            m = float(body_part[p + 2])
            x = int(body_part[p])
            y = int(body_part[p + 1])

    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory features from OpenPose points to Human Action Recognition"
    )

    parser.add_argument("--points_base_dir", type=str,
                        default='/home/murilo/dataset/KTH',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='VideosTrainValidationTest_Hough_points',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='VideosTrainValidationTest_Hough_features_trajectories_l20_s10_all',
                        help="Name of directory to output computed features.")

    parser.add_argument("--number_frames", type=int,
                        default=20,
                        help="Number of frames to extract features.")

    parser.add_argument("--stride", type=int,
                        default=10,
                        help="Stride to compute features from the frames.")

    args = parser.parse_args()

    print(args)
    compute_features_trajectories(args)


if __name__ == "__main__":
    main()
