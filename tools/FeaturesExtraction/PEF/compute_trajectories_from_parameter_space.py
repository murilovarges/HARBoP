import glob
import json
import os
import argparse
import numpy as np


def compute_features_trajectories(args):
    points = 14
    for root, directories, filenames in os.walk(os.path.join(args.space_param_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            features_video_norm1 = []
            features_video_norm2 = []
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                for x in range(0, len(frames), args.stride):
                    if x + args.number_frames < len(frames):
                        features = np.zeros(shape=(points, args.number_frames, 2))
                        prev_points_data = None
                        for y in range(x, x + args.number_frames + 1):
                            with open(frames[y], 'r') as fjson:
                                points_data = {int(k): v for k, v in json.load(fjson).items()}

                            if prev_points_data is None:
                                prev_points_data = points_data
                            else:
                                diffs = compute_displacement_space_param(prev_points_data, points_data)

                                idx = (y - 1) - x
                                for a in range(points):
                                    features[a, idx] = diffs[a]

                        features_norm1 = features.copy()
                        features_norm2 = features.copy()
                        # Here normalize1
                        for y in range(points):
                            for k in range(2):
                                s = np.sum(abs(features_norm1[y, :, k]))
                                if s != 0:
                                    features_norm1[y, :, k] = features_norm1[y, :, k] / s

                        # Here normalize 2
                        for y in range(points):
                            for k in range(features_norm2.shape[1]):
                                s = np.sum(abs(features_norm2[y, k, :]))
                                if s != 0:
                                    features_norm2[y, k, :] = features_norm2[y, k, :] / s

                        features_video_norm1.append(features_norm1.flatten())
                        features_video_norm2.append(features_norm2.flatten())

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                file = os.path.join(features_dir, video_name + '_trajectories_norm1_features.json')
                np.savetxt(file, np.asarray(features_video_norm1), delimiter=',', fmt='%.7f')

                file = os.path.join(features_dir, video_name + '_trajectories_norm2_features.json')
                np.savetxt(file, np.asarray(features_video_norm2), delimiter=',', fmt='%.7f')


def compute_displacement_space_param(prev_body_parts, body_parts):
    diffs = np.zeros(shape=(14, 2))
    for x in range(14):
        x1 = prev_body_parts[x][0]
        y1 = prev_body_parts[x][1]
        x2 = body_parts[x][0]
        y2 = body_parts[x][1]
        diffs[x, :] = (x2 - x1, y2 - y1)

    return diffs


def compute_displacement_cartesian(prev_body_parts, body_parts):
    diffs = np.zeros(shape=(15, 2))
    for x in range(15):
        x1, y1 = get_max_prob(prev_body_parts[x])
        x2, y2 = get_max_prob(body_parts[x])
        diffs[x, :] = (x2 - x1, y2 - y1)

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

    parser.add_argument("--space_param_base_dir", type=str,
                        default='/home/murilo/dataset/Weizmann',
                        help="Name of directory where space parameter points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='2DPoses_SpaceParam',
                        help="Name of input directory to computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='2DPoses_SpaceParam_Trajectories_l20_s1',
                        help="Name of directory to output computed features.")

    parser.add_argument("--number_frames", type=int,
                        default=20,
                        help="Number of frames to extract features.")

    parser.add_argument("--stride", type=int,
                        default=1,
                        help="Stride to compute features from the frames.")

    args = parser.parse_args()

    print(args)
    compute_features_trajectories(args)


if __name__ == "__main__":
    main()


