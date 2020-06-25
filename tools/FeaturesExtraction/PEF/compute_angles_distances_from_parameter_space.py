import numpy as np
import argparse
import glob
import json
import os
import csv
from tools.Utils.EnumBody import BodyStraight
from scipy.spatial import distance


def compute_features_video(args):
    print(os.path.join(args.space_param_base_dir, args.input_dir))
    for root, directories, filenames in os.walk(os.path.join(args.space_param_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                angles_volume = []
                distances_euclidean_volume = []
                distances_mahalanobis_volume = []
                full_volume = []
                for frame in frames:
                    with open(frame, 'r') as fjson:
                        points_data = {int(k): v for k, v in json.load(fjson).items()}
                        angles = compute_adjacent_angles(points_data)
                        distances_euclidean = compute_adjacent_distances_euclidean(points_data)
                        distances_mahalanobis = compute_adjacent_distances_mahalanobis(points_data)

                        angles_volume.append(angles)
                        distances_euclidean_volume.append(distances_euclidean)
                        distances_mahalanobis_volume.append(distances_mahalanobis)

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                angles_file = os.path.join(features_dir, video_name + '_angles.json')
                distances_euclidean_file = os.path.join(features_dir, video_name + '_distances_euclidean.json')
                distances_mahanalobis_file = os.path.join(features_dir, video_name + '_distances_mahalanobis.json')

                save_csv(angles_file, angles_volume)
                save_csv(distances_euclidean_file, distances_euclidean_volume)
                save_csv(distances_mahanalobis_file, distances_mahalanobis_volume)


def save_csv(file_name, list_info):
    with open(file_name, "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_info)


def compute_adjacent_angles(points):
    angles = []
    # Upper Body
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Left_shoulder.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Right_shoulder.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Left_hip.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Right_hip.value][1])
    # Shoulders
    angles.append(points[BodyStraight.Left_shoulder.value][1] - points[BodyStraight.Neck.value][1])
    angles.append(points[BodyStraight.Right_shoulder.value][1] - points[BodyStraight.Neck.value][1])
    # Left Arm
    angles.append(points[BodyStraight.Left_forearm.value][1] - points[BodyStraight.Left_arm.value][1])
    angles.append(points[BodyStraight.Left_arm.value][1] - points[BodyStraight.Left_shoulder.value][1])
    # Right Arm
    angles.append(points[BodyStraight.Right_forearm.value][1] - points[BodyStraight.Right_arm.value][1])
    angles.append(points[BodyStraight.Right_arm.value][1] - points[BodyStraight.Right_shoulder.value][1])
    # Left Thigh
    angles.append(points[BodyStraight.Left_thigh.value][1] - points[BodyStraight.Left_hip.value][1])
    angles.append(points[BodyStraight.Left_thigh.value][1] - points[BodyStraight.Left_leg.value][1])
    # Right Thigh
    angles.append(points[BodyStraight.Right_thigh.value][1] - points[BodyStraight.Right_hip.value][1])
    angles.append(points[BodyStraight.Right_thigh.value][1] - points[BodyStraight.Right_leg.value][1])
    return angles


def compute_adjacent_distances_euclidean(points):
    distances = []

    # Upper Body
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Upper_body.value],
                                   points[BodyStraight.Left_shoulder.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Upper_body.value],
                                   points[BodyStraight.Right_shoulder.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Upper_body.value],
                                   points[BodyStraight.Left_hip.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Upper_body.value],
                                   points[BodyStraight.Right_hip.value]))
    # Shoulders
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Left_shoulder.value],
                                   points[BodyStraight.Neck.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Right_shoulder.value],
                                   points[BodyStraight.Neck.value]))
    # Left Arm
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Left_forearm.value],
                                   points[BodyStraight.Left_arm.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Left_arm.value],
                                   points[BodyStraight.Left_shoulder.value]))
    # Right Arm
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Right_forearm.value],
                                   points[BodyStraight.Right_arm.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Right_arm.value],
                                   points[BodyStraight.Right_shoulder.value]))
    # Left Thigh
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Left_thigh.value],
                                   points[BodyStraight.Left_hip.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Left_thigh.value],
                                   points[BodyStraight.Left_leg.value]))
    # Right Thigh
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Right_thigh.value],
                                   points[BodyStraight.Right_hip.value]))
    distances.append(
        compute_euclidean_distance(points[BodyStraight.Right_thigh.value],
                                   points[BodyStraight.Right_leg.value]))
    return distances


def compute_adjacent_distances_mahalanobis(points):
    distances = []
    inv_cov = compute_inverse_covariance(np.array(list(points.values())))

    # Upper Body
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Upper_body.value],
                                     points[BodyStraight.Left_shoulder.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Upper_body.value],
                                     points[BodyStraight.Right_shoulder.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Upper_body.value],
                                     points[BodyStraight.Left_hip.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Upper_body.value],
                                     points[BodyStraight.Right_hip.value],
                                     inv_cov))
    # Shoulders
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Left_shoulder.value],
                                     points[BodyStraight.Neck.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Right_shoulder.value],
                                     points[BodyStraight.Neck.value],
                                     inv_cov))
    # Left Arm
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Left_forearm.value],
                                     points[BodyStraight.Left_arm.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Left_arm.value],
                                     points[BodyStraight.Left_shoulder.value],
                                     inv_cov))
    # Right Arm
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Right_forearm.value],
                                     points[BodyStraight.Right_arm.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Right_arm.value],
                                     points[BodyStraight.Right_shoulder.value],
                                     inv_cov))
    # Left Thigh
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Left_thigh.value],
                                     points[BodyStraight.Left_hip.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Left_thigh.value],
                                     points[BodyStraight.Left_leg.value],
                                     inv_cov))
    # Right Thigh
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Right_thigh.value],
                                     points[BodyStraight.Right_hip.value],
                                     inv_cov))
    distances.append(
        compute_mahalanobis_distance(points[BodyStraight.Right_thigh.value],
                                     points[BodyStraight.Right_leg.value],
                                     inv_cov))

    return distances


def compute_angles(points, base_straight=BodyStraight.Upper_body):
    angles = []
    base_angle = points[base_straight.value]
    for k, v in points.items():
        if k != base_straight.value:
            dif = base_angle[0] - v[0]
            angles.append(dif)

    return angles


def compute_distances(points, base_straight=BodyStraight.Upper_body):
    distances = []
    base_angle = points[base_straight.value]
    for k, v in points.items():
        if k != base_straight.value:
            distances.append(compute_euclidean_distance(base_angle, v))

    return distances


def compute_euclidean_distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))


def compute_inverse_covariance(data):
    cov = np.cov(data.T)
    return np.linalg.inv(cov)


def compute_mahalanobis_distance(u, v, inv_cov):
    return distance.mahalanobis(u, v, inv_cov)


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Space Param Points to Human Action Recognition"
    )

    parser.add_argument("--space_param_base_dir", type=str,
                        default='/home/murilo/dataset/Weizmann',
                        help="Name of directory where space parameter points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='2DPoses_SpaceParam',
                        help="Name of input directory to computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='2DPoses_SpaceParam_Features',
                        help="Name of directory to output computed features.")

    args = parser.parse_args()
    print(args.output_dir)
    compute_features_video(args)


if __name__ == "__main__":
    main()
