import numpy as np
import argparse
import glob
import json
import os
from Enumerators import BodyStraight
import csv
import itertools

#frames_dir = '/home/murilo/dataset/KTH/VideosTrainValidationTest_Hough3'
#videos_dir = 'VideosTrainValidationTest_Hough3_Videos'

datasets = ['test','training','validation']
actions = ['boxing','handclapping','handwaving','jogging','running','walking']
#actions = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk', 'wave1', 'wave2']

''' To Remove Further
def compute_features_video(args):
    print(args)
    for dataset in datasets:
        for action in actions:
            base_dir = os.path.join(args.points_base_dir, args.input_dir, dataset, action)
            #base_dir = os.path.join(args.points_base_dir, args.input_dir, action)
            print(base_dir)
            videos = sorted(glob.glob(base_dir + '/*'))
            for video_dir in videos:
                print(video_dir)
                frames = sorted(glob.glob(video_dir + '/*.json'))
                angles_volume = []
                angles_adj_volume = []
                distances_volume = []
                distances_adj_volume = []
                full_volume = []
                for frame in frames:
                    with open(frame, 'r') as fjson:
                        points_data = {int(k): v for k, v in json.load(fjson).items()}
                        angles = compute_angles(points_data)
                        angles_adj = compute_adjacent_angles(points_data)
                        distances = compute_distances(points_data)
                        distances_adj = compute_adjacent_distances(points_data)

                        angles_volume.append(angles)
                        angles_adj_volume.append(angles_adj)
                        distances_volume.append(distances)
                        distances_adj_volume.append(distances_adj)
                        full_volume.append(angles + angles_adj + distances + distances_adj)

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                angles_file = os.path.join(features_dir, video_name + '_angles.json')
                angles_adj_file = os.path.join(features_dir, video_name + '_adjangles.json')
                distances_file = os.path.join(features_dir, video_name + '_distances.json')
                distances_adj_file = os.path.join(features_dir, video_name + '_adjdistances.json')
                full_file = os.path.join(features_dir, video_name + '_full.json')

                save_csv(angles_file,angles_volume)
                save_csv(angles_adj_file,angles_adj_volume)
                save_csv(distances_file,distances_volume)
                save_csv(distances_adj_file,distances_adj_volume)
                save_csv(full_file, full_volume)
'''


def compute_features_video(args):
    for root, directories, filenames in os.walk(os.path.join(args.points_base_dir, args.input_dir)):
        for directory in directories:
            video_dir = os.path.join(root, directory)
            print(video_dir)
            frames = sorted(glob.glob(video_dir + '/*.json'))
            if len(frames) > 0:
                angles_volume = []
                angles_adj_volume = []
                distances_volume = []
                distances_adj_volume = []
                full_volume = []
                for frame in frames:
                    with open(frame, 'r') as fjson:
                        points_data = {int(k): v for k, v in json.load(fjson).items()}
                        angles = compute_angles(points_data)
                        angles_adj = compute_adjacent_angles(points_data)
                        distances = compute_distances(points_data)
                        distances_adj = compute_adjacent_distances(points_data)

                        angles_volume.append(angles)
                        angles_adj_volume.append(angles_adj)
                        distances_volume.append(distances)
                        distances_adj_volume.append(distances_adj)
                        full_volume.append(angles + angles_adj + distances + distances_adj)

                features_dir = video_dir.replace(args.input_dir, args.output_dir)
                features_dir, video_name = os.path.split(features_dir)
                if not os.path.exists(features_dir):
                    os.makedirs(features_dir)

                angles_file = os.path.join(features_dir, video_name + '_angles.json')
                angles_adj_file = os.path.join(features_dir, video_name + '_adjangles.json')
                distances_file = os.path.join(features_dir, video_name + '_distances.json')
                distances_adj_file = os.path.join(features_dir, video_name + '_adjdistances.json')
                full_file = os.path.join(features_dir, video_name + '_full.json')

                save_csv(angles_file,angles_volume)
                save_csv(angles_adj_file,angles_adj_volume)
                save_csv(distances_file,distances_volume)
                save_csv(distances_adj_file,distances_adj_volume)
                save_csv(full_file, full_volume)


def save_csv(file_name, list_info):
    with open(file_name, "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_info)


def compute_features(args):
    print(args)

    frames_ctd = 0

    for dataset in datasets:
        for action in actions:
            base_dir = os.path.join(args.points_base_dir, args.input_dir, dataset, action)
            print(base_dir)
            videos = sorted(glob.glob(base_dir + '/*'))
            for video_dir in videos:
                print(video_dir)
                frames = sorted(glob.glob(video_dir + '/*.json'))
                for x in range(0,len(frames),args.stride):
                    angles_volume = []
                    angles_adj_volume = []
                    distances_volume = []
                    distances_adj_volume = []
                    #last_frame = ('%012d' % (x + args.number_frames-1))
                    #if any(last_frame in s for s in frames):
                    if x + args.number_frames-1 < len(frames):
                        for y in range(x,x+args.number_frames):
                            with open(frames[y], 'r') as fjson:
                                points_data = {int(k): v for k, v in json.load(fjson).items()}
                                angles_volume.append(compute_angles(points_data))
                                angles_adj_volume.append(compute_adjacent_angles(points_data))
                                distances_volume.append(compute_distances(points_data))
                                distances_adj_volume.append(compute_adjacent_distances(points_data))

                        features_dir = video_dir.replace(args.input_dir, args.output_dir)
                        if not os.path.exists(features_dir):
                            os.makedirs(features_dir)

                        angles_file = os.path.join(features_dir,'%012d_angles.json' % x)
                        angles_adj_file = os.path.join(features_dir, '%012d_adjangles.json' % x)
                        distances_file = os.path.join(features_dir, '%012d_distances.json' % x)
                        distances_adj_file = os.path.join(features_dir, '%012d_adjdistances.json' % x)
                        full_file = os.path.join(features_dir, '%012d_full.json' % x)
                        with open(angles_file, 'w') as fjson:
                            json.dump(angles_volume, fjson)
                        with open(angles_adj_file, 'w') as fjson:
                            json.dump(angles_adj_volume, fjson)
                        with open(distances_file, 'w') as fjson:
                            json.dump(distances_volume, fjson)
                        with open(distances_adj_file, 'w') as fjson:
                            json.dump(distances_adj_volume, fjson)
                        with open(full_file, 'w') as fjson:
                            json.dump(angles_volume + angles_adj_volume + distances_volume + distances_adj_volume, fjson)

                    else:
                        break
                '''
                for frame in frames:
                    with open(frame, 'r') as fjson:
                        points_data = {int(k): v for k, v in json.load(fjson).items()}
                        print(points_data)
                        angles = compute_angles(points_data)
                        distances = compute_distances(points_data)
                
                    frames_ctd = frames_ctd + 1
                '''


def compute_adjacent_angles(points):
    angles = []
    '''
    # Upper Body
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Neck.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Left_shoulder.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Right_shoulder.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Left_hip.value][1])
    angles.append(points[BodyStraight.Upper_body.value][1] - points[BodyStraight.Right_hip.value][1])
    # Left Forearm
    angles.append(points[BodyStraight.Left_forearm.value][1] - points[BodyStraight.Left_arm.value][1])
    angles.append(points[BodyStraight.Left_forearm.value][1] - points[BodyStraight.Left_shoulder.value][1])
    Right Forearm
    angles.append(points[BodyStraight.Right_forearm.value][1] - points[BodyStraight.Right_arm.value][1])
    angles.append(points[BodyStraight.Right_forearm.value][1] - points[BodyStraight.Right_shoulder.value][1])
    # Left Thigh
    angles.append(points[BodyStraight.Left_thigh.value][1] - points[BodyStraight.Left_hip.value][1])
    angles.append(points[BodyStraight.Left_thigh.value][1] - points[BodyStraight.Left_leg.value][1])
    # Right Thigh
    angles.append(points[BodyStraight.Right_thigh.value][1] - points[BodyStraight.Right_hip.value][1])
    angles.append(points[BodyStraight.Right_thigh.value][1] - points[BodyStraight.Right_leg.value][1])
    '''

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


def compute_adjacent_distances(points):
    distances = []
    '''
    # Upper Body
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Neck.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Left_shoulder.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Right_shoulder.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Left_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Right_hip.value]))
    # Left Forearm
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_forearm.value],points[BodyStraight.Left_arm.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_forearm.value],points[BodyStraight.Left_shoulder.value]))
    # Right Forearm
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_forearm.value],points[BodyStraight.Right_arm.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_forearm.value],points[BodyStraight.Right_shoulder.value]))
    # Left Thigh
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_thigh.value],points[BodyStraight.Left_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_thigh.value],points[BodyStraight.Left_leg.value]))
    # Right Thigh
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_thigh.value],points[BodyStraight.Right_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_thigh.value],points[BodyStraight.Right_leg.value]))
    '''
    # Upper Body
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Left_shoulder.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Right_shoulder.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Left_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Upper_body.value],points[BodyStraight.Right_hip.value]))
    # Shoulders
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_shoulder.value],points[BodyStraight.Neck.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_shoulder.value],points[BodyStraight.Neck.value]))
    # Left Arm
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_forearm.value],points[BodyStraight.Left_arm.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_arm.value],points[BodyStraight.Left_shoulder.value]))
    # Right Arm
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_forearm.value],points[BodyStraight.Right_arm.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_arm.value],points[BodyStraight.Right_shoulder.value]))
    # Left Thigh
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_thigh.value],points[BodyStraight.Left_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Left_thigh.value],points[BodyStraight.Left_leg.value]))
    # Right Thigh
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_thigh.value],points[BodyStraight.Right_hip.value]))
    distances.append(compute_euclidean_distance(points[BodyStraight.Right_thigh.value],points[BodyStraight.Right_leg.value]))

    return distances


def compute_angles(points, base_straight=BodyStraight.Upper_body):
    angles = []
    base_angle = points[base_straight.value]
    for k, v in points.items():
        if k != base_straight.value:
            dif = base_angle[1] - v[1]
            angles.append(dif)

    return angles


def compute_distances(points, base_straight=BodyStraight.Upper_body):
    distances = []
    base_angle = points[base_straight.value]
    for k, v in points.items():
        if k != base_straight.value:
            distances.append(compute_euclidean_distance(base_angle,v))

    return distances


def compute_euclidean_distance(p1,p2):
    return np.sqrt(np.power(p1[0]-p2[0],2) + np.power(p1[3]-p2[3],2))


def compute_distances2(points):
    distances = []
    for x in range(14):
        for y in range(14):
            if y > x:
                dist = np.sqrt(np.power(points[x][0]-points[y][0],2) + np.power(points[x][3]-points[y][3],2))
                distances.append(dist)

    return distances


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Hough Points to Human Action Recognition"
    )

    parser.add_argument("--points_base_dir", type=str,
                        default='/home/murilo/dataset/KTH',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='VideosTrainValidationTest_Hough_Points_ope',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_dir", type=str,
                        default='VideosTrainValidationTest_Hough_Features_BOW_New',
                        help="Name of directory to output computed features.")

    parser.add_argument("--number_frames", type=int,
                        default=32,
                        help="Number of frames to extract features.")

    parser.add_argument("--stride", type=int,
                        default=1,
                        help="Stride to compute features from the frames.")

    parser.add_argument("--one_file_per_video",
                        type=int,
                        default=1,
                        help="Whether to save feature in one file per video")

    args = parser.parse_args()

    print(args.output_dir)
    print(args.number_frames)

    if args.one_file_per_video:
        compute_features_video(args)
    else:
        compute_features(args)


if __name__ == "__main__":
    main()
