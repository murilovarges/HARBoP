# Description..: Convert Openpose json files with body parts to hough points
# Date.........: 12/11/2018
# Author.......: Murilo Varges da Silva

import glob
import json
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import (hough_line, hough_line_peaks)

POSE_BODY_25_PAIRS_RENDER_GPU = \
    [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11,
     8, 12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16, 16, 18, 14,
     19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]

POSE_BODY_25_COLORS_RENDER_GPU = \
    [255, 0, 85,
     255, 0, 0,
     255, 85, 0,
     255, 170, 0,
     255, 255, 0,
     170, 255, 0,
     85, 255, 0,
     0, 255, 0,
     255, 0, 0,
     0, 255, 85,
     0, 255, 170,
     0, 255, 255,
     0, 170, 255,
     0, 85, 255,
     0, 0, 255,
     255, 0, 170,
     170, 0, 255,
     255, 0, 255,
     85, 0, 255,
     0, 0, 255,
     0, 0, 255,
     0, 0, 255,
     0, 255, 255,
     0, 255, 255,
     0, 255, 255]


def main():
    parser = argparse.ArgumentParser(
        description="Compute features from Hough Points to Human Action Recognition"
    )

    parser.add_argument("--key_points_base_dir", type=str,
                        default='/home/murilo/dataset/KTH',
                        help="Name of directory where input points are located.")

    parser.add_argument("--input_dir", type=str,
                        default='VideosTrainValidationTest_OP_PartKeys',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_points_dir", type=str,
                        default='VideosTrainValidationTest_Hough_points',
                        help="Name of directory to output computed features.")

    parser.add_argument("--output_images_dir", type=str,
                        default='VideosTrainValidationTest_Hough_images',
                        help="Name of directory to output hough images.")

    parser.add_argument("--image_height", type=int,
                        default='240',
                        help="(Frame Size)Image height to compute max distance in hough space.")

    parser.add_argument("--image_width", type=int,
                        default='320',
                        help="(Frame Size)Image width to compute max distance in hough space.")

    parser.add_argument("--save_image", type=int,
                        default='1',
                        help="Whether save image with points in hough space.")

    parser.add_argument("--draw_body_ids", type=int,
                        default='1',
                        help="Whether draw body ids in image with points in hough space.")

    args = parser.parse_args()
    compute_hough_point(args)


def compute_hough_point(args):
    # here compute image diagonal = max distance in hough space
    max_distance = int(((args.image_height**2) + (args.image_width**2))**(1/2))
    print(max_distance)

    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180)

    key_points_dir = os.path.join(args.key_points_base_dir, args.input_dir)

    frames_ctd = 0
    key_points_files = sorted(glob.glob(key_points_dir + "/**/*.json", recursive=True))
    print('Frames to process: %i' % len(key_points_files))
    for key_points_file in key_points_files:
        if frames_ctd % 100 == 0:
            print('Frame: %i from: %i' % (frames_ctd, len(key_points_files)))
            print(key_points_file)

        body_parts = read_body_parts_file(key_points_file)
        if len(body_parts) > 0:
            file_name_points = os.path.basename(key_points_file)
            points_hough_dir = key_points_file.replace(args.input_dir, args.output_points_dir)
            points_hough_dir = os.path.dirname(points_hough_dir)
            points_hough_name = os.path.join(points_hough_dir, file_name_points)
            if not os.path.exists(points_hough_dir):
                os.makedirs(points_hough_dir)

            # compute hough point and draw image with points
            img_hough, points_hough = draw_hough(body_parts, max_distance, thetas, args.draw_body_ids)

            with open(points_hough_name, 'w') as fjson:
                json.dump(points_hough, fjson)

            if args.save_image:
                file_name_img = os.path.basename(key_points_file)
                file_name_img = file_name_img.replace('_keypoints.json', '.png')
                img_hough_dir = key_points_file.replace(args.input_dir, args.output_images_dir)
                img_hough_dir = os.path.dirname(img_hough_dir)
                img_hough_full_name = os.path.join(img_hough_dir, file_name_img)
                if not os.path.exists(img_hough_dir):
                    os.makedirs(img_hough_dir)
                img_hough.save(img_hough_full_name)

        frames_ctd = frames_ctd + 1


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


def draw_hough(body_parts, max_distance, thetas, draw_body_ids=True):
    # Create image degrees x max_distance
    img_hough = Image.new('RGB', (180 + 20, int(max_distance/2)), color='black')
    points_hough = {}
    draw = ImageDraw.Draw(img_hough)
    for i in range(0, 14, 1):
        degree = degree_calc = theta = rho1 = rho2 = 0
        x1, y1, x2, y2, color_id, id1, id2 = return_body_points_coord(i, body_parts)
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            print(i)
            #print('x1:\t%i\ty1:\t%i\t\tx2:\t%i\ty2:\t%i' % (x1, y1, x2, y2))
            if y1 - y2 != 0:
                theta = np.arctan((x2 - x1) / (y1 - y2))
            else:
                theta = 0

            # here convert theta from radians to degrees
            degree_calc = int(round(theta * (180/np.pi)))

            # here find theta in thetas discrete list
            degree = min(range(len(thetas)), key=lambda x: abs(thetas[x] - theta))
            position_min_degree = min(thetas, key=lambda x: abs(x - theta))

            # compute rho from theta
            rho1 = x1 * np.cos(theta) + y1 * np.sin(theta)
            rho2 = x2 * np.cos(theta) + y2 * np.sin(theta)
            #print(rho1,rho2)

            print(int(rho1), int(degree), x1, y1)
            # draw ellipse that represent body part in hough space
            draw.ellipse((degree - 6, abs(rho1) - 6, degree + 6,abs(rho1) + 6), fill=get_color(color_id))

            if draw_body_ids:
                font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 10)
                draw.text((degree, abs(rho1)), '%i-%i' % (id1, id2), font=font, fill=(255, 255, 255, 128))

            #print('theta Calc:\t%.4f\t\trho Calc:\t\t%i' % (theta, rho1))
            #print('Theta Find:\t%.4f\t\tAngulo:\t\t%.2f' % (position_min_degree, degree))
            #print('\n')

        #points_hough.append((degree, degree_calc, theta,int(rho)))
        points_hough[i] = (degree, degree_calc, theta, int(rho1))

    return img_hough, points_hough


def return_body_points_coord(i, body_parts):
    x1 = y1 = x2 = y2 = x = color_id = id1 = id2 = 0
    if i == 0:  # 1 => 0 Neck
        x = 13
    elif i == 1:  # 1 => 8 Upper body
        x = 0
    elif i == 2:  # 2 => 3 Right Arm
        x = 3
    elif i == 3:  # 3 => 4 Right Forearm
        x = 4
    elif i == 4:  # 5 => 6 Left Arm
        x = 5
    elif i == 5:  # 6 => 7 Left Forearm
        x = 6
    elif i == 6:  # 9 => 10 Right Thigh
        x = 8
    elif i == 7:  # 10 => 11 Right Leg
        x = 9
    elif i == 8:  # 12 => 13 Left Thigh
        x = 11
    elif i == 9:  # 13 => 14 Left Leg
        x = 12
    elif i == 10:  # 8 => 9 Right Hip
        x = 7
    elif i == 11:  # 8 => 12 Left Hip
        x = 10
    elif i == 12:  # 1 => 2 Right Shoulder
        x = 1
    elif i == 13:  # 1 => 5 Left Shoulder
        x = 2

    x = x * 2
    if (len(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]]) > 0 and len(
            body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]]) > 0):
        x1, y1 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]])
        x2, y2 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]])
        color_id = POSE_BODY_25_PAIRS_RENDER_GPU[x + 1] * 3
        id1 = POSE_BODY_25_PAIRS_RENDER_GPU[x]
        id2 = POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]

    return x1, y1, x2, y2, color_id, id1, id2


def draw_hough_body_part(body_ids, body_parts, height, width, thetas):
    img = Image.new('1', (width, height), color='black')
    draw = ImageDraw.Draw(img)

    for i in body_ids:
        x = 0
        if i == 0:  # 1 => 0 Neck
            x = 13
        elif i == 1:  # 1 => 8 Upper body
            x = 0
        elif i == 2:  # 2 => 3 Right Arm
            x = 3
        elif i == 3:  # 3 => 4 Right Forearm
            x = 4
        elif i == 4:  # 5 => 6 Left Arm
            x = 5
        elif i == 5:  # 6 => 7 Left Forearm
            x = 6
        elif i == 6:  # 9 => 10 Right Thigh
            x = 8
        elif i == 7:  # 10 => 11 Right Leg
            x = 9
        elif i == 8:  # 12 => 13 Left Thigh
            x = 11
        elif i == 9:  # 13 => 14 Left Leg
            x = 12

        x = x * 2
        if (len(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]]) > 0 and len(
                body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]]) > 0):
            x1, y1 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]])
            x2, y2 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]])

            draw.line((x1, y1, x2, y2), fill='white')

    # img.show()
    img_nd = np.asarray(img, dtype="int32")
    h, theta, d = hough_line(img_nd)
    h1, theta1, d1 = hough_line_peaks(h, theta, d)
    aaa = min(thetas, key=lambda x: abs(x - theta1[0]))

    bbb = min(range(len(thetas)), key=lambda x: abs(thetas[x] - theta1[0]))
    print('Theta Hough:\t%.4f\t\tp Hough:\t%i' % (theta1[0], d1[0]))
    print('Theta Find:\t%.4f\t\tAngulo:\t\t%.2f' % (aaa, bbb))

    h = h * 10
    h_space_color = Image.fromarray(np.uint8(h)).convert('RGB')
    pixels = h_space_color.load()
    for lin in range(h_space_color.size[0]):  # for every pixel:
        for col in range(h_space_color.size[1]):
            if 0 in body_ids:
                pixels[lin, col] = (pixels[lin, col][0], 0, 0)  # red
            elif 2 in body_ids:
                pixels[lin, col] = (0, pixels[lin, col][0], 0)  # green
            elif 4 in body_ids:
                pixels[lin, col] = (0, 0, pixels[lin, col][0])  # blue
            elif 6 in body_ids:
                pixels[lin, col] = (pixels[lin, col][0], pixels[lin, col][0], 0)  # yellow
            elif 8 in body_ids:
                pixels[lin, col] = (pixels[lin, col][0], 0, pixels[lin, col][0])  # pink

    h_space_color.save('h_space_color_%i.png' % i)
    # return h_space_color, h, theta, d, img_nd
    return img


def draw_body(body_parts, height, width):
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)

    for k in sorted(body_parts):
        if len(body_parts[k]) > 0:
            x, y = get_max_prob(body_parts[k])
            draw.point((x, y), fill=get_color(k * 3))

    ctd = 0
    for x in range(0, len(POSE_BODY_25_PAIRS_RENDER_GPU), 2):
        print(x, x + 1)
        print(POSE_BODY_25_PAIRS_RENDER_GPU[x], POSE_BODY_25_PAIRS_RENDER_GPU[x + 1])
        print(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]], body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]])
        print('\n')
        if (len(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]]) > 0 and len(
                body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]]) > 0):
            x1, y1 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x]])
            x2, y2 = get_max_prob(body_parts[POSE_BODY_25_PAIRS_RENDER_GPU[x + 1]])
            draw.line((x1, y1, x2, y2), fill=get_color(POSE_BODY_25_PAIRS_RENDER_GPU[x + 1] * 3), width=1)
            ctd = ctd + 1
    print(ctd)

    img.show()
    img.save('pil_red.png')


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


def get_color(k):
    return POSE_BODY_25_COLORS_RENDER_GPU[k], \
           POSE_BODY_25_COLORS_RENDER_GPU[k + 1], \
           POSE_BODY_25_COLORS_RENDER_GPU[k + 2]


if __name__ == "__main__":
    main()
