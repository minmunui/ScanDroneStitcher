import argparse
import os
import time

import cv2
from tqdm import tqdm

from src.gps import get_gps_from_image, get_angles, determine_rotation_angles
from src.img_io import make_output_name
from template_match import search_locations

DISCARDED = -1
ORIGINAL = 0
ROTATED = 1

ROT = {
    '0': "NO ROTATION",
    '-1': "DISCARD",
    '1': "ROTATE 180",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input directory name', nargs='?', default="input")
    parser.add_argument('-o', '--output', type=str, help='output file name', nargs='?', default="result")
    parser.add_argument('-l', '--log', type=str, help='is remain log at console', nargs='?', default="log")
    parser.add_argument('-r', '--resize', type=int, help='resize images', nargs='?', default=1)
    parser.add_argument('-p', '--pano_conf', type=float, help='panorama confidence', nargs='?', default=1.0)
    parser.add_argument('-d', '--drop', type=int, help='ratio of drop', nargs='?', default=1)
    args = parser.parse_args()

    input_dir = args.input  # 입력 디렉토리 이름
    is_log = False
    if args.log == "log":
        is_log = True
    input_path = os.path.join(os.getcwd(), "data", "input", input_dir)
    image_names = os.listdir(input_path)

    image_paths = [os.path.join(input_path, image_name) for image_name in image_names]

    coordinates = []

    for image_path in tqdm(image_paths, desc="extract gps"):
        coordinates.append(get_gps_from_image(img_path=image_path)[:2])

    angles = get_angles(coordinates)

    rotate = determine_rotation_angles(angles)

    if is_log:
        for idx, image_name in enumerate(image_names):
            print(f"{idx} \t {image_name}")
        for idx, angle in enumerate(angles):
            print(f"angle {idx} \t {angle}")
        for idx, r in enumerate(rotate):
            print(f"rotate {idx} \t {ROT[str(r)]}")

    discard_index = []

    for i in range(len(rotate)):
        if rotate[i] == DISCARDED:
            discard_index.append(i)

    for i in list(reversed(discard_index)):
        del image_names[i]
        del angles[i]
        del rotate[i]
        del coordinates[i]
        del image_paths[i]

    images = []

    for i in tqdm(range(len(image_paths)), desc="read images"):
        image = cv2.imread(image_paths[i])
        if rotate[i] == ROTATED:
            image = cv2.rotate(image, cv2.ROTATE_180)
        images.append(image)



    output_dir = make_output_name()

    if args.output != "result":
        output_dir = args.output
    output_base = os.path.join(os.getcwd(), "data", "output", output_dir)
    os.makedirs(output_base, exist_ok=True)

    if args.drop != 1:
        images = [images[i] for i in range(len(images)) if i % args.drop == 0]
        image_names = [image_names[i] for i in range(len(image_names)) if i % args.drop == 0]
        coordinates = [coordinates[i] for i in range(len(coordinates)) if i % args.drop == 0]

    for idx, image_name in enumerate(image_names):
        cv2.imwrite(os.path.join(os.getcwd(), "data", "output", output_dir, f"{idx}_{image_name}"), images[idx])

    print(f"result will be saved on {output_base}")
    print(f"selected images : {image_names}")
    stitcher = cv2.Stitcher.create(mode=cv2.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(args.pano_conf)

    if args.resize != 1:
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (images[i].shape[1] // args.resize, images[i].shape[0] // args.resize))

    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching successful.")
        print("find points")
        cv2.imwrite(os.path.join(output_base, "stitched_output.jpg"), stitched)
        points, pointed_image = search_locations(images, stitched)

        point_coordinates = []
        with open(os.path.join(output_base, "pointed_output.txt"), 'w') as f:
            for i in range(len(points)):
                if points[i] is not None:
                    point_coordinates.append((points[i], coordinates[i]))
                    f.write(f"{point_coordinates[-1]}\n")
                    cv2.circle(stitched, points[i], 5, 255, -1)
                    cv2.putText(stitched, f"{i}", points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_base, "pointed_output.jpg"), stitched)
        print("Point Coordinates: ")
    else:
        print("Stitching failed. Error code: ", status)


if __name__ == '__main__':
    """
    Stitch images in the input directory sequentially.
    python main_sequential.py <input directory name>
    """
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time spent : {end_time - start_time}")
