import argparse
import os
import time

import cv2
from tqdm import tqdm

from src.metadata.gps import get_gps_from_image, get_angles, determine_rotation_angles
from src.img_io import make_output_name

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
    args = parser.parse_args()

    input_dir = args.input  # 입력 디렉토리 이름

    input_path = os.path.join(os.getcwd(), "data", "input", input_dir)
    image_names = os.listdir(input_path)

    image_paths = [os.path.join(input_path, image_name) for image_name in image_names]

    coordinates = []
    images = []

    for image_path in tqdm(image_paths, desc="reading images"):
        coordinates.append(get_gps_from_image(img_path=image_path)[:2])
        images.append(cv2.imread(image_path))

    angles = get_angles(coordinates)

    rotate = determine_rotation_angles(angles)

    discard_index = []

    for i in tqdm(range(len(images)), desc="refine images"):
        if rotate[i] == ORIGINAL:
            continue
        elif rotate[i] == ROTATED:
            images[i] = cv2.rotate(images[i], cv2.ROTATE_180)
        else:
            discard_index.append(i)

    for i in list(reversed(discard_index)):
        del images[i]
        del image_names[i]
        del angles[i]
        del rotate[i]
        del coordinates[i]

    output_dir = make_output_name()

    if args.output != "result":
        output_dir = args.output
    output_base = os.path.join(os.getcwd(), "data", "output", output_dir)
    os.makedirs(output_base, exist_ok=True)
    for idx, image_name in enumerate(image_names):
        cv2.imwrite(os.path.join(os.getcwd(), "data", "output", output_dir, f"{idx}_{image_name}"), images[idx])

    print(f"result will be saved on {output_base}")
    print(f"selected images : {image_names}")
    stitcher = cv2.Stitcher.create(mode=cv2.STITCHER_SCANS)
    pano_conf_thresh = 0.9
    while pano_conf_thresh < 1.5:
        iter_start_time = time.time()
        stitcher.setPanoConfidenceThresh(pano_conf_thresh)
        try:
            status, stitched = stitcher.stitch(images)
            if status == cv2.Stitcher_OK:
                print("Stitching successful.")
                cv2.imwrite(os.path.join(output_base, f"pano_conf_thresh_{pano_conf_thresh}.jpg"), stitched)
            else:
                print(f"Stitching failed. Error code: {status}. Trying with higher confidence threshold.")
        except cv2.error as e:
            print(f"OpenCV Error: {e}")
        iter_end_time = time.time()
        print(f"Time spent for iteration - {pano_conf_thresh}: {iter_end_time - iter_start_time}")
        pano_conf_thresh = round(pano_conf_thresh + 0.05, 2)


if __name__ == '__main__':
    """
    Stitch images in the input directory sequentially.
    python main_sequential.py <input directory name>
    """
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time spent : {end_time - start_time}")
