import argparse
import os
import time

import cv2
from tqdm import tqdm

from src.gps import get_gps_from_image, get_angles, determine_rotation_angles
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
    parser.add_argument('-l', '--log', type=str, help='is remain log at console', nargs='?', default="log")
    parser.add_argument('-c', '--resize', type=int, help='resize images', nargs='?', default=1)
    parser.add_argument('-p', '--pano_conf', type=float, help='panorama confidence', nargs='?', default=1.0)
    args = parser.parse_args()

    input_dir = args.input  # 입력 디렉토리 이름
    is_log = False
    if args.log == "log":
        is_log = True
    input_path = os.path.join(os.getcwd(), "data", "input", input_dir)
    image_names = os.listdir(input_path)

    image_paths = [os.path.join(input_path, image_name) for image_name in image_names]

    images = [cv2.imread(image_path) for image_path in image_paths]

    output_dir = make_output_name()

    if args.output != "result":
        output_dir = args.output
    output_base = os.path.join(os.getcwd(), "data", "output", output_dir)
    os.makedirs(output_base, exist_ok=True)
    print(f"result will be saved on {output_base}")
    print(f"selected images : {image_names}")
    stitcher = cv2.Stitcher.create(mode=cv2.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(args.pano_conf)

    print(f"resize images : {args.resize}")
    if args.resize != 1:
        for i in range(len(images)):
            images[i] = cv2.resize(images[i], (images[i].shape[1] // args.resize, images[i].shape[0] // args.resize))

    status, stitched = stitcher.stitch(images, masks=None)

    if status == cv2.Stitcher_OK:
        print("Stitching successful.")
        cv2.imwrite(os.path.join(output_base, f"{args.output}.tiff"), stitched)
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
