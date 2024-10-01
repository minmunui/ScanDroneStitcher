import argparse
import os
import time

import cv2

from src.gps import align_images
from src.img_io import make_output_name
from template_match import search_locations


ROT = {
    '0': "NO ROTATION",
    '-1': "DISCARD",
    '1': "ROTATE 180",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input directory name', nargs='?', default="input")
    parser.add_argument('-o', '--output', type=str, help='output file name', nargs='?', default="result")
    parser.add_argument('-p', '--pano_conf', type=float, help='panorama confidence', nargs='?', default=1.0)
    args = parser.parse_args()

    input_path = args.input  # 입력 디렉토리 이름

    images, image_names, coordinates = align_images(dir_path=input_path)

    print(f"len(images) : {len(images)}")
    print(f"selected images : {len(image_names)}\n{image_names}")
    output_dir = make_output_name()

    if args.output != "result":
        output_dir = args.output
    output_base = os.path.join(os.getcwd(), "data", "output", output_dir)
    os.makedirs(output_base, exist_ok=True)

    for idx, image_name in enumerate(image_names):
        print(f"{idx} \t {image_name}")
        cv2.imwrite(os.path.join(os.getcwd(), "data", "output", output_dir, f"{idx}_{image_name}"), images[idx])

    print(f"result will be saved on {output_base}")
    stitcher = cv2.Stitcher.create(mode=cv2.STITCHER_SCANS)
    stitcher.setPanoConfidenceThresh(args.pano_conf)

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
