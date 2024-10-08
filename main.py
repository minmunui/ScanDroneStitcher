import argparse
import os
import time

import cv2

from src.img_io import make_output_name
from src.metadata.gps import align_images, plotClusteredPoints, getClusteredIndices
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
    parser.add_argument('-c', '--cluster', type=int, help='num of cluster', nargs='?', default=1)
    args = parser.parse_args()

    input_path = args.input  # 입력 디렉토리 이름

    images, image_names, coordinates = align_images(dir_path=input_path)

    print(f"len(images) : {len(images)}")
    print(f"selected images : {len(image_names)}\n{image_names}")

    clustered_indices = getClusteredIndices(coordinates, n_clusters=args.cluster)

    print(f"clustered_indices : {clustered_indices}")

    output_dir = make_output_name()

    if args.output != "result":
        output_dir = args.output

    output_base = os.path.join(os.getcwd(), "data", "output", output_dir)
    os.makedirs(output_base, exist_ok=True)
    plotClusteredPoints(coordinates, clustered_indices, output_path=os.path.join(output_base, "clustered.png"))

    for idx, clustered_index in enumerate(clustered_indices):
        print(f"clustered_index : {clustered_index}")
        clustered_images = [images[i] for i in clustered_index]
        clustered_coordinates = [coordinates[i] for i in clustered_index]
        clustered_image_names = [image_names[i] for i in clustered_index]

        cluster_output_base = os.path.join(output_base, f"cluster_{idx}")

        os.makedirs(cluster_output_base, exist_ok=True)

        for _idx, image_name in enumerate(clustered_image_names):
            print(f"save image : {image_name}")
            cv2.imwrite(os.path.join(cluster_output_base, f"{_idx}_{image_name.split('\\')[-1]}"), clustered_images[_idx])

        print(f"result will be saved on {cluster_output_base}")
        stitcher = cv2.Stitcher.create(mode=cv2.STITCHER_SCANS)
        stitcher.setPanoConfidenceThresh(args.pano_conf)

        status, stitched = stitcher.stitch(clustered_images)

        if status == cv2.Stitcher_OK:
            print("Stitching successful.")
            print("find points")
            cv2.imwrite(os.path.join(cluster_output_base, f"{output_dir}.jpg"), stitched)
            points, pointed_image = search_locations(clustered_images, stitched)

            point_coordinates = []
            with open(os.path.join(cluster_output_base, f"points.txt"), 'w') as f:
                for i in range(len(points)):
                    if points[i] is not None:
                        point_coordinates.append((points[i], clustered_coordinates[i]))
                        f.write(f"{point_coordinates[-1]}\n")
                        cv2.circle(stitched, points[i], 5, 255, -1)
                        cv2.putText(stitched, f"{i}", points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(cluster_output_base, f"pointed.jpg"), stitched)
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
