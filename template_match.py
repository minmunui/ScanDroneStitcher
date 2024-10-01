"""
python .\template_match.py --template .\data\input\find_pints\deokgok_230720_1_templates\ --original .\data\input\find_pints\deokgok_230720_1_output\pano_conf_thresh_1.15.jpg
"""

import argparse
import os
from typing import Tuple, Any

import cv2
import numpy as np
from cv2 import Mat, UMat
from numpy import ndarray, dtype

from src.img_io import get_file_name
from src.refine import slice_image


# SIFT 객체 생성 (OpenCV 버전에 따라 다를 수 있음)
def search_sub_image(template_image, origin_image, template_crop_ratio: float = 1.0,
                     original_image_resize_ratio: int = 4) \
        -> tuple[Mat | ndarray | UMat, Mat | ndarray | UMat, bool]:
    """
    템플릿 이미지와 원본 이미지 사이에서 템플릿 이미지의 위치를 찾아 반환,
    find location of template image in original image and return it
    :param template_image: image to search
    :param origin_image: image to search in
    :param original_image_resize_ratio: ratio original image to resize, default 4 (1/4) because it is difficult to find proper number of keypoints in large image
    :param template_crop_ratio: ratio template image to search
    :return: original image with rectangle around template image, location of template image
    """
    if template_crop_ratio != 1.0:
        template_image = slice_image(template_image, template_crop_ratio)

    if original_image_resize_ratio != 1:
        origin_image = cv2.resize(origin_image, (
            origin_image.shape[1] // original_image_resize_ratio, origin_image.shape[0] // original_image_resize_ratio))

    sift = cv2.SIFT.create()

    # SIFT로 특징점과 디스크립터 추출
    keypoints1, descriptors1 = sift.detectAndCompute(template_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(origin_image, None)
    # BFMatcher (Brute Force 매칭 객체 생성)
    bf = cv2.BFMatcher()

    # 디스크립터 간 매칭 수행 (KNN 매칭을 사용하여 가장 좋은 2개의 매칭 점을 찾음)
    try:
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    except Exception as e:
        print(e)
        return origin_image, None, False

    # Lowe's ratio test 적용 (좋은 매칭 점 필터링)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 최소한 10개 이상의 좋은 매칭이 있어야 템플릿 매칭을 진행
    if len(good_matches) > 10:
        # 좋은 매칭 점들로부터 특징점 좌표 추출
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # RANSAC을 사용해 매칭 점들 사이의 변환 행렬 계산
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 템플릿 이미지의 크기 정보를 바탕으로 원본 이미지에서의 위치 찾기
        print(f"template image shape : {template_image.shape}")
        h, w, _ = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        dst_center = list(map(int, dst.mean(axis=0).flatten()))

        origin_dst_center = tuple(map(lambda x: x * original_image_resize_ratio, dst_center))
        return origin_image, origin_dst_center, True
    else:
        return origin_image, None, False


def search_sub_image_from_file(template_image_path, original_image_path, template_crop_ratio=0.5) -> tuple[
    np.ndarray, tuple[int, int]]:
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    output_image, found_coordinate, found = search_sub_image(template_image, original_image, template_crop_ratio)

    return output_image, found_coordinate


def search_locations(template_images: list[Mat | ndarray | UMat],
                     target_image: Mat | ndarray | UMat,
                     template_crop_ratio: float = 1.0,
                     resize_ratio: int = 4,
                     ) \
        -> Tuple[list[Tuple[int, int] | None], Mat | ndarray | UMat]:
    """
    Search locations of template images in target image

    :param resize_ratio: ratio of template image to resize (default 4) ex) 4 -> 1/4
    :param template_images: list of template images
    :param target_image: target image
    :param template_crop_ratio: ratio of template image to search
    :param original_image_resize_ratio: ratio of original image to resize
    :return:
    """

    if resize_ratio < 1:
        Exception("template resize ratio should be greater than 1")

    if resize_ratio != 1:
        for _i in range(len(template_images)):
            template_images[_i] = cv2.resize(template_images[_i], (
                template_images[_i].shape[1] // resize_ratio,
                template_images[_i].shape[0] // resize_ratio))
        target_image = cv2.resize(target_image, (
            target_image.shape[1] // resize_ratio, target_image.shape[0] // resize_ratio))

    _points = []

    for image in template_images:
        print(f"searching in target image")

        _template_cutting_ratio = template_crop_ratio
        _, _found_coordinate, found = search_sub_image(image, target_image,
                                                       template_crop_ratio=1,
                                                       original_image_resize_ratio=1
                                                       )
        if found:
            _points.append(tuple(map(lambda x: x * resize_ratio, _found_coordinate)))

    for point in _points:
        if point is not None:
            cv2.circle(target_image, point, 5, 255, -1)

    cv2.imwrite("pointed.jpg", target_image)
    return _points, target_image


# 입력 : 이미지가 든 파일들, 타겟 이미지
# 출력 : 타겟 이미지에서 유추 가능한 gps 좌표와 그림 상의 좌표 (gps 좌표, 그림 상의 좌표)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template', type=str, help='template image path', nargs='?', default="")
    parser.add_argument('-o', '--original', type=str, help='original image path', nargs='?', default="")
    parser.add_argument('-r', '--crop_ratio', type=float, help='template cutting ratio', nargs='?', default=1.0)
    parser.add_argument('-p', '--resize_ratio', type=int, help='resize ratio', nargs='?', default=4)
    args = parser.parse_args()

    template = args.template
    original = args.original
    template_crop_ratio = args.crop_ratio
    resize_ratio = args.resize_ratio

    if template == "" or original == "":
        print("Please provide template and original image path")
        exit(1)

    done = True

    template_paths = os.listdir(template)
    template_paths = [os.path.join(template, path) for path in template_paths]
    print(f"template paths : {template_paths}")
    templates = []

    for path in template_paths:
        templates.append(cv2.imread(path))

    original_image = cv2.imread(original)

    print(f"length of templates : {len(templates)}")

    points, _ = search_locations(templates, original_image, template_crop_ratio=template_crop_ratio, resize_ratio=resize_ratio)

    print(f"points : {points}")
    print(f"point length : {len(points)}")

    output_base = os.path.join(os.getcwd(), "data", "output", "pointed", template.split("\\")[-1])
    print(f"output will be saved on {output_base}")
    os.makedirs(output_base, exist_ok=True)
    with open(os.path.join(output_base, "pointed_output.txt"), 'w') as f:
        for i in range(len(points)):
            if points[i] is not None:
                print(f"writting {i}")
                f.write(f"{points[i]}\t{template_paths[i]}\n")
                cv2.circle(original_image, points[i], 5, 255, -1)
                cv2.putText(original_image, f"{i}", points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_base, "pointed_output.jpg"), original_image)
