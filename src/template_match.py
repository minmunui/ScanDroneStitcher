import argparse
from typing import Tuple, Any

import cv2
import numpy as np
from cv2 import Mat, UMat
from numpy import ndarray, dtype

from refine import slice_image


# SIFT 객체 생성 (OpenCV 버전에 따라 다를 수 있음)
def search_sub_image(template_image, original_image, template_cutting_ratio=1.0) -> tuple[
    Mat | ndarray | UMat, Mat | ndarray | UMat]:
    """
    템플릿 이미지와 원본 이미지 사이에서 템플릿 이미지의 위치를 찾아 반환
    find location of template image in original image and return it
    :param template_image: image to search
    :param original_image: image to search in
    :param template_cutting_ratio: ratio template image to search
    :return: original image with rectangle around template image, location of template image
    """
    if template_cutting_ratio != 1.0:
        template_image = slice_image(template_image, template_cutting_ratio)

    resized_orig = cv2.resize(original_image, (original_image.shape[1] // 4, original_image.shape[0] // 4))

    sift = cv2.SIFT.create()

    # SIFT로 특징점과 디스크립터 추출
    keypoints1, descriptors1 = sift.detectAndCompute(template_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(resized_orig, None)
    # BFMatcher (Brute Force 매칭 객체 생성)
    bf = cv2.BFMatcher()

    # 디스크립터 간 매칭 수행 (KNN 매칭을 사용하여 가장 좋은 2개의 매칭 점을 찾음)
    try:
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    except Exception as e:
        print(e)
        raise Exception(f"match failed : template_match.py L: 40")

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
        h, w = template_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        print(f"template image location: {dst}")

        # 원본 이미지에 템플릿의 위치를 사각형으로 표시
        resized_orig = cv2.polylines(resized_orig, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        return resized_orig, dst
    else:
        print("Not enough matches are found - {}/10".format(len(good_matches)))
        Exception("Not enough matches are found - {}/10".format(len(good_matches)))


def search_sub_image_from_file(template_image_path, original_image_path, template_cutting_ratio=1.0) -> tuple[
    np.ndarray, tuple[int, int]]:
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    return search_sub_image(template_image, original_image, template_cutting_ratio)


def search_locations(template_images: list[Mat | ndarray | UMat], target_image: Mat | ndarray | UMat,
                     template_cutting_ratio: float = 1.0, minimum_ratio: float = 0.1) -> list[
    tuple[Mat | ndarray | UMat, list[tuple[int, int]]]]:
    """
    Search locations of template images in target image

    :param template_images:
    :param target_image:
    :param template_cutting_ratio:
    :param minimum_ratio:
    :return:
    """

    points = []

    for image in template_images:
        found = False
        not_found = False
        _template_cutting_ratio = template_cutting_ratio
        while not found and not not_found:
            try:
                _, found_coordinate = search_sub_image(image, target_image, _template_cutting_ratio)
                points.append(found_coordinate)
                found = True
            except Exception as e:
                print(e)
                if _template_cutting_ratio <= minimum_ratio:
                    points.append(None)
                    not_found = True
                _template_cutting_ratio -= 0.1

    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template', type=str, help='template image path', nargs='?', default="")
    parser.add_argument('-o', '--original', type=str, help='original image path', nargs='?', default="")
    parser.add_argument('-r', '--ratio', type=float, help='template cutting ratio', nargs='?', default=1.0)
    args = parser.parse_args()

    template = args.template
    original = args.original
    ratio = args.ratio

    if template == "" or original == "":
        print("Please provide template and original image path")
        exit(1)

    done = True

    while done:
        try:
            print(f"Searching template image in original image... \t ratio : {ratio}")
            output_image, found_coordinate = search_sub_image_from_file(template, original, ratio)
            done = False
            print("Done")
            print(f"found coordinate: {found_coordinate}")
            cv2.imwrite("data/output/.jpg", output_image)
        except Exception as e:
            print(e)
            print(f"failing to search template image in original image... \t ratio : {ratio}")
            if ratio <= 0:
                print("Failed to search template image in original image")
                break
            ratio -= 0.1
