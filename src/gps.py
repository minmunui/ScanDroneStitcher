import math
import os
from typing import Tuple, List, Any

import cv2
import exifread
import matplotlib.pyplot as plt
from cv2 import Mat
from numpy import ndarray, dtype
from tqdm import tqdm

DISCARD = -1
NORMAL = 0
ROTATED = 1

RANGE = 5.0


def get_exif_data(img_dir: str = None, img_name: str = None, img_path: str = None):
    """
    Get exif data from image, if img_path is None, img_dir and img_name must be provided
    :param img_dir: image directory
    :param img_name:  image name
    :param img_path:  image path
    :return:
    """
    if img_dir != None and img_name != None:
        img_path = os.path.join(os.getcwd(), img_dir, img_name)
    elif img_path == None:
        raise Exception("img_path is None")

    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)
        return tags


def get_geotagging(exif_data):
    """
    Get geotagging data from exif data of image, insert exif data from get_exif_data()
    :param exif_data: exif data from image
    :return:
    """
    geotagging = {}
    for (key, val) in exif_data.items():
        if key.startswith('GPS'):
            geotagging[key] = val
    if not geotagging:
        raise ValueError("No EXIF geotagging found")
    return geotagging


def get_decimal_from_dms(dms, ref):
    degrees = dms.values[0].num / dms.values[0].den
    minutes = dms.values[1].num / dms.values[1].den / 60.0
    seconds = dms.values[2].num / dms.values[2].den / 3600.0

    decimal = degrees + minutes + seconds
    if ref == 'S' or ref == 'W':
        decimal = -decimal
    return decimal


def get_coordinates(geotags):
    """
    Get latitude and longitude from geotags data gotten from get_geotagging()
    :param geotags:
    :return:
    """
    lat = get_decimal_from_dms(geotags['GPS GPSLatitude'], geotags['GPS GPSLatitudeRef'].printable)
    lon = get_decimal_from_dms(geotags['GPS GPSLongitude'], geotags['GPS GPSLongitudeRef'].printable)
    return lat, lon


def get_altitude(geotags):
    altitude = geotags.get('GPS GPSAltitude', None)
    altitude_ref = geotags.get('GPS GPSAltitudeRef', None)
    if altitude and altitude_ref:
        alt = altitude.values[0].num / altitude.values[0].den
        if altitude_ref.values[0] == 1:
            alt = -alt
        return alt
    raise ValueError("No altitude data found")


def get_gps_from_image(img_dir=None, img_name=None, img_path=None):
    """
    Get latitude, longitude, and altitude from image. If img_path is None, img_dir and img_name must be provided
    :param img_dir:
    :param img_name:
    :param img_path:
    :return:
    """
    exif_data = get_exif_data(img_dir, img_name, img_path)
    geotags = get_geotagging(exif_data)

    if geotags:
        lat, lon = get_coordinates(geotags)
        alt = get_altitude(geotags)
        return lat, lon, alt

    else:
        print()
        raise ValueError("No GPS data found")


def plot_coordinates(coordinates):
    lats, lons, alts = zip(*coordinates)
    plt.figure(figsize=(10, 6))
    plt.scatter(lons, lats, c='blue', marker='o')

    for i, (lat, lon, alt) in enumerate(coordinates):
        plt.text(lon, lat, f'{i}:{alt}', fontsize=12, ha='right')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Coordinates Plot')
    plt.grid(True)
    plt.show()


def get_direction(coord1: tuple[float, float], coord2: tuple[float, float]) -> tuple[float, float]:
    """
    Get direction from coord1 to coord2
    :param coord1:
    :param coord2:
    :return:
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    return dlat, dlon


def direction_to_angle(direction: tuple[float, float]) -> float:
    """
    Convert direction to angle
    :param direction:
    :return:
    """
    dlat, dlon = direction
    angle = (180 / math.pi) * (math.pi + math.pi / 2 - math.atan2(dlat, dlon))
    if angle < 0:
        return 360.0 + angle
    if angle > 360:
        return angle - 360.0
    return angle


def get_angel_between_coordinates(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """
    Get angle between two coordinates
    :param coord1:
    :param coord2:
    :return:
    """
    direction = get_direction(coord1, coord2)
    return direction_to_angle(direction)


def get_angles(coordinates: list[tuple[float, float]]) -> list[float]:
    """
    Get angles between coordinates. coordinates must be list of (latitude, longitude)
    :param coordinates:
    :return:
    """
    angles = []
    for i in range(1, len(coordinates)):
        angle = get_angel_between_coordinates(coordinates[i - 1], coordinates[i])
        if i == 1:
            angles.append(angle)
        angles.append(angle)

    return angles


def to_360_angle(angle: float) -> float:
    """
    Convert angle from 0 to 360 degree
    :param angle:
    :return:
    """
    if angle < 0:
        return 360.0 + angle
    if angle > 360:
        return angle - 360.0
    return angle


def determine_rotation(standard: float, angle: float, threshold_range: float = RANGE) -> int:
    """
    Determine rotation of angle from standard angle. If angle is in threshold range, return NORMAL(0), REVERSED(1) if reversed, DISCARD(-1) if discarded
    :param standard:
    :param angle:
    :param threshold_range:
    :return:
    """
    if standard - threshold_range <= angle <= standard + threshold_range:
        return NORMAL
    elif to_360_angle(standard - threshold_range + 180) <= angle <= to_360_angle(standard + threshold_range + 180):
        return ROTATED
    else:
        return DISCARD


def determine_rotation_angles(angles: list[float]) -> list[float]:
    """
    Determine rotation of angles from standard angle. If angle is in threshold range, return NORMAL(0), REVERSED(1) if reversed, DISCARD(-1) if discarded
    :param angles:
    :return:
    """
    standard = angles[0]
    results = []
    for angle in angles:
        results.append(determine_rotation(standard, angle))
    return results


def getClusteredIndices(points, n_clusters, max_iterations=100):
    """
    Get clustered indices from points.

    :param max_iterations : int, maximum number of iterations
    :param points: list of (x, y) points
    :param n_clusters: int, number of clusters
    :return: list of indices
    """
    import numpy as np

    # points를 NumPy 배열로 변환 (이미 NumPy 배열이면 변환하지 않음)
    points_array = np.array(points)

    # 데이터의 개수
    n_points = points_array.shape[0]

    # 초기 중심점 선택 (데이터 중에서 무작위로 선택)
    np.random.seed(42)
    initial_centroids_indices = np.random.choice(n_points, n_clusters, replace=False)
    centroids = points_array[initial_centroids_indices]
    labels = np.zeros(n_points)

    for iteration in range(max_iterations):
        # 각 점에 대해 가장 가까운 중심점의 인덱스를 찾음
        distances = np.linalg.norm(points_array[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 새로운 중심점 계산
        new_centroids = np.array(
            [points_array[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in
             range(n_clusters)])

        # 중심점의 변화량 확인
        if np.allclose(centroids, new_centroids):
            break  # 수렴하면 종료

        centroids = new_centroids

    # 각 클러스터에 속하는 점들의 인덱스를 저장할 리스트 초기화
    cluster_indices = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    return cluster_indices


def plotClusteredPoints(points, clustered_indices):
    """
    Plot clustered points.

    :param points: list of (x, y) points
    :param clustered_indices: list of indices
    """
    import matplotlib.pyplot as plt

    for i in range(len(clustered_indices)):
        cluster = points[clustered_indices[i]]
        plt.scatter(cluster[:, 0], cluster[:, 1])

    plt.show()


def align_images(dir_path: str = None, image_paths: list[str] = None) -> tuple[
    list[Mat | ndarray], list[str] | None, list[tuple]]:
    """
    Align images from directory or image paths. rotate images if needed, discard images if needed, and return and save aligned images
    :param dir_path:
    :param image_paths:
    :return: aligned images, aligned image paths
    """

    if dir_path is not None:
        image_names = os.listdir(dir_path)
        image_paths = [os.path.join(dir_path, image_name) for image_name in image_names]
    elif image_paths is None:
        raise Exception("dir_path and image_paths are None")

    coordinates = []
    images = []

    for image_path in tqdm(image_paths, desc="reading image coordinates"):
        coordinates.append(get_gps_from_image(img_path=image_path)[:2])

    angles = get_angles(coordinates)
    rotate = determine_rotation_angles(angles)

    discard_index = []

    for i in tqdm(range(len(image_paths)), desc="refine images"):
        if rotate[i] == NORMAL:
            image = cv2.imread(image_paths[i])
            images.append(image)
        elif rotate[i] == ROTATED:
            image = cv2.imread(image_paths[i])
            image = cv2.rotate(image, cv2.ROTATE_180)
            images.append(image)
        else:
            discard_index.append(i)

    for i in list(reversed(discard_index)):
        del image_paths[i]
        del coordinates[i]
        del angles[i]
        del rotate[i]

    return images, image_paths, coordinates
