import math
import os

import exifread
import matplotlib.pyplot as plt

DISCARD = -1
NORMAL = 0
REVERSED = 1

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
        return REVERSED
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
