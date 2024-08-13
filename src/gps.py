import math
import os

import exifread
import matplotlib.pyplot as plt

DISCARD = -1
NORMAL = 0
REVERSED = 1

RANGE = 5.0


def get_exif_data(img_dir: str = None, img_name: str = None, img_path: str = None):
    if img_dir != None and img_name != None:
        img_path = os.path.join(os.getcwd(), img_dir, img_name)
    elif img_path == None:
        raise Exception("img_path is None")

    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)
        return tags


def get_geotagging(exif_data):
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
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    return dlat, dlon


def direction_to_angle(direction: tuple[float, float]) -> float:
    dlat, dlon = direction
    angle = (180 / math.pi) * (math.pi + math.pi / 2 - math.atan2(dlat, dlon))
    if angle < 0:
        return 360.0 + angle
    if angle > 360:
        return angle - 360.0
    return angle


def get_angel_between_coordinates(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    direction = get_direction(coord1, coord2)
    return direction_to_angle(direction)


def get_angles(coordinates: list[tuple[float, float]]) -> list[float]:
    angles = []
    for i in range(1, len(coordinates)):
        angle = get_angel_between_coordinates(coordinates[i - 1], coordinates[i])
        if i == 1:
            angles.append(angle)
        angles.append(angle)

    return angles


def to_360_angle(angle: float) -> float:
    if angle < 0:
        return 360.0 + angle
    if angle > 360:
        return angle - 360.0
    return angle


def determine_rotation(standard: float, angle: float, threshold_range: float = RANGE) -> int:
    if standard - threshold_range <= angle <= standard + threshold_range:
        return NORMAL
    elif to_360_angle(standard - threshold_range + 180) <= angle <= to_360_angle(standard + threshold_range + 180):
        return REVERSED
    else:
        return DISCARD


def determine_rotation_angles(angles: list[float]) -> list[float]:
    standard = angles[0]
    results = []
    for angle in angles:
        results.append(determine_rotation(standard, angle))
    return results
