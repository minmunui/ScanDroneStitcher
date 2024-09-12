import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread('data/input/origin/deokgok_all/DJI_20240404115647_0001_W.JPG')

# 이미지의 높이와 너비를 얻기
(h, w) = image.shape[:2]

# 회전할 중심점 (이미지의 중앙)
center = (w // 2, h // 2)

# 회전 매트릭스를 생성 (30도, 확대/축소는 1로 유지)
M = cv2.getRotationMatrix2D(center, 30, 1.0)

# 이미지의 새로운 경계 크기를 계산
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])

# 새로운 경계 크기 계산
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# 회전 후 이미지를 전체가 보이도록 하기 위해 이동
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

# 이미지 회전
rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

mask = np.zeros((h, w), dtype="uint8")
cv2.rectangle(mask, (0, 0), (w, h), 255, -1)
rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))

# 회전된 이미지 저장 또는 출력
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 회전된 이미지 저장
cv2.imwrite('rotated_image.jpg', rotated_image)
cv2.imwrite('rotated_mask.jpg', rotated_mask)