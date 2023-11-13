import numpy as np
import math
import cv2
from numpy import dot, sqrt
from PIL import Image

def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                     ((b[1] - a[1]) * (b[1] - a[1])))

def align_face(raw_face, left_eye, right_eye):
    right_eye_x = right_eye[0]
    right_eye_y = right_eye[1]

    left_eye_x = left_eye[0]
    left_eye_y = left_eye[1]

    # finding rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate image direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    a = trignometry_for_distance(left_eye, point_3rd)
    b = trignometry_for_distance(right_eye, point_3rd)
    c = trignometry_for_distance(right_eye, left_eye)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = (np.arccos(cos_a) * 180) / math.pi

    if direction == -1:
        angle = 90 - angle

    # Rotate the face image
    
    rows, cols = raw_face.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle * direction, 1)
    rotated_image = cv2.warpAffine(raw_face, rotation_matrix, (cols, rows))

    # Fill the background with white color
    mask = np.all(rotated_image == [0, 0, 0], axis=-1)
    rotated_image[mask] = [255, 255, 255]

    # Save the resulting image
    # cv2.imwrite("rotated_filled_image.jpg", rotated_image)
    return rotated_image

def draw_box(image, boxes, color=(125, 255, 125), thickness = 10):
    """Draw square boxes on image"""
    edge_pixel = 20
    for box in boxes:
        # cv2.rectangle(image,
        #                 (int(box[0]), int(box[1])),
        #                 (int(box[2]), int(box[3])), color, 3)
        # Top-left
        cv2.line(image, (int(box[0]), int(box[1])), (int(box[0] + edge_pixel), int(box[1])), color, thickness)
        cv2.line(image, (int(box[0]), int(box[1])), (int(box[0]), int(box[1] + edge_pixel)), color, thickness)
        # Top-right
        cv2.line(image, (int(box[2]), int(box[1])), (int(box[2] - edge_pixel), int(box[1])), color, thickness)
        cv2.line(image, (int(box[2]), int(box[1])), (int(box[2]), int(box[1] + edge_pixel)), color, thickness)
        # Bottom-right
        cv2.line(image, (int(box[2]), int(box[3])), (int(box[2] - edge_pixel), int(box[3])), color, thickness)
        cv2.line(image, (int(box[2]), int(box[3])), (int(box[2]), int(box[3] - edge_pixel)), color, thickness)
        # # Bottom-left
        cv2.line(image, (int(box[0]), int(box[3])), (int(box[0] + edge_pixel), int(box[3])), color, thickness)
        cv2.line(image, (int(box[0]), int(box[3])), (int(box[0]), int(box[3] - edge_pixel)), color, thickness)

def draw_landmark(image, landmarks, color=(125, 255, 125)):
    """Draw landmarks on image"""
    # temp_img = image.copy()
    for index, idI in enumerate(landmarks):
        if index == 34 or index == 38 or index == 92 or index == 88:
            cv2.circle(image, (int(landmarks[index][0]), int(landmarks[index][1])), 2, (0, 0, 255), -1)  
        else:
            cv2.circle(image, (int(landmarks[index][0]), int(landmarks[index][1])), 1, color, -1)  
        # if (index % 2) == 0: 
        #     cv2.putText(temp_img, '{0}'.format(index), (int(landmarks[index][0]), int(landmarks[index][1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        # else:
        #     cv2.putText(temp_img, '{0}'.format(index), (int(landmarks[index][0]), int(landmarks[index][1])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("{}.jpg".format(index), temp_img)