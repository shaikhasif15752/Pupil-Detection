"""Pupil detection using openCV"""

import argparse
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import dlib


def eye_aspect_ratio(eye):
    """compute the euclidean distances between the two sets of
    vertical eye landmarks (x, y)-coordinates"""
    dist_15 = dist.euclidean(eye[1], eye[5])
    dist_24 = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    dist03 = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (dist_15 + dist_24) / (2.0 * dist03)

    # return the eye aspect ratio
    return ear


AP = argparse.ArgumentParser()
AP.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
AP.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
AP.add_argument("-v", "--video", help="path to the (optional) video file")
ARGS = vars(AP.parse_args())

print "[INFO] loading facial landmark predictor..."

DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(ARGS["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(LSTART, LEND) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(RSTART, REND) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

if not ARGS.get("video", False):
    VS = VideoStream(src=ARGS["webcam"]).start()

# otherwise, grab a reference to the video file
else:
    VS = cv2.VideoCapture(ARGS["video"])

PADDING_X = 5
PADDING_Y = 3
EYE_AR_THRESH = 0.3

while True:
    FRAME = VS.read()
    # handle the frame from VideoCapture or VideoStream
    FRAME = FRAME[1] if ARGS.get("video", False) else FRAME

    if FRAME is None:
        break

    BLURRED = cv2.GaussianBlur(FRAME, (11, 11), 0)
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)

    # detect faces in the gray scale frame
    FACES = DETECTOR(GRAY, 0)

    for face in FACES:
        facial_landmarks = PREDICTOR(GRAY, face)
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)

        leftEye = facial_landmarks[LSTART:LEND]
        rightEye = facial_landmarks[RSTART:REND]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear_avg = (leftEAR + rightEAR) / 2.0

        # Location of left bounding box
        xLeft, yLeft = leftEye[0][0], leftEye[2][1]
        widthL, heightL = leftEye[3][0], leftEye[4][1]

        # Location of Right bounding box
        xRight, yRight = rightEye[0][0], rightEye[2][1]
        widthR, heightR = rightEye[3][0], rightEye[4][1]

        # draw rectangle around ayes
        cv2.rectangle(FRAME, (xLeft + PADDING_X, yLeft + PADDING_Y), \
                      (widthL - PADDING_X, heightL - PADDING_Y), (0, 255, 0), 1)
        cv2.rectangle(FRAME, (xRight + PADDING_X, yRight + PADDING_Y), \
                      (widthR - PADDING_X, heightR - PADDING_Y), (0, 255, 0), 1)
        # Extracting region of left eye for further process
        leftPart = GRAY[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]
        leftPartColor = FRAME[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]

        # Extracting region of right eye for further process
        rightPart = GRAY[yRight + PADDING_Y:heightR - PADDING_Y, xRight + \
                                                                 PADDING_X:widthR - PADDING_X]
        rightPartColor = FRAME[yRight + PADDING_Y:heightR - PADDING_Y, \
                         xRight + PADDING_X:widthR - PADDING_X]

        # Verify that eyes are not closed
        if ear_avg >= EYE_AR_THRESH:
            # finding location of darker pixel inside eye region
            (_, _, minLocL, _) = cv2.minMaxLoc(leftPart)
            cv2.circle(leftPartColor, minLocL, 5, (0, 0, 255), 2)

            (_, _, minLocR, _) = cv2.minMaxLoc(rightPart)
            cv2.circle(rightPartColor, minLocR, 5, (0, 0, 255), 2)

    cv2.imshow("Frame", FRAME)

    KEY = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if KEY == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
