# Pupil detection using openCV
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import dlib

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

if not args.get("video", False):
    vs = VideoStream(src=args["webcam"]).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])


PADDING_X = 5
PADDING_Y = 3
EYE_AR_THRESH = 0.3

while True:
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the gray scale frame
    faces = detector(gray, 0)

    for face in faces:
        facial_landmarks = predictor(gray, face)
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)

        leftEye = facial_landmarks[lStart:lEnd]
        rightEye = facial_landmarks[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Location of left bounding box
        xLeft , yLeft = leftEye[0][0], leftEye[2][1]
        widthL, heightL = leftEye[3][0], leftEye[4][1]

        # Location of Right bounding box
        xRight, yRight= rightEye[0][0], rightEye[2][1]
        widthR, heightR = rightEye[3][0], rightEye[4][1]

        # draw rectangle around ayes
        #cv2.rectangle(frame, (xLeft+PADDING_X, yLeft+PADDING_Y), (widthL-PADDING_X, heightL-PADDING_Y), (0, 255, 0), 1)
        #cv2.rectangle(frame, (xRight+PADDING_X, yRight+PADDING_Y), (widthR-PADDING_X, heightR-PADDING_Y), (0, 255, 0), 1)

        # Extracting region of left eye for further process
        leftPart = gray[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]
        leftPartColor = frame[yLeft + PADDING_Y:heightL, xLeft + PADDING_X:widthL - PADDING_X]

        # Extracting region of right eye for further process
        rightPart = gray[yRight + PADDING_Y:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]
        rightPartColor = frame[yRight + PADDING_Y:heightR - PADDING_Y, xRight + PADDING_X:widthR - PADDING_X]

        # Verify that eyes are not closed
        if ear >= EYE_AR_THRESH:
            # finding location of darker pixel inside eye region
            (_, _, minLocL, _) = cv2.minMaxLoc(leftPart)
            cv2.circle(leftPartColor, minLocL, 5, (0, 0, 255), 2)

            (_, _, minLocR, _) = cv2.minMaxLoc(rightPart)
            cv2.circle(rightPartColor, minLocR, 5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
