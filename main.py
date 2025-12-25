import cv2
import dlib
import time
from scipy.spatial import distance

EYE_CLOSED_THRESHOLD = 0.25
CLOSED_DURATION_THRESHOLD = 2.0
frame_rate = 30

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)
closed_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        left_eye_points = [landmarks_points[i] for i in LEFT_EYE]
        right_eye_points = [landmarks_points[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2

        for (x, y) in left_eye_points + right_eye_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if avg_ear < EYE_CLOSED_THRESHOLD:
            if closed_start_time is None:
                closed_start_time = time.time()
            elif time.time() - closed_start_time > CLOSED_DURATION_THRESHOLD:
                cv2.putText(frame, "ALERT! Eyes closed too long", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                print("ALERT! Eyes closed too long")
        else:
            closed_start_time = None

    cv2.imshow("OcuTrack - Dlib", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
