import cv2
import pyvirtualcam

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
deblur_active = False
video_capture = cv2.VideoCapture(0)
# with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if not deblur_active:
                frame[y:y + h, x:x + w] = 0
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)




    cv2.imshow('Video', frame)
    if cv2.waitKey(1)  == ord('q'):
        break
    elif cv2.waitKey(1) == ord('d'):
        deblur_active = True
    elif cv2.waitKey(1) == ord('e'):
        deblur_active = False
    elif cv2.waitKey(1) == ord('c'):
        cv2.imwrite('frame_copy.jpg',frame)
cv2.destroyAllWindows()
video_capture.release()

