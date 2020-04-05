import numpy as np
import dlib
import cv2
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
output = open("output.txt","w")

for (i, rect) in enumerate(rects):
    output.write("Face {}:\n".format(i+1))    
    (x, y, w, h) = (rect.left(), rect.top(), rect.right()-rect.left(), rect.bottom()-rect.top())
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Face:{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    landmarks = predictor(gray, rect)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    landmarks = coords
	
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        output.write("{},{}\n".format(x,y))
    
    jaw = landmarks[3:14]
    z = np.polyfit(jaw[:,0], jaw[:,1], 6)
    x = np.linspace(jaw[:,0][0], jaw[:,0][10], 50)
    y = np.polyval(z, x)
    jawline = (np.asarray([x, y]).T).astype(np.int32)
    cv2.polylines(image, [jawline], False ,(0,255,255) , 2, cv2.LINE_AA)
    output.write("\n")

screen_res = (1280,720)
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)

cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', window_width, window_height)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
output.close()