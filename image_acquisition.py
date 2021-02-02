import cv2
import os
import time

cam = cv2.VideoCapture(0)

cv2.namedWindow("image")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("image", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    else:
        time.sleep(5) #write 1 images for every 5 seconds. 
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        path = 'C:/Users/Shawn/Pictures/image_acquisition'
        cv2.imwrite(os.path.join(path, img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()