import cv2
from pathlib import Path
import time

# Settings for camera resolution
cam = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
cam.set(6, codec)
cam.set(5, 30)
cam.set(3, 1280)
cam.set(4, 720)
cv2.namedWindow("image")

# Create folder in same folder of code if isn't created
folder_of_images = "img_folder"
folder_path = Path(folder_of_images)
if not folder_path.exists():
    folder_path.mkdir()
print(f"Created Folder named: {folder_of_images}")

# Main loop for capturing frames
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("image", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    else:
        time.sleep(5)  # write 1 images for every 5 seconds.
        # SPACE pressed
        img_name = f"opencv_frame_{img_counter}.png"
        save_path = (folder_path / img_name).as_posix()
        cv2.imwrite(save_path, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
