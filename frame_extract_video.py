import cv2
from pathlib import Path

vid_file = "N_feb21_115_cube_30.mp4"
vid_file_name = Path(vid_file).stem
vid_cap = cv2.VideoCapture(vid_file)

success, image = vid_cap.read()
count = 0

while success:
    image = image[100:150, 128:352]

    # save frame as PNG file
    cv2.imwrite(fr"images_to_try\{vid_file_name}_{count}.png",
                image)

    success, image = vid_cap.read()
    print("Read a new frame: ", success)
    count += 1
