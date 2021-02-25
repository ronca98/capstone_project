import cv2
from pathlib import Path

vid_file = "F_feb21_block_105_30.mp4"
vid_file_name = Path(vid_file).stem
vid_cap = cv2.VideoCapture(vid_file)

success, image = vid_cap.read()
count = 0

while success:
    image = image[0:224, 128:352]

    # save frame as PNG file
    cv2.imwrite(fr"validation_images\normal\{vid_file_name}_{count}.png",
                image)

    success, image = vid_cap.read()
    print("Read a new frame: ", success)
    count += 1
