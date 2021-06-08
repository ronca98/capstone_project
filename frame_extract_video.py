import cv2
from pathlib import Path
import numpy as np


def video_to_frames(vid_file, output_folder):
    vid_file_name = Path(vid_file).stem
    vid_cap = cv2.VideoCapture(vid_file)
    censor = True

    success, image = vid_cap.read()
    count = 0

    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[35:259, 128:352]

        if censor:
            # image = cv2.rectangle(image, (0, 0), (224, 45), (0, 0, 0), -1)
            # diagonal boundry lines
            image = cv2.line(image, (165, 45 - 45), (115, 95 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (45, 45 - 45), (95, 95 - 45), (0, 0, 0), 10)
            # filling between diagonal lines
            image = cv2.line(image, (45, 45 - 45), (165, 45 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (55, 55 - 45), (155, 55 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (65, 65 - 45), (145, 65 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (75, 75 - 45), (135, 75 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (85, 85 - 45), (125, 85 - 45), (0, 0, 0), 10)
            image = cv2.line(image, (95, 95 - 45), (115, 95 - 45), (0, 0, 0), 10)

        cv2.imwrite(f"{output_folder}/{vid_file_name}_{count}.png",
                    image)

        success, image = vid_cap.read()
        print("Read a new frame: ", success)
        count += 1


def main():
    video_files_list = ["video_files/random_shape_4_15-_fill_normal.mp4",
                        "video_files/random_shape_4_15-_fill_over.mp4",
                        "video_files/random_shape_4_15-_fill_under.mp4"]

    output_folders_list = ["images_to_try",
                           "images_to_try",
                           "images_to_try"]

    for vid_file, output_folder in zip(video_files_list,
                                       output_folders_list):
        video_to_frames(vid_file,
                        output_folder)


if __name__ == "__main__":
    main()
