# capstone_project

demo_scripts are just for reference

1.) frame_extract_video.py attempts to target a specified video file in the folder location as the script <br />
and attempts to output the frames in a specified folder, also same location as the script.

2.) For training, use create_CNN.py
Create these folders in the same location as the script:

normal_images <br />
under_extruded_images <br /> 
over_extruded images <br />
validation_images ->> normal, over_extruded, under_extruded <br />

3.) For predicting, use project_main.py need to create folder called: <br />

images_to_try

Need to change the number range to loop over the specified images under def main():
