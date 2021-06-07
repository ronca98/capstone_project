# capstone_project

demo_scripts are just for reference

1.) frame_extract_video.py attempts to target a specified video file in the folder location called: "video_files" <br />
and attempts to output the frames in a specified folder, also same location as the script.

2.) For training, use create_CNN.py <br />
Script will read training data from folder: "data_set" and will contain these subfolders with .png image data

no_pattern <br />
normal_images <br />
under_extruded_images <br /> 
over_extruded images <br />

3.) For predicting, use project_main.py and you need to create folder called: <br />

"images_to_try" (must be in same folder as script) <br />
After prediction is completed, it will create two files for results in same folder as script: <br />

prediction_results.png <br />
prediction_results.csv



