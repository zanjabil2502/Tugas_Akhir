#INFERENCE PROGRAM
=============================
In this repository there is a dataset folder filled with the COMPARE 2021 dataset.

There are two inference program files, for the file with the name 'inference_program.py' for decision making, mode statistics are used from the detection results of the segmented cough sound. Then for the file with the name 'inference_program-if.py' for decision making use the 'if' command if the segmentation result is detected positive, it will be declared positive for all

-------------------------------------------------------------------------------

If you want to test the inference program using the terminal or command prompt, use the command below:

python inference_program.py -f your_path_audio.wav

-------------------------------------------------------------------------------
If you want to test the inference program using a notebook, use the file 'test_inference_program.ipynb'