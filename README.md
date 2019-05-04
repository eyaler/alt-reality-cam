# Alt Reality Cam (2019)
## by Mahanaim 134 (Eran Hadas and Eyal Gruss)
### Work in progress

ARC is a "camera" that finds and shows similar images with a possible semantic bias shift.

Image matching is based on spatial object layout, allowing for leeway in the number of instances, bounding box location and size, detection confidence, and categorical semantic hierarchy.

An example of the top 5 matches from Open Images v4 dataset (without adding a bias):

Source:
![Source](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/input_overlay.jpg "Source")

Match #1:
![Match #1](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/bias0_img1_overlay.jpg "Match #1")

Match #2:
![Match #2](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/bias0_img2_overlay.jpg "Match #2")

Match #3:
![Match #3](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/bias0_img3_overlay.jpg "Match #3")

Match #4:
![Match #4](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/bias0_img4_overlay.jpg "Match #4")

Match #5:
![Match #5](https://github.com/eyaler/alt-reality-cam/raw/master/demo/1/bias0_img5_overlay.jpg "Match #5")

More examples in [demo](https://github.com/eyaler/alt-reality-cam/tree/master/demo)

To run:

1) get_data.sh

2) prepare_data.py

3) arc.py
