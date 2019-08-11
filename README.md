# Alt Reality Cam (2019)
## by Mahanaim 134: Eran Hadas and Eyal Gruss
### Developed for the "Deep Feeling: AI and emotions" exhibition at Petach Tikva Museum ([link](http://www.petachtikvamuseum.com/en/Exhibitions.aspx?aid=5007&eid=4987), [video](https://youtu.be/5QBdj40ZQ-Y))

ARC is a "camera" that finds and shows similar images, with a possible political or emotional semantic manipulation.

Image matching is based on spatial object layout, allowing for leeway in the number of instances, bounding box location and size, detection confidence, and categorical semantic hierarchy. Word embeddings are used for semantic manipulation.

An example of the top 5 matches from Open Images v5 dataset (with no manipulation):

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

An example of semantic manipulation matches from Open Images v5 dataset:

Source:
![Source](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/input_overlay.jpg "Source")

Gender (match #1):
![Match #1](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/bias1_img1_overlay.jpg "Gender (match #1)")

War (match #1):
![Match #2](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/bias2_img1_overlay.jpg "War (match #1)")

Money (match #2):
![Match #3](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/bias3_img2_overlay.jpg "Money (match #2)")

Love (match #4):
![Match #4](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/bias4_img4_overlay.jpg "Love (match #4)")

Fear (match #1):
![Match #5](https://github.com/eyaler/alt-reality-cam/raw/master/demo/2/bias5_img1_overlay.jpg "Fear (match #1)")


To run:

1) get_data.sh

2) prepare_data.py

3) arc.py

Or try our twitter bot by uploading an image and tagging [@altrealitycam](https://twitter.com/altrealitycam). use #zero for no manipulation, or one of #gender #war #money #love #fear if you feel adventurous. 