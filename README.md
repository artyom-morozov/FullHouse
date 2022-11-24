# 🃏 FullHouse

The project implements a program that helps to score card hands in Texas Holdem Poker using computer vision. The program takes an input in the form of a picture with 5 cards on the table. The program then identifies rank and suit of all the cards on the table using a trained neural network and prints out the rank of the hand and name of the combination.


![results](/output/predicted.jpg)

## Instructions

Install opencv and run the program.

The program takes 1 positional argument

- **image_name** -  filename for the image in the `input/` folder 


To test your own image take a picture of 5 cards on uniform background and add it to the `/input` folder.

Execution example 

```
python detect.py flush.jpg
```

The predicted image is generated in the `/output` folder.

## Background

Even though the problem of determining cards in images is achiavable using opencv alone. Which was demonstrated in some previous [papers](https://web.fe.up.pt/~niadr/PUBLICATIONS/LIACC_publications_2011_12/pdf/C62_Poker_Vision_Playing_PM_LPR_LFT.pdf). It was decided to use a Convolutional Neural network instead to classify images. The project relies on [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4), fast and accurate real-time neural network for determining objects in images. YOLO is an open-source library that provides an easy-to-use framework tuseful for training a custom object classifier that can detect objects in images and videos.

## Training Process

For the neural network detector to work well it requires a lot of data. It was decided to hand collect the card images.
For the purposes of the project a 30 second video for each of the 52 standard deck playing cards was recorded with changing lighting conditions.
![frame](https://user-images.githubusercontent.com/47092586/203698302-a49dc1c0-3369-47cc-9823-76daf71f4768.png)

A custom script `create_dataset.ipynb` that heavily utilizes opencv, was then used to extract the card from the image and determine the convex hull of the rank and suit.

![card](https://user-images.githubusercontent.com/47092586/203697993-4a0b2057-c1b4-408c-beae-6b7f0de8d4cb.png)

The script then generates a dataset of 60000 images with random backgrounds using [imagaug](https://github.com/aleju/imgaug) library.

![dataset_sample](https://user-images.githubusercontent.com/47092586/203698527-a30b22fc-17c9-4a7f-b8ed-540300843706.png)

The dataset was then used to train the yolo model and classify images using `detect.py` script

![result](https://user-images.githubusercontent.com/47092586/203698734-27e2a6b3-d19a-498a-9423-9ebdb6eb210e.png)
