# Final project: Remaining puzzle pieces counter
### Area: Image segmentation

## Proposal
The task of build a puzzle is very nice and could be hard sometimes, depending on the size of the puzzle. In cases that the puzzle has more than five hundred, a thousand of pieces or even more, it's quiet interesting if we had a tool that could count the remaining pieces with the objective to observe your evolution. Thinking in this, thi project proposes the development of an algorithm with the objective of detecting how many pieces are remaining of a puzzle, based on an image of the remaining pieces, on a surface.

## Input images
The input images of this project were collected by taking photos of a puzzle. The type of the file is .png and they were taken in a blank surface. Below we can see two examples of the photos.

![](/Final Project/images/4.png)
**Figure 1** - First example of input image

![](/Final Project/images/7.png)
**Figure 2** - Second example of input image

The other images can be found in [Images folder](/Final Project/images/)
## Steps to reach the objective
For this project, we will define some steps to develop the objective of this tool and, when obtaining them, improvements will be added. The steps are:

- Check if the image is good enough to perform the identification process, that is, check if the blank surface is sufficient for this (segmentation).
- If needed, an API will be used to improve the image background.
- Count how many pieces are in the figure using a greedy solution.
- Separate each piece of the puzzle using masks and try to find the one that best fits in a set of assembled pieces (feature extraction and color descriptors).

## Restrictions
To follow restrictions needs to be considered to reach the objective well.

- The pieces can't be superposed.
- Good ilumination is needed to the images to preserve it's characteristics.
- The captured images needs to have an 90$^{\circ}$ angle with the surface to avoid problems with positioning.
- The puzzle pieces must stay in a blank surface or in a surface that preserves good enough it's colors.

## First results

The first results can be founded in [Notebook](/Final Project/final_project.ipynb) and the scripts are located in [Scripts folder](/Final Project/scripts/)
