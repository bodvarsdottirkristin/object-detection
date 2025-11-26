# Improve road safety: Detecting potholes in the wild
**Introduction to Deep Learning in Computer Vision**
**October/November 2025**

A pothole is a depression in a road surface, usually asphalt pavement, where traffic has removed broken pieces of the pavement. It is typically the result of water in the underlying soil structure and traffic passing over the affected area. Potholes have a great impact on the road safety.

In this project, you are asked to build a deep-learning object detection system that can automatically detect potholes in images in the wild. This object detection can then be deployed in robotic machines or cars that can scan areas and improve road conditions. Detecting potholes in the wild can be a very challenging problem.

In this exercise, you will use the Potholes dataset (Fig. 1). You can find the dataset at `/dtu/datasets1/02516/potholes/`. The file contains the images and the annotation files in an XML PascalVOC-style format. The `splits.json` file contains the filenames of the training and test sets. Note that if you need a validation set you need to split the training part of the dataset.

There are several ways to read and parse this format in Python. You may find these links useful for that: link1, link2.

---

## Project 4.1 - Object proposals

**The task**
In this exercise, you will focus on the extraction, preparation, and evaluation of the object proposals. This is a useful step for the following exercises where you will build an object detector from scratch.

Your tasks are:
1.  Familiarise yourself with the data and visualize some examples with the ground-truth bounding boxes.
2.  Extract object proposals for all the images of the dataset (e.g. Selecting Search, Edge Boxes, etc). Note that you may have to resize the images before you run SS for better efficiency.
3.  Evaluate the extracted proposals on the training set of the dataset and determine the number of required proposals.
4.  Prepare the proposals for the training of the object detector. This requires assigning a label (i.e., class or background label) to each proposal.

**Hand-in**
Your process, performance evaluation, and results should be documented and discussed in a PDF report to be uploaded to DTU Learn. All three parts of this project should be described in the same report (up to 3 pages in total).

---

## Project 4.2 - Object detector

**The task**
In this exercise, you will focus on the building and training of an object detector.

Your tasks are:
1.  Build a convolutional neural network to classify object proposals ($N+1$ classes).
2.  Build a dataloader for the object detection task. Think about the class imbalance issue of the background proposals.
3.  Finetune the network on the training set.
4.  Evaluate the classification accuracy of the network on the validation set. Note that this is different from the evaluation of the object detection task.

**Hand-in**
Your process, performance evaluation, and results should be documented and discussed in a PDF report to be uploaded to DTU Learn. All three parts of this project should be described in the same report (up to 3 pages in total).

---

## Exercise 4.3 - Testing and object detector

**The task**
In this exercise, you will focus on testing and evaluating an object detector.

Your tasks are:
1.  Apply the CNN that you trained on the test images.
2.  Implement and apply NMS to discard overlapping boxes.
3.  Evaluate the object detection output using the Average Precision (AP) metric.

**Hand-in**
Your process, performance evaluation, and results should be documented and discussed in a PDF report to be uploaded to DTU Learn. All three parts of this project should be described in the same report (up to 3 pages in total).