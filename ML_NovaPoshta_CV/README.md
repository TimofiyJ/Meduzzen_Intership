Environment: IDE: Visual Studio Code, Jupyter notebook, Google Colab; RAM: 16 GB; Processor: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz 2.80 GHz.
<br>1. Problem:
<br>The ukrainian postal company "Nova Poshta" has given data and asked to get type of postoffice box out of the picture, find out how many cells does this box has, discover whether it is lobby, outdoor or indoor. This is a tough problem, but it is possible to solve.
<br>2. Challanges:
<br>Object detection in an image, object classification, determine focal length.
<br>3. Data exploration:
<br>After analysing the given data I have come to the main conclusions:
<br>a. There are 48 different types of postoffice boxes. Each type has it's own amount of stacks, cells and possible location. There are 47 pictures of types and lack of 3rd type, I got it from Excel file.
<br>b. There are 3 different locations of box: indoor, outdoor, lobby.
<br>c. Nova Poshta has given dimensions of each type of cell.
<br>4. Data optimization:
<br> At the beginning I was given dataset with < 200 jpg pictures, but was informed that future data could be in different formats: jpg, jpeg, png, frame of video, etc, so I should take into consideration and remember this information during data preprocessing. In the current dataset filename of each image contains index of postoffice box. Some of the images cut the postoffice box so it would be very hard to classify which type of box it is and such data should be deleted. Main colors of the boxes are gray, black and red. The color does not impact on the box type.
<br>5. Data preprocessing:
<br> The main challange of the data preprocessing is to execute homography to shift perspectives correctly. Once the frontal view of postoffice box is done I would extract features (such as straight lines or Histogram of Oriented Gradients). But firstly and most importantly I need to find the postoffice box. Here, the filename would help me in such way: I know what is the index of postoffice box, having this information I could find the number on the picture and then get the color of the box (it would be the neighbor color of the index), using color-manipulation techniques (such as color segmentation with LAB color space)
<p align="center">
    <img width="500" src="https://github.com/TimofiyJ/Meduzzen_Intership/blob/main/ML_NovaPoshta_CV/media/LAB_example_1.png" alt="EG1">
</p>
<p align="center">
    <img width="500" src="https://github.com/TimofiyJ/Meduzzen_Intership/blob/main/ML_NovaPoshta_CV/media/LAB_example_2.png" alt="EG2">
</p>
I would get the position of the postoffice box by applying the mask created with color extraction and minimum area rectangle. (or i could just use homography for the index to get the front view of the box)
<br>During the image preprocessing stage I would consider to resize and reduce file size if the image would be too big.<br>Since I don't have a large dataset to experiment with I won't consider solution that requires deep learning. I believe that having the given data it is more useful to focus on getting the frontal view of the box, extract features (horizontal and vertical lines using kernels, HOG + SVM, etc.) or identify single cells having their sizes, borders on the picture and how many pixels are in 1 cm.
<br> 6. Metadata:
<br> Hidden gold in this problem is metadata of the images. It consists of location of the picture, dimensions, resolution, bit depth, F-stop, focal lenghth which would help to get how many pixels in one cm and much more.
<p align="center">
    <img width="500" src="https://github.com/TimofiyJ/Meduzzen_Intership/blob/main/ML_NovaPoshta_CV/media/metadata_location_example.png" alt="EG2">
</p>
<br> 7. Thoughts on libs:
<br> To my mind libraries are the things that can change during the process so I don't see much sense in limiting myself. I can name basic libraries that I use in every CV project:
<br> OpenCV, numpy, pandas, tensorflow.
<br> During the progress I would expand this list but right now most of operations would be covered using these libraries.
<br> 8. Main framework:
<br> The main tech stack would be based on recommendations and my personal experience:
<br> Python 3.11, JSON, Paydantic, Gradio, FastAPI, TensorFlow, OpenCv, Numpy, Pandas.
<br> 9. Plan of the project:
<br> a. EDA in jupyter notebook file
<br> b. Image optimization
<br> c. Postoffice box extraction
<br> d. Homography
<br> e. Discover how namy pixels in cn
<br> f. Classification of cells
<br> g. Classification of postoffice box
<br>
<br> 10. Useful sources:
<br> https://learnopencv.com/tag/homography/
<br> https://towardsdatascience.com/understanding-transformations-in-computer-vision-b001f49a9e61
<br> https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
<br>https://study.com/skill/learn/how-to-use-the-lens-equation-to-find-the-distance-of-an-object-from-a-lens-explanation.html
<br>https://www.youtube.com/watch?v=MmBBVTniWFg
<br>https://www.dynamicmarketing.sg/blog/image-optimization-what-is-it-and-why-it-matters/
<br>https://en.wikipedia.org/wiki/F-number
