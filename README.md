# Steel-Defect-Detection
Steel Defect Detection Application
This repository contains a web application and associated Python scripts to detect steel defects using a Vanilla U-Net model. The application allows users to upload images of steel surfaces, and it processes the images to predict and display the detected steel defects.

## Files:
### app_new.py:

This file contains the Streamlit web application for steel defect detection. Users can upload single or multiple images of steel surfaces through the web interface. The uploaded images are then processed using the Vanilla U-Net model to predict and visualize the detected steel defects. The application provides an interactive way for users to detect defects in steel surfaces.

### main.py:

This Python script is a standalone version of the steel defect detection process. It uses the "Vanila_Unet_model" and "process_image" modules to predict steel defects in a single image specified by the 'path' variable. The script visualizes the original image and the predicted mask side by side using Matplotlib.

### vanila_unet_model.py:

This file contains the implementation of the Vanilla U-Net model used for steel defect detection. The model architecture consists of convolutional and transpose convolutional layers with skip connections. It is compiled with a custom loss function and used to predict steel defects.

## Instructions:
- Clone the repository to your local machine.
- Install the required dependencies using pip:
```bash
pip install tensorflow keras opencv-python streamlit matplotlib
```
- To use the web application, run the "app_new.py" file:
```bash
streamlit run app_new.py
```
For standalone image prediction, modify the "path" variable in "main.py" to specify the path of the image you want to test. Then run the script:
```bash
python main.py
```
Please note that the web application relies on the "Vanila_Unet_model" and "process_image" modules, so ensure you have all the required files in your working directory.
