# Gho-Kira-Detection-CNN
This project was done to study and understand how Convolutional Neural Networks worked.
While implementing this project I used the tensorflow flower classification tutorial as a reference and adapted it for Gho and Kira image classification task.

# How it Works
In the test folder, there is a script where the code will read the trained and saved model which is in pickel format. After it has been loaded, the program will try to identify the images listed in the program file.
If you want to test it with different images, edit the program file where the image file name are loaded; you just have to change the image name.

# About Notebook
The jupyter notebook contains the code used to accomplish the task.
The model was trained on 600 images of gho and 600 images of kira collected through our known friends.
The dataset was increased using image augmentation
NOTE: The dataset is not uploaded for privacy reasons.

# Running
1. Install the necessary libraries depending on the imports.
2. Run the test script with `$ python predict-gho-kira.py`
3. See the predicted results made by the trained model

# Future tasks
- Increase the dataset and use transfere learing to improve the model accuracy
- Try with Transformer models to see if this architectures produces better accuracy
