🖼️ CIFAR-10 Image Classifier Streamlit App

This repository contains a Convolutional Neural Network (CNN) based image classifier for the CIFAR-10 dataset, packaged as an interactive web application using Streamlit. Users can upload an image to get a prediction or view the model's overall performance metrics on the CIFAR-10 test set.

✨ Features
Single Image Prediction: Upload an image and get a real-time prediction from the trained CNN model.

Confidence Scores: See the probability distribution across all 10 classes for the uploaded image.

Full Model Evaluation: View the Confusion Matrix and Classification Report for the model's performance on the entire CIFAR-10 test dataset.

User-Friendly Interface: Built with Streamlit for a simple and intuitive web experience.

🚀 How It Works
The application uses a CNN model trained on the CIFAR-10 dataset.

CIFAR-10 Dataset: Consists of 60,000 32x32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

CNN Model: A custom-built CNN architecture with multiple convolutional layers, Batch Normalization, ReLU activations, Max Pooling, and Dropout for robust feature learning and regularization.

Training: The model is trained using data augmentation and Early Stopping to prevent overfitting and achieve good generalization.

💻 Technologies Used
Python: The core programming language for the entire project.

TensorFlow & Keras: Used for building, training, and evaluating the Convolutional Neural Network (CNN) model.

Streamlit: The framework used to create the interactive web application.

NumPy: Essential for numerical operations, especially with image data and model predictions.

Pillow (PIL): Used for image loading and basic image manipulation within the Streamlit app.

Scikit-learn: Utilized for calculating performance metrics like the Confusion Matrix and Classification Report.

Matplotlib & Seaborn: Used for creating visualizations, particularly the Confusion Matrix heatmap and training plots.

🏁 Getting Started (Local Run)
To get a quick local instance of the app running:

Clone the Repository:

git clone https://github.com/your-username/cifar10-image-classifier.git
cd cifar10-image-classifier

(Remember to replace your-username with your actual GitHub username)

Install Dependencies:
Ensure you have Python installed, then install the required packages:

pip install -r requirements.txt

(Note: The requirements.txt specifies tensorflow==2.8.0 for optimal compatibility.)

Run the Streamlit App:

streamlit run cnn_web_app.py

This command will open the application in your default web browser. Make sure the cifar10_cnn_model.h5 file (your trained model) is in the same directory as cnn_web_app.py.

📈 Model Performance
The model typically achieves a test accuracy of ~70-75% on the CIFAR-10 dataset. The included "Full Model Evaluation" section in the app provides detailed metrics like the Confusion Matrix and Classification Report to analyze performance per class.

🤝 Contributing
Feel free to fork this repository, make improvements, and submit pull requests.

📄 License
This project is open-source and available under the MIT License.
