🖼️ CIFAR-10 Image Classifier Web App
Built with TensorFlow • Keras • Streamlit
Live App: https://vm-image-classifier-using-cnn.streamlit.app/
linkedin: https://www.linkedin.com/in/vinay-meena-036119326/

This project is an interactive image classification web app powered by a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. Users can upload images to get real-time predictions, explore confidence scores, and analyze the model’s performance metrics — all in an intuitive web interface built using Streamlit.

✨ Features
✅ Single Image Prediction
Upload any 32x32 image and instantly get predictions from the trained CNN model.

✅ Confidence Scores
See a probability distribution across all 10 CIFAR-10 classes for any uploaded image.

✅ Full Model Evaluation
Explore the Confusion Matrix and Classification Report to assess per-class performance.

✅ User-Friendly Interface
Built using Streamlit for a seamless and interactive user experience.

🚀 How It Works
🔹 Dataset:
CIFAR-10 contains 60,000 color images (32x32 pixels) across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🔹 CNN Architecture:
A custom-built Convolutional Neural Network using:

Multiple Conv2D layers

BatchNormalization, ReLU, MaxPooling, and Dropout

Fully connected Dense layers for final classification

🔹 Training:

Trained using data augmentation for better generalization

EarlyStopping used to avoid overfitting and ensure optimal validation performance

💻 Tech Stack
Technology	Purpose
Python	Core programming language
TensorFlow & Keras	Model architecture, training, and evaluation
Streamlit	Web app interface and interaction
NumPy	Array and tensor computations
Pillow (PIL)	Image upload and manipulation in Streamlit
Scikit-learn	Confusion matrix, classification report
Matplotlib & Seaborn	Visualizing metrics and performance

🏁 Getting Started Locally
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/cifar10-image-classifier.git
cd cifar10-image-classifier
🔁 Replace your-username with your GitHub username.

2️⃣ Install Dependencies
Make sure you have Python installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
📌 Note: TensorFlow v2.8.0 is specified for compatibility.

3️⃣ Launch the App
bash
Copy
Edit
streamlit run cnn_web_app.py
🌐 The app will automatically open in your web browser. Ensure cifar10_cnn_model.h5 (trained model) is in the same directory.

📈 Model Performance
Test Accuracy: ~70–75% on CIFAR-10

Analyze performance class-wise with:

✅ Confusion Matrix

✅ Classification Report

You’ll find these in the “📊 Full Evaluation” section of the app.

🤝 Contributing
Feel free to fork the repo, suggest improvements, or submit pull requests. Contributions are welcome!

📄 License
This project is licensed under the MIT License — free to use, modify, and distribute.
