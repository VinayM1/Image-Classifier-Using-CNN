🖼️ CIFAR-10 Image Classifier Web App
Built with TensorFlow • Keras • Streamlit
🔗 Live App | 🔗 LinkedIn

🎯 Description
This project is an end-to-end image classification web app built with TensorFlow, Keras, and Streamlit, showcasing the power of Convolutional Neural Networks (CNNs) on the popular CIFAR-10 dataset.
Users can upload images to get real-time class predictions with confidence scores, and explore the model's performance through a Confusion Matrix and Classification Report — all wrapped in a sleek, interactive UI.
Whether you’re learning deep learning or showcasing your ML deployment skills, this app delivers a complete experience from training to deployment. 🚀

✨ Features
✅ Single Image Prediction
Upload any 32x32 image and instantly receive the model’s prediction.

✅ Confidence Scores
View probability distribution across all 10 classes for each image.

✅ Full Model Evaluation
Confusion Matrix + Classification Report for detailed analysis.

✅ Streamlit-Powered Interface
Simple, modern, and intuitive for everyone to use.

🚀 How It Works
📦 Dataset:
CIFAR-10 — 60,000 color images (32×32) across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

🧠 CNN Architecture:
Custom CNN with:

Multiple Conv2D layers

BatchNormalization, ReLU, MaxPooling, Dropout

Final Dense layers for classification

🛠️ Training:

Data Augmentation for generalization

EarlyStopping to prevent overfitting

💻 Tech Stack
Technology	Purpose
Python	Core programming
TensorFlow & Keras	Model building and training
Streamlit	Web app interface
NumPy	Numerical operations
Pillow (PIL)	Image manipulation in the app
Scikit-learn	Metrics (Confusion Matrix, Classification Report)
Matplotlib & Seaborn	Data visualization

🏁 Getting Started (Local)
🔧 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/cifar10-image-classifier.git
cd cifar10-image-classifier
Replace your-username with your GitHub handle.

📦 2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
TensorFlow version 2.8.0 is recommended for compatibility.

🚀 3. Launch the App
bash
Copy
Edit
streamlit run cnn_web_app.py
Ensure cifar10_cnn_model.h5 is in the same directory.

📈 Model Performance
Test Accuracy: ~70–75%

Full evaluation includes:

✅ Confusion Matrix

✅ Classification Report
Check the 📊 Full Evaluation tab inside the app for visual breakdowns.

🤝 Contributing
Open to pull requests, ideas, or collaboration!
Feel free to fork the repository and improve the app ✨

📄 License
This project is licensed under the MIT License — free to use, modify, and distribute.
