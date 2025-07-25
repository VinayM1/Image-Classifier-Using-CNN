import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --- Load the Trained CNN Model ---
@st.cache_resource  # Cache the model loading to avoid reloading on every rerun
def load_cnn_model():
    try:
        model = tf.keras.models.load_model('cifar10_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Make sure 'cifar10_cnn_model.h5' is in the same directory.")
        return None


# Define the class names for CIFAR-10 (must match the order used during training)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# --- Load and Preprocess CIFAR-10 Test Data for Full Evaluation ---
# This data is needed to generate the confusion matrix and classification report
@st.cache_data  # Cache this data loading as it's static
def load_cifar10_test_data():
    # Load raw data
    (train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = tf.keras.datasets.cifar10.load_data()

    # Cast to float32 and normalize
    test_images = tf.cast(test_images_raw, tf.float32) / 255.0
    # Flatten labels to 1D int32 array for scikit-learn metrics
    test_labels = test_labels_raw.flatten().astype('int32')
    return test_images, test_labels


# Load the model and test data when the app starts
model = load_cnn_model()
full_test_images, full_test_labels = load_cifar10_test_data()


# --- Image Preprocessing Function (for single uploaded image) ---
def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0
    return tf.cast(img_array, tf.float32)  # Ensure float32 dtype for consistency with model input


# --- Streamlit UI ---

st.title("🖼️ CIFAR-10 Image Classifier")
st.markdown("Upload an image and let the CNN predict what it is!")

st.info(
    f"This model is trained on the CIFAR-10 dataset and can only predict images belonging to these categories: {', '.join(class_names)}.")
st.warning("🔍 **Note:** While the model performs well on most classes, it struggles to accurately classify images of **cats** and some similar-looking animals. Results for these categories may be less reliable.")


if model is None:
    st.warning("Model could not be loaded. Please ensure 'cifar10_cnn_model.h5' is in the correct path.")
else:
    # --- Single Image Prediction Section ---
    st.subheader("Predict a Single Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption='Uploaded Image', use_container_width=True)
        st.write("")

        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                processed_image = preprocess_image(image_to_predict)
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                st.success(f"Prediction: **{predicted_class_name}** (Confidence: {confidence:.2f}%)")

                st.subheader("All Class Probabilities:")
                prob_dict = {name: prob * 100 for name, prob in zip(class_names, predictions[0])}
                sorted_prob_dict = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)

                for class_name_prob, prob_val in sorted_prob_dict:
                    st.write(f"- {class_name_prob}: {prob_val:.2f}%")

    # --- Full Model Evaluation Section ---
    st.markdown("---")
    st.subheader("Full Model Performance on CIFAR-10 Test Set")
    st.markdown("This section shows how the model performs on the entire 10,000-image CIFAR-10 test set.")

    if st.button("Show Full Evaluation Metrics"):
        with st.spinner("Calculating full evaluation... This might take a moment."):
            # Make predictions on the entire test set
            y_pred_probs = model.predict(full_test_images)
            y_pred_labels = np.argmax(y_pred_probs, axis=1)

            # --- Display Confusion Matrix ---
            st.markdown("#### Confusion Matrix")
            st.markdown("Rows are True Labels, Columns are Predicted Labels. Diagonal values are correct predictions.")

            cm = confusion_matrix(full_test_labels, y_pred_labels)

            fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)  # Display the plot in Streamlit

            # --- Display Classification Report ---
            st.markdown("#### Classification Report")
            st.markdown("Provides Precision, Recall, and F1-Score for each class.")

            report = classification_report(full_test_labels, y_pred_labels, target_names=class_names, output_dict=True)

            # Convert report to a DataFrame for better display in Streamlit
            import pandas as pd

            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)

    st.markdown("---")
    st.markdown("This app uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.")

