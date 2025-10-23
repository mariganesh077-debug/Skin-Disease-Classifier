🩺 Skin Disease Classification from Images 🌿

📘 Overview

Skin diseases affect millions worldwide, and early detection plays a vital role in effective treatment. However, accurate diagnosis often requires expert dermatologists, which can be time-consuming and expensive.

This project aims to develop an intelligent, automated skin disease classification system using Deep Learning (CNNs) that can analyze dermoscopic images and classify them into various skin disease categories such as:

🌞 Melanoma

🌸 Eczema

🌿 Psoriasis

🌼 Acne

🍃 Healthy Skin

🎯 Objective

To build a robust and efficient image classification model that can detect and classify skin diseases from dermoscopic images with high accuracy — assisting medical professionals and improving accessibility to dermatological diagnosis.

🧠 Technologies Used
Category	Tools/Frameworks
💻 Programming Language	Python
🧩 Deep Learning Framework	TensorFlow / Keras
🧮 Data Handling	NumPy, Pandas
🖼️ Image Processing	OpenCV, PIL
📊 Visualization	Matplotlib, Seaborn
🔍 Model Evaluation	Scikit-learn
⚙️ How It Works

Image Preprocessing:

Images are resized, normalized, and augmented for better generalization.

Model Training:

A Convolutional Neural Network (CNN) is trained on the dataset to learn distinguishing features of different skin conditions.

Evaluation:

The model is tested using accuracy, precision, recall, and F1-score.

Prediction:

Upload an image, and the system predicts the corresponding skin disease class.

🧩 Project Structure

📁 Skin-Disease-Classification/

├── 📄 README.md

├── 📂 dataset/

│   ├── train/

│   ├── test/

│   └── validation/

├── 📂 models/

│   └── skin_disease_model.h5

├── 📂 notebooks/

│   └── skin_disease_classification.ipynb

├── 📂 static/

│   └── sample_images/

├── 📄 requirements.txt

└── 📄 app.py   (for Streamlit or Flask app)


🚀 How to Run

🧰 Prerequisites

Make sure you have the following installed:

Python 3.8 or above

pip

⚙️ Installation Steps

# Clone the repository
git clone https://github.com/mariganesh077-debug/Skin-Disease-Classification.git
cd Skin-Disease-Classification

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

🌐 If using Streamlit:
streamlit run app.py


Then open your browser at http://localhost:8501/
 🌍

📈 Results

✅ Accuracy: ~90% (varies based on dataset and training time)

📊 Model: CNN with multiple convolutional and pooling layers

🧾 Loss Function: Categorical Crossentropy

⚡ Optimizer: Adam

🧬 Future Enhancements

✨ Integrate Explainable AI (Grad-CAM) to visualize model attention
✨ Expand dataset with more diverse skin tones and conditions
✨ Deploy the model on a cloud platform or mobile app for real-world use

🤝 Contribution

Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests to make this project better.

🛡️ License

This project is licensed under the MIT License — you are free to use, modify, and distribute it.

💬 Acknowledgments

Special thanks to:

🌍 Public skin disease image datasets (like HAM10000)

🧠 TensorFlow and Keras community

💡 Open-source contributors
