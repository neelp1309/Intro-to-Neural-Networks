Intro To Neural Network

This project explores two different applications of machine learning and neural networks:

Part A: Predicting signal quality using measurable signal parameters.

Part B: Recognizing digits in real-world images from the SVHN (Street View House Numbers) dataset.

🧠 Project Structure
Part A: Signal Quality Classification
Objective: Predict the signal quality based on input parameters.

Data: Structured tabular data with features representing signal characteristics.

Approach:

Data preprocessing and missing value handling

Feature scaling and transformation

Model building using Artificial Neural Networks (ANN) with TensorFlow/Keras

Evaluation using accuracy and loss metrics

Part B: Street View Digit Recognition (SVHN)
Objective: Build a digit classifier using the SVHN dataset.

Data: Image data (32x32 RGB) of digits captured from Google Street View.

Approach:

Data loading using h5py files

Image normalization and label encoding

Construction of a Convolutional Neural Network (CNN)

Model training and performance analysis

📊 Technologies & Tools
Python

Pandas, NumPy, Matplotlib, Seaborn

TensorFlow / Keras

scikit-learn

h5py (for SVHN dataset)

🧪 Results
Part A: Achieved high classification accuracy with well-tuned ANN for signal prediction.

Part B: Successfully built a CNN that generalizes well to digit recognition in natural scenes.

📌 Key Learnings
Effective preprocessing and feature engineering for tabular and image datasets.

Construction and training of ANN and CNN models.

Handling real-world image noise, distractors, and multi-digit labeling challenges.

Evaluating deep learning models using visualization and performance metrics.
