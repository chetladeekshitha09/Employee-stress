# Employee-stress
This Python desktop app detects employee stress from tweets using machine learning and NLP. Users upload tweet datasets, preprocess data, and apply SVM or Random Forest algorithms to classify stress levels. The app extracts features, trains models, predicts stress, and visualizes accuracy, helping organizations monitor employee well-being.

## Detection of Employee Stress Using Machine Learning

In this project, I developed a **Tkinter-based desktop application** that detects employee stress from tweets using **machine learning**. The system integrates data preprocessing, feature extraction, model training, prediction, and visualization within an interactive graphical user interface.

The process begins with uploading a **CSV dataset** containing tweets and stress labels. Using **pandas**, the data is loaded, and tweets are preprocessed by converting to lowercase, removing stopwords (via NLTK), and filtering out very short words. This cleaning step ensures that only relevant words are retained for analysis. A **word count** is generated to guide feature extraction.

For feature representation, the application uses the **Keras Tokenizer** to convert text into numerical sequences, which are then padded to a fixed length using `pad_sequences`. The dataset is randomized and split into training (87%) and testing (13%) subsets using **scikit-learn’s** `train_test_split`.

Two classification models are implemented:
1. **Support Vector Machine (SVM)** with RBF kernel  
2. **Random Forest Classifier**  

Both models are trained on the extracted features, and their accuracies are calculated using `accuracy_score`. The Random Forest model is stored for later use in prediction.

The prediction feature allows users to upload a new dataset of tweets. Each tweet is preprocessed, converted into a sequence, padded, and passed to the trained Random Forest model to classify it as **“Stressed”** or **“Not Stressed”**.

The application also includes a visualization module that displays a **bar chart** comparing the accuracies of SVM and Random Forest models using **Matplotlib**.

By combining **Python**, **Tkinter**, **scikit-learn**, **Keras**, and **NLTK**, I built a complete machine learning pipeline with an accessible GUI. This project enhanced my skills in **text preprocessing, feature engineering, model evaluation**, and **user interface design**, while demonstrating how machine learning can be applied to real-world sentiment and stress detection tasks.
