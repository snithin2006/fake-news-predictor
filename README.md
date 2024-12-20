# Fake News Detection API

This project is a backend solution for detecting fake news using machine learning. The model is trained on data in the `data` folder, and the backend API allows users to locally query the model to predict whether a given news article is real or fake.

## Project Overview
The goal of this project is to provide an API that leverages machine learning for fake news detection. It trains a machine learning model based on a dataset of news articles, which is then used to predict whether a given piece of news is real or fake.

- The model is stored in the `model` folder, where it is trained using the data from the `data` folder.
- Once the model is trained, it is saved and used in the `backend` folder, where the Flask-based API is created to serve predictions.

## Technologies Used
- **Flask**: A lightweight web framework for Python used to create the API that allows users to query the model.
- **Python**: The primary language used for both the machine learning model and the API backend.
- **Machine Learning Libraries**: 
  - **scikit-learn**: A popular library for machine learning in Python used for model training and evaluation.
  - **TensorFlow** or **Keras**: These libraries are used for more advanced machine learning models (if applicable).
- **Pandas** and **NumPy**: Libraries for data manipulation and processing.
- **Joblib** or **Pickle**: Libraries for saving and loading the trained model for use in predictions.

## Model Training
The model is trained using a dataset stored in the `data` folder. The training process involves preprocessing the text data, converting it into numerical features, and using machine learning algorithms to classify news articles as either real or fake.

Once trained, the model is saved for later use in the backend API. The saved model can be loaded into the Flask app, which uses it to predict the authenticity of news articles based on user input.

## How It Works
1. **Training**: The model is trained using data from the `data` folder.
2. **API**: The backend API is built using Flask and exposes endpoints for querying the trained model.
3. **Prediction**: When a request is made to the API with a news article, the backend uses the trained model to predict if the article is real or fake and returns the result.

This project demonstrates how machine learning can be used to solve real-world problems such as detecting fake news. It provides a simple, scalable backend solution that can be expanded for use in a full-fledged application.

## Acknowledgements

This project uses the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), which contains labeled statements for the task of fake news detection.

### Citation:

*William Yang Wang. "LIAR: A Benchmark Dataset for Fake News Detection." *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017).* Available at [arXiv](https://arxiv.org/abs/1705.00648).*
