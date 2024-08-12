# House Price Prediction Web Application

This project is a Flask-based web application that predicts house prices based on input features such as the number of bedrooms, bathrooms, size in square feet, and location. The model is trained using a dataset from Kaggle, and the application provides an interactive form to input house features and predict prices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Implementation](#model-implementation)
5. [Application Structure](#application-structure)
6. [Running the Application](#running-the-application)
7. [Snapshots](#snapshots)
8. [License](#license)

## Project Overview

This web application uses a machine learning model to predict house prices based on various input features. The application is built using Flask, and the model is implemented using the `scikit-learn` library.

## Installation

To run this application locally, you'll need to have Python installed. Follow the steps below to set up the environment and run the application.

### Prerequisites

- Python (version 3.7 or later): [Download Python](https://www.python.org/downloads/)
- Jupyter Notebook: [Install Jupyter Notebook](https://jupyter.org/install)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### Step 2: Install Required Libraries

You need to install the following Python libraries:

- Pandas: `pip install pandas`
- NumPy: `pip install numpy`
- Scikit-learn: `pip install scikit-learn`
- Flask: `pip install Flask`
- Jupyter Notebook: `pip install notebook`

Alternatively, you can install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset

The dataset used for this project can be downloaded from Kaggle:

[House Price Prediction - Seattle Dataset](https://www.kaggle.com/datasets/samuelcortinhas/house-price-prediction-seattle?resource=download)

After downloading, place the `train.csv` file in the project directory.

## Dataset

The dataset used in this project contains various features related to houses, such as the number of bedrooms, bathrooms, size in square feet, and zip code. The dataset is cleaned, and the final processed dataset is used for training the model.

## Model Implementation

The machine learning model is implemented using the `scikit-learn` library. The model used for this application is Ridge Regression, which is a type of linear model that applies L2 regularization to reduce overfitting.

### Key Steps:

1. **Data Cleaning**: Removing null values and irrelevant columns.
2. **Feature Engineering**: Adding new features such as `price_per_sqft`.
3. **Model Training**: Using Ridge Regression to train the model on the processed dataset.
4. **Model Serialization**: Saving the trained model using Python's `pickle` library.

## Application Structure

- **app.py**: The main Flask application file.
- **templates/index.html**: The HTML template for the user interface.
- **final_dataset.csv**: The processed dataset used for model training.
- **RidgeModel.pkl**: The serialized machine learning model.
- **requirements.txt**: List of required Python libraries.

## Running the Application

To run the Flask application, follow these steps:

1. **Run Jupyter Notebook**:
   - Open Jupyter Notebook: `jupyter notebook`
   - Run the notebook to clean data, train the model, and save the model as `RidgeModel.pkl`.

2. **Run Flask Application**:
   - Navigate to the project directory and run the Flask app:
   
   ```bash
   python app.py
   ```
   
   The application will be available at `http://127.0.0.1:5000/`.

3. **Use the Application**:
   - Open the application in your browser.
   - Enter the house features such as the number of bedrooms, bathrooms, size, and zip code.
   - Click on "Predict Price" to see the predicted house price.


## Snapshots

Here are some snapshots of the application:
### Form Submission
![Screenshot](templates/sanpshots/Screenshot%20(4).png)

![Screenshot 5](./templates/snapshots/Screenshot%20(5).png)

### Response
![Response](templates\sanpshots\Screenshot (5).png)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
