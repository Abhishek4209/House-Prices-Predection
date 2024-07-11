

# Advanced House Prices Prediction
![alt text](image.png)
## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model](#model)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

## Introduction
This project aims to predict house prices using machine learning techniques. It leverages a dataset with various features that describe different aspects of the houses.

## Dataset Description
The dataset contains information about houses including but not limited to the number of bedrooms, bathrooms, and the area of the house. The goal is to use these features to predict the price of a house.

## Features
The following features are used in the dataset:

- **No of Bedrooms:** Number of bedrooms in the house.
- **No of Bathrooms:** Number of bathrooms in the house.
- **Flat Area (in Sqft):** The flat area of the house in square feet.
- **Lot Area (in Sqft):** The lot area of the house in square feet.
- **No of Floors:** Number of floors in the house.
- **Condition of the House:** Condition of the house on a scale.
- **Overall Grade:** Overall grade given to the house.
- **Area of the House from Basement (in Sqft):** The area of the house including the basement in square feet.
- **Basement Area (in Sqft):** The area of the basement in square feet.
- **Age of House (in Years):** The age of the house in years.
- **Living Area after Renovation (in Sqft):** The living area after renovation in square feet.
- **Lot Area after Renovation (in Sqft):** The lot area after renovation in square feet.

## Requirements
- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhishek4209/HousePricesPrediction
   ```
2. Navigate to the project directory:
   ```bash
   cd HousePricePrediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run the `model.ipynb` notebook to see the data analysis, model training, and predictions.

## Model
The project uses various regression models to predict house prices, including but not limited to:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGB Regressor

## Results
The results of the models are evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) score. The model with the best performance is chosen for the final predictions.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, please contact:
- Your Name - [Abhishek Upadhyay](mailto:abhishekupadhyay9336@gmail.com)
