# Project Title: Streamlit Linear Regression with Outlier Detection

## Project Overview
This project implements a simple linear regression model using `scikit-learn` and visualizes the results with `matplotlib` within a `Streamlit` web application. Users can interactively adjust parameters like the number of data points, slope, intercept, and noise variance to observe their impact on the linear regression model. The application also identifies and labels the top 5 outliers in the generated dataset.

## Technologies Used
- Python
- Streamlit
- NumPy
- scikit-learn
- Matplotlib

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/huanchen1107/test2.git
    cd test2
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser, usually at `http://localhost:8501`.

## Features Implemented

-   **Interactive Data Generation:** Adjust the number of data points (`n`), true slope (`a`), true intercept (`b`), and noise variance (`var`) using sidebar sliders.
-   **Linear Regression Model:** Trains a linear regression model on the generated data.
-   **Model Evaluation:** Displays learned coefficients (slope and intercept), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score.
-   **Visualization:** Plots the generated data points and the predicted regression line.
-   **Outlier Detection:** Identifies and labels the top 5 data points with the largest absolute errors (outliers) on the plot.
-   **Responsive Layout:** Uses Streamlit's sidebar for input controls, providing a clean left and right layout.

## Future Improvements

-   Add more regression algorithms (e.g., Polynomial Regression, Ridge, Lasso).
-   Implement cross-validation for more robust model evaluation.
-   Allow users to upload their own datasets.
-   Add more detailed explanations and interpretations of the metrics and plots.
-   Improve outlier labeling to be more dynamic or offer different outlier detection methods.
-   Add unit tests for the core logic.
