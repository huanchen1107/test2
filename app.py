import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Business Understanding: Predict linear relationship
# Example: y = ax + b + noise

# 2. Data Understanding (Generate synthetic dataset)
st.sidebar.title('Configuration')

# Streamlit widgets for user input
n = st.sidebar.slider('Number of data points (n)', 100, 5000, 1000)
a = st.sidebar.slider('Slope (a)', -10.0, 10.0, 3.0)
b = st.sidebar.slider('Intercept (b)', -10.0, 10.0, 5.0)
var = st.sidebar.slider('Noise Variance (var)', 0.0, 1000.0, 1.0)

st.title('Linear Regression with CRISP-DM workflow')

np.random.seed(42)
x = 2 * np.random.rand(n, 1)
noise = np.sqrt(var) * np.random.randn(n, 1)
y = a * x + b + noise

# 3. Data Preparation (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. Modeling (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write("Learned Coefficients:")
st.write("Slope (a):", model.coef_[0][0])
st.write("Intercept (b):", model.intercept_[0])
st.write("MSE:", mse)
st.write("RMSE:", rmse)
st.write("R^2 Score:", r2)

# Visualization
plt.scatter(X_test, y_test, color="blue", label="True data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Regression with sklearn (CRISP-DM workflow)")
# Identify top 5 outliers
errors = np.abs(y_test - y_pred)
top_5_outlier_indices = np.argsort(errors.flatten())[-5:]

for i in top_5_outlier_indices:
    plt.annotate(f'Outlier {i+1}', (X_test[i, 0], y_test[i, 0]),
                 textcoords="offset points", xytext=(0,10), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
st.pyplot(plt)
