from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the file to a desired location
    file.save('/Users/bharath/Downloads/Major_project/data_set/data')
    
    # Perform any necessary processing on the uploaded file
    
    return "File uploaded successfully!"

@app.route('/analytics',methods=["GET", "POST"])
def sales_analytics():
    # Load data from CSV file
    data = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

    # Convert date column to datetime format
    data["date"] = pd.to_datetime(data["date_sale"], format='%d-%m-%Y')

    # Calculate total sales and average order value
    total_sales = data["total_revenue"].sum()
    average_order_value = data["total_revenue"].mean()

    # Calculate sales and average order value for each category
    category_sales = data.groupby("category")["total_revenue"].sum()
    category_avg_order_value = data.groupby("category")["total_revenue"].mean()

    # Find top 5 selling products for each category
    top_selling_products_by_category = {}
    for category, group in data.groupby("category"):
        top_selling_products_by_category[category] = group.nlargest(5, "quantity_stock")

    return render_template('analytics.html', total_sales=total_sales,
                           average_order_value=average_order_value,
                           category_sales=category_sales,
                           category_avg_order_value=category_avg_order_value,
                           top_selling_products_by_category=top_selling_products_by_category)

@app.route('/predict', methods=['GET', 'POST'])
def forecast():
    df = pd.read_csv("data_set/data")
    if request.method == 'POST':
        category = request.form['category']
        # Filter data for the selected category
        category_data = df[df['category'] == category]
        # Convert 'date_sale' column to datetime
        category_data['date_sale'] = pd.to_datetime(category_data['date_sale'], format='%d-%m-%Y')
        # Sort the data by 'date_sale'
        category_data.sort_values(by='date_sale', inplace=True)
        # Split data into training and testing sets
        train_size = int(len(category_data) * 0.8)
        train_data, test_data = category_data[:train_size], category_data[train_size:]
        # Train the ARIMA model
        model = ARIMA(train_data['quantity_sold'], order=(5,1,0))
        model_fit = model.fit()
        # Validate the model
        predictions = model_fit.forecast(steps=len(test_data))
        rmse = np.sqrt(mean_squared_error(test_data['quantity_sold'], predictions))
        # Make forecasts for future time periods
        future_forecast = model_fit.forecast(steps=12)
        # Calculate minimum, average, and maximum values
        min_forecast = int(future_forecast.min())
        avg_forecast = int(future_forecast.mean())
        max_forecast = int(future_forecast.max())

        return render_template("forecast.html", category=category, rmse=rmse, future_forecast=future_forecast)
    return render_template("forecast.html", category=None)

def recommend_near_expiry_products():
    # Read data from CSV file
    df = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

    # Convert Expiration_Date to datetime
    df['Expiration_Date'] = pd.to_datetime(df['expiry_date'])

    # Filter out products with past expiration dates
    df = df[df['Expiration_Date'] >= pd.Timestamp.today()]

    # Calculate days until expiration
    df['Days_Until_Expiry'] = (df['Expiration_Date'] - pd.Timestamp.today()).dt.days

    # Identify products nearing expiration (e.g., within 7 days)
    near_expiry_threshold = 7
    near_expiry_products = df[df['Days_Until_Expiry'] <= near_expiry_threshold]

    # Remove duplicate products
    near_expiry_products = near_expiry_products.drop_duplicates(subset=['product_id'])

    # Convert DataFrame to list of dictionaries for easy rendering in HTML
    recommendations = near_expiry_products.to_dict(orient='records')

    return recommendations


def recommend_restock():
    # Read inventory data from CSV file
    inventory_df = pd.read_csv("/Users/bharath/Downloads/Major_project/data_set/data")

    # Check for low stock and suggest restocking
    low_stock_recommendations = []
    for index, row in inventory_df.iterrows():
        if row["quantity_stock"] <= 300:
            stock_difference = 301 - row["quantity_stock"]
            recommendation = {
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "stock_difference": max(stock_difference, 0)  # Ensure non-negative stock difference
            }
            low_stock_recommendations.append(recommendation)

    # Remove duplicate low stock recommendations
    seen_product_names = set()
    unique_low_stock_recommendations = []
    for recommendation in low_stock_recommendations:
        if recommendation["product_name"] not in seen_product_names:
            unique_low_stock_recommendations.append(recommendation)
            seen_product_names.add(recommendation["product_name"])

    # Get near expiry recommendations
    near_expiry_recommendations = recommend_near_expiry_products()

    # Return both low stock and near expiry recommendations
    return unique_low_stock_recommendations, near_expiry_recommendations

@app.route('/inventory')
def inventory():
    # Call both recommendation functions
    low_stock_recommendations, near_expiry_recommendations = recommend_restock()
    return render_template('inventory.html', restock_recommendations=low_stock_recommendations, near_expiry_recommendations=near_expiry_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
