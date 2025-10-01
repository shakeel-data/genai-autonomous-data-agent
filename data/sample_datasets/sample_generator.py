"""
Sample Dataset Generator
Creates realistic sample datasets for testing the platform
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_datasets():
    """Create all sample datasets"""
    
    output_dir = Path("data/sample_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # E-commerce dataset
    ecommerce_df = generate_ecommerce_data(1000)
    ecommerce_df.to_csv(output_dir / "ecommerce_sales.csv", index=False)
    
    # Healthcare dataset  
    healthcare_df = generate_healthcare_data(800)
    healthcare_df.to_csv(output_dir / "healthcare_data.csv", index=False)
    
    # Financial dataset
    financial_df = generate_financial_data(1200)
    financial_df.to_csv(output_dir / "financial_market.csv", index=False)
    
    # Automotive dataset
    automotive_df = generate_automotive_data(600)
    automotive_df.to_csv(output_dir / "automotive_performance.csv", index=False)
    
    print("✅ Sample datasets created successfully!")

def generate_ecommerce_data(n_rows=1000):
    """Generate e-commerce sales data"""
    np.random.seed(42)
    
    segments = ['Premium', 'Standard', 'Budget']
    products = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'customer_segment': np.random.choice(segments, n_rows, p=[0.2, 0.5, 0.3]),
        'product_category': np.random.choice(products, n_rows),
        'region': np.random.choice(regions, n_rows),
        'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='D')[:n_rows],
        'quantity': np.random.randint(1, 10, n_rows),
        'unit_price': np.random.uniform(10, 500, n_rows).round(2),
        'discount_percent': np.random.uniform(0, 30, n_rows).round(1),
        'customer_age': np.random.randint(18, 70, n_rows),
        'is_repeat_customer': np.random.choice([True, False], n_rows, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    df['discount_amount'] = (df['total_amount'] * df['discount_percent'] / 100).round(2)
    df['final_amount'] = (df['total_amount'] - df['discount_amount']).round(2)
    
    return df

def generate_healthcare_data(n_rows=800):
    """Generate healthcare analytics data"""
    np.random.seed(42)
    
    conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Healthy']
    treatments = ['Treatment_A', 'Treatment_B', 'Treatment_C', 'No_Treatment']
    
    data = {
        'patient_id': range(1, n_rows + 1),
        'age': np.random.randint(20, 90, n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'condition': np.random.choice(conditions, n_rows, p=[0.15, 0.2, 0.15, 0.1, 0.4]),
        'treatment': np.random.choice(treatments, n_rows),
        'days_in_hospital': np.random.randint(0, 30, n_rows),
        'treatment_cost': np.random.uniform(1000, 50000, n_rows).round(2),
        'recovery_time': np.random.randint(1, 365, n_rows),
        'readmission': np.random.choice([True, False], n_rows, p=[0.15, 0.85])
    }
    
    return pd.DataFrame(data)

def generate_financial_data(n_rows=1200):
    """Generate financial market data"""
    np.random.seed(42)
    
    sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='H')
    
    data = {
        'timestamp': dates,
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'], n_rows),
        'sector': np.random.choice(sectors, n_rows),
        'volume': np.random.randint(1000000, 50000000, n_rows),
        'market_cap': np.random.uniform(1e9, 1e12, n_rows)
    }
    
    # Generate realistic price movements
    base_price = 100
    price_changes = np.random.normal(0, 2, n_rows)
    prices = [base_price]
    
    for change in price_changes[:-1]:
        new_price = prices[-1] * (1 + change/100)
        prices.append(max(new_price, 1))
    
    data['open_price'] = prices
    data['close_price'] = [p * (1 + np.random.normal(0, 0.5)/100) for p in prices]
    
    df = pd.DataFrame(data)
    df['price_change'] = ((df['close_price'] - df['open_price']) / df['open_price'] * 100).round(2)
    
    return df

def generate_automotive_data(n_rows=600):
    """Generate automotive performance data"""
    np.random.seed(42)
    
    makes = ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford', 'Audi']
    fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid']
    
    data = {
        'vehicle_id': range(1, n_rows + 1),
        'make': np.random.choice(makes, n_rows),
        'model_year': np.random.randint(2015, 2024, n_rows),
        'fuel_type': np.random.choice(fuel_types, n_rows, p=[0.4, 0.2, 0.2, 0.2]),
        'engine_size': np.random.uniform(1.0, 6.0, n_rows).round(1),
        'horsepower': np.random.randint(100, 500, n_rows),
        'weight': np.random.randint(1200, 2500, n_rows),
        'fuel_efficiency': np.random.uniform(15, 50, n_rows).round(1),
        'maintenance_cost': np.random.uniform(500, 5000, n_rows).round(2),
        'safety_rating': np.random.randint(3, 6, n_rows)
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    create_sample_datasets()
