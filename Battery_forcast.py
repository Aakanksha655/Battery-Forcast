import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Deep Learning (optional - uncomment if tensorflow is available)
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

class BatteryEnergyForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.forecast_results = {}
        
    def generate_synthetic_data(self, days=365, start_date='2023-01-01'):
        """Generate synthetic battery manufacturing energy consumption data"""
        
        # Create datetime index
        start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=days*24, freq='H')
        
        # Base parameters for different equipment
        np.random.seed(42)
        
        # Generate realistic energy consumption patterns
        data = []
        
        for i, dt in enumerate(dates):
            hour = dt.hour
            day_of_week = dt.dayofweek
            month = dt.month
            
            # Base consumption patterns
            # Oven energy consumption (kWh)
            oven_base = 450 + 100 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            oven_seasonal = 30 * np.sin(2 * np.pi * month / 12)  # Seasonal variation
            oven_noise = np.random.normal(0, 15)
            oven_efficiency = 0.85 + 0.1 * np.random.random()  # Efficiency factor
            
            # Chamber energy consumption (kWh)
            chamber_base = 320 + 80 * np.sin(2 * np.pi * hour / 24 + np.pi/4)
            chamber_seasonal = 20 * np.sin(2 * np.pi * month / 12)
            chamber_noise = np.random.normal(0, 12)
            chamber_efficiency = 0.80 + 0.15 * np.random.random()
            
            # Production variables
            production_rate = 1000 + 200 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 50)
            production_rate = max(500, production_rate)  # Minimum production
            
            # Temperature settings
            oven_temp = 850 + 50 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 10)
            chamber_temp = 25 + 5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            
            # External factors
            ambient_temp = 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 3)
            humidity = 45 + 15 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5)
            
            # Calculate actual energy consumption
            oven_energy = (oven_base + oven_seasonal + oven_noise) / oven_efficiency
            chamber_energy = (chamber_base + chamber_seasonal + chamber_noise) / chamber_efficiency
            
            # Add production dependency
            production_factor = production_rate / 1000
            oven_energy *= production_factor
            chamber_energy *= production_factor
            
            # Weekend reduction
            if day_of_week >= 5:  # Weekend
                oven_energy *= 0.7
                chamber_energy *= 0.7
                production_rate *= 0.6
            
            # Total energy
            total_energy = oven_energy + chamber_energy
            
            data.append({
                'datetime': dt,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'oven_energy_kwh': oven_energy,
                'chamber_energy_kwh': chamber_energy,
                'total_energy_kwh': total_energy,
                'production_rate_units': production_rate,
                'oven_temperature_c': oven_temp,
                'chamber_temperature_c': chamber_temp,
                'ambient_temperature_c': ambient_temp,
                'humidity_percent': humidity,
                'oven_efficiency': oven_efficiency,
                'chamber_efficiency': chamber_efficiency
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df
    
    def create_features(self, df):
        """Create additional features for forecasting"""
        df_features = df.copy()
        
        # Time-based features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df_features[f'total_energy_lag_{lag}h'] = df_features['total_energy_kwh'].shift(lag)
            df_features[f'oven_energy_lag_{lag}h'] = df_features['oven_energy_kwh'].shift(lag)
            df_features[f'chamber_energy_lag_{lag}h'] = df_features['chamber_energy_kwh'].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            df_features[f'total_energy_rolling_mean_{window}h'] = df_features['total_energy_kwh'].rolling(window=window).mean()
            df_features[f'total_energy_rolling_std_{window}h'] = df_features['total_energy_kwh'].rolling(window=window).std()
        
        # Energy efficiency ratios
        df_features['energy_per_unit'] = df_features['total_energy_kwh'] / df_features['production_rate_units']
        df_features['oven_chamber_ratio'] = df_features['oven_energy_kwh'] / df_features['chamber_energy_kwh']
        
        # Temperature differentials
        df_features['temp_differential'] = df_features['oven_temperature_c'] - df_features['ambient_temperature_c']
        
        # Drop rows with NaN values (due to lag features)
        df_features.dropna(inplace=True)
        
        return df_features
    
    def train_models(self, df, target_col='total_energy_kwh'):
        """Train multiple forecasting models"""
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_col] = scaler
        
        # Model 1: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        
        # Model 2: Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        
        # Model 3: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Store models
        self.models[target_col] = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'linear_regression': lr_model
        }
        
        # Evaluate models
        results = {}
        for name, model in self.models[target_col].items():
            y_pred = model.predict(X_test_scaled)
            results[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Feature importance (Random Forest)
        self.feature_importance[target_col] = pd.Series(
            rf_model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        return results, X_test, y_test, X_test_scaled
    
    def forecast_energy(self, df, hours_ahead=24, target_col='total_energy_kwh'):
        """Generate energy consumption forecast"""
        
        # Use the best model (Random Forest typically performs well)
        model = self.models[target_col]['random_forest']
        scaler = self.scalers[target_col]
        
        # Prepare last known data point
        last_row = df.iloc[-1:].copy()
        forecasts = []
        
        for h in range(hours_ahead):
            # Create future timestamp
            future_time = df.index[-1] + timedelta(hours=h+1)
            
            # Extract time features
            hour = future_time.hour
            day_of_week = future_time.dayofweek
            month = future_time.month
            
            # Update time-based features
            last_row['hour'] = hour
            last_row['day_of_week'] = day_of_week
            last_row['month'] = month
            last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            last_row['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            last_row['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            last_row['month_sin'] = np.sin(2 * np.pi * month / 12)
            last_row['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Scale features
            feature_cols = [col for col in last_row.columns if col != target_col]
            X_scaled = scaler.transform(last_row[feature_cols])
            
            # Make prediction
            forecast = model.predict(X_scaled)[0]
            forecasts.append({
                'datetime': future_time,
                'forecast_energy_kwh': forecast
            })
            
            # Update lag features for next iteration (simplified)
            # In a real scenario, you'd update all lag features properly
            
        return pd.DataFrame(forecasts)
    
    def optimize_energy_consumption(self, df, optimization_hours=24):
        """Optimize energy consumption using forecasts"""
        
        # Get forecast
        forecast_df = self.forecast_energy(df, optimization_hours)
        
        # Identify optimization opportunities
        optimizations = []
        
        for idx, row in forecast_df.iterrows():
            forecast_energy = row['forecast_energy_kwh']
            
            # Define optimization strategies
            strategies = {
                'temperature_optimization': {
                    'energy_saving_percent': 8,
                    'description': 'Optimize oven temperature profile'
                },
                'production_scheduling': {
                    'energy_saving_percent': 12,
                    'description': 'Schedule production during off-peak hours'
                },
                'equipment_efficiency': {
                    'energy_saving_percent': 15,
                    'description': 'Improve equipment efficiency'
                },
                'predictive_maintenance': {
                    'energy_saving_percent': 6,
                    'description': 'Prevent efficiency degradation'
                }
            }
            
            # Calculate potential savings
            total_savings = 0
            applied_strategies = []
            
            for strategy, params in strategies.items():
                if forecast_energy > 600:  # High consumption threshold
                    savings = forecast_energy * (params['energy_saving_percent'] / 100)
                    total_savings += savings
                    applied_strategies.append(strategy)
            
            optimizations.append({
                'datetime': row['datetime'],
                'forecast_energy_kwh': forecast_energy,
                'optimized_energy_kwh': forecast_energy - total_savings,
                'energy_savings_kwh': total_savings,
                'savings_percent': (total_savings / forecast_energy) * 100,
                'strategies': applied_strategies
            })
        
        return pd.DataFrame(optimizations)
    
    def plot_results(self, df, forecast_df, optimization_df):
        """Create comprehensive visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Historical energy consumption
        axes[0, 0].plot(df.index[-168:], df['total_energy_kwh'].tail(168), 
                       label='Historical', color='blue', alpha=0.7)
        axes[0, 0].set_title('Historical Energy Consumption (Last 7 Days)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Energy (kWh)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Forecast vs Optimization
        axes[0, 1].plot(forecast_df['datetime'], forecast_df['forecast_energy_kwh'], 
                       label='Forecast', color='red', linewidth=2)
        axes[0, 1].plot(optimization_df['datetime'], optimization_df['optimized_energy_kwh'], 
                       label='Optimized', color='green', linewidth=2)
        axes[0, 1].set_title('Energy Forecast vs Optimized Consumption')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Energy (kWh)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy savings over time
        axes[1, 0].bar(range(len(optimization_df)), optimization_df['energy_savings_kwh'], 
                      color='green', alpha=0.7)
        axes[1, 0].set_title('Energy Savings by Hour')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Energy Savings (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature importance
        top_features = self.feature_importance['total_energy_kwh'].head(10)
        axes[1, 1].barh(range(len(top_features)), top_features.values)
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels(top_features.index)
        axes[1, 1].set_title('Top 10 Feature Importance')
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Energy components breakdown
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Last 7 days breakdown
        recent_data = df.tail(168)
        ax.plot(recent_data.index, recent_data['oven_energy_kwh'], 
               label='Oven Energy', color='red', alpha=0.7)
        ax.plot(recent_data.index, recent_data['chamber_energy_kwh'], 
               label='Chamber Energy', color='blue', alpha=0.7)
        ax.plot(recent_data.index, recent_data['total_energy_kwh'], 
               label='Total Energy', color='black', linewidth=2)
        
        ax.set_title('Energy Consumption Breakdown (Last 7 Days)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    
    def generate_report(self, df, forecast_df, optimization_df, model_results):
        """Generate comprehensive energy optimization report"""
        
        total_forecast_energy = forecast_df['forecast_energy_kwh'].sum()
        total_optimized_energy = optimization_df['optimized_energy_kwh'].sum()
        total_savings = optimization_df['energy_savings_kwh'].sum()
        
        print("="*70)
        print("BATTERY MANUFACTURING ENERGY OPTIMIZATION REPORT")
        print("="*70)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Forecast Period: {len(forecast_df)} hours")
        print()
        
        print("ENERGY CONSUMPTION FORECAST:")
        print(f"  Total Forecast Energy: {total_forecast_energy:,.1f} kWh")
        print(f"  Average Hourly Consumption: {total_forecast_energy/len(forecast_df):,.1f} kWh")
        print(f"  Peak Hour Consumption: {forecast_df['forecast_energy_kwh'].max():,.1f} kWh")
        print(f"  Minimum Hour Consumption: {forecast_df['forecast_energy_kwh'].min():,.1f} kWh")
        print()
        
        print("OPTIMIZATION RESULTS:")
        print(f"  Total Optimized Energy: {total_optimized_energy:,.1f} kWh")
        print(f"  Total Energy Savings: {total_savings:,.1f} kWh")
        print(f"  Average Savings per Hour: {total_savings/len(optimization_df):,.1f} kWh")
        print(f"  Overall Savings Percentage: {(total_savings/total_forecast_energy)*100:.1f}%")
        print()
        
        print("MODEL PERFORMANCE:")
        for model_name, metrics in model_results.items():
            print(f"  {model_name.title()}:")
            print(f"    Mean Absolute Error: {metrics['mae']:.2f} kWh")
            print(f"    R² Score: {metrics['r2']:.3f}")
        print()
        
        print("COST SAVINGS ESTIMATION:")
        # Assuming average electricity cost
        cost_per_kwh = 0.12  # $0.12 per kWh
        hourly_savings = total_savings * cost_per_kwh
        daily_savings = hourly_savings * 24
        monthly_savings = daily_savings * 30
        annual_savings = monthly_savings * 12
        
        print(f"  Hourly Cost Savings: ${hourly_savings:,.2f}")
        print(f"  Daily Cost Savings: ${daily_savings:,.2f}")
        print(f"  Monthly Cost Savings: ${monthly_savings:,.2f}")
        print(f"  Annual Cost Savings: ${annual_savings:,.2f}")
        print()
        
        print("OPTIMIZATION STRATEGIES IMPACT:")
        strategy_counts = {}
        for strategies in optimization_df['strategies']:
            for strategy in strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy.replace('_', ' ').title()}: Applied {count} times")
        print()
        
        print("RECOMMENDATIONS:")
        print("  1. Implement real-time temperature optimization system")
        print("  2. Schedule energy-intensive operations during off-peak hours")
        print("  3. Invest in equipment efficiency improvements")
        print("  4. Establish predictive maintenance program")
        print("  5. Consider renewable energy integration for sustainability")
        print()
        
        print("="*70)

def main():
    """Main function to run the energy forecasting system"""
    
    print("Initializing Battery Energy Forecasting System...")
    forecaster = BatteryEnergyForecaster()
    
    # Generate synthetic data
    print("Generating synthetic manufacturing data...")
    df = forecaster.generate_synthetic_data(days=90)
    
    # Create features
    print("Creating features for machine learning...")
    df_features = forecaster.create_features(df)
    
    # Train models
    print("Training forecasting models...")
    model_results, X_test, y_test, X_test_scaled = forecaster.train_models(df_features)
    
    # Generate forecast
    print("Generating energy consumption forecast...")
    forecast_df = forecaster.forecast_energy(df_features, hours_ahead=24)
    
    # Optimize energy consumption
    print("Optimizing energy consumption...")
    optimization_df = forecaster.optimize_energy_consumption(df_features)
    
    # Create visualizations
    print("Creating visualizations...")
    forecaster.plot_results(df, forecast_df, optimization_df)
    
    # Generate comprehensive report
    print("Generating optimization report...")
    forecaster.generate_report(df, forecast_df, optimization_df, model_results)
    
    # Save models and results
    print("Saving models and results...")
    joblib.dump(forecaster.models, 'energy_forecasting_models.pkl')
    forecast_df.to_csv('energy_forecast.csv')
    optimization_df.to_csv('energy_optimization.csv')
    
    print("\nEnergy forecasting and optimization completed successfully!")
    print("Files saved: energy_forecasting_models.pkl, energy_forecast.csv, energy_optimization.csv")

if __name__ == "__main__":
    main()