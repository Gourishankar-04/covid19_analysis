# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from datetime import datetime
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import numpy as np

# app = Flask(__name__)
# CORS(app)

# # Load and clean dataset
# df = pd.read_csv('data/covid_19_clean_complete.csv')

# df['Date'] = pd.to_datetime(df['Date'])

# # Ensure all necessary columns exist
# assert 'Country/Region' in df.columns and 'WHO Region' in df.columns

# AVAILABLE_METRICS = ['Confirmed', 'Deaths', 'Recovered', 'Active']
# AVAILABLE_MODELS = ['arima', 'sarimax']

# @app.route('/meta', methods=['GET'])
# def meta():
#     countries = sorted(df['Country/Region'].dropna().unique().tolist())
#     regions = sorted(df['WHO Region'].dropna().unique().tolist())
#     metrics = [m.lower() for m in AVAILABLE_METRICS]
#     return jsonify({'countries': countries, 'regions': regions, 'metrics': metrics})

# @app.route('/forecast', methods=['POST'])
# def forecast():
#     data = request.get_json()

#     level = data.get('level')  # 'country' or 'region'
#     name = data.get('name')    # e.g. 'India' or 'EURO'
#     model_type = data.get('model', 'arima')  # 'arima' or 'sarimax'
#     steps = int(data.get('steps', 14))
#     metric = data.get('metric', 'confirmed').capitalize()  # match CSV column

#     if metric not in AVAILABLE_METRICS:
#         return jsonify({'error': f'Metric {metric} not supported'}), 400

#     if level == 'country':
#         subset = df[df['Country/Region'] == name]
#     elif level == 'region':
#         subset = df[df['WHO Region'] == name]
#     else:
#         return jsonify({'error': 'Invalid level'}), 400

#     if subset.empty:
#         return jsonify({'error': f'No data found for {name}'}), 404

#     ts = subset.groupby('Date')[metric].sum().reset_index()
#     ts = ts.set_index('Date')
#     y = ts[metric]

#     if len(y) < 10:
#         return jsonify({'error': 'Not enough data for forecasting'}), 400

#     # Save history
#     history_dates = [d.strftime("%Y-%m-%d") for d in y.index]
#     history_values = y.values.tolist()

#     try:
#         if model_type == 'arima':
#             model = ARIMA(y, order=(2, 1, 2))
#         elif model_type == 'sarimax':
#             model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
#         else:
#             return jsonify({'error': 'Invalid model type'}), 400

#         model_fit = model.fit()
#         forecast = model_fit.forecast(steps=steps)

#         forecast_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=steps)
#         forecast_values = forecast.values.tolist()

#         # Evaluate last 10 points for RMSE/MAPE
#         y_pred = model_fit.fittedvalues[-10:]
#         y_true = y[-10:]

#         rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
#         mape = (np.abs((y_true - y_pred) / y_true).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

#         return jsonify({
#             'status': 'success',
#             'name': name,
#             'level': level,
#             'metric': metric.lower(),
#             'model': model_type,
#             'history_dates': history_dates,
#             'history_values': history_values,
#             'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
#             'forecast_values': forecast_values,
#             'metrics': {
#                 'rmse': rmse,
#                 'mape': mape
#             }
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from Vercel frontend

# Load and clean dataset
df = pd.read_csv('data/covid_19_clean_complete.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Ensure all necessary columns exist
assert 'Country/Region' in df.columns and 'WHO Region' in df.columns

AVAILABLE_METRICS = ['Confirmed', 'Deaths', 'Recovered', 'Active']
AVAILABLE_MODELS = ['arima', 'sarimax']

@app.route('/meta', methods=['GET'])
def meta():
    countries = sorted(df['Country/Region'].dropna().unique().tolist())
    regions = sorted(df['WHO Region'].dropna().unique().tolist())
    metrics = [m.lower() for m in AVAILABLE_METRICS]
    return jsonify({'countries': countries, 'regions': regions, 'metrics': metrics})

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()

    level = data.get('level')  # 'country' or 'region'
    name = data.get('name')    # e.g. 'India' or 'EURO'
    model_type = data.get('model', 'arima')  # 'arima' or 'sarimax'
    steps = int(data.get('steps', 14))
    metric = data.get('metric', 'confirmed').capitalize()  # match CSV column

    if metric not in AVAILABLE_METRICS:
        return jsonify({'error': f'Metric {metric} not supported'}), 400

    if level == 'country':
        subset = df[df['Country/Region'] == name]
    elif level == 'region':
        subset = df[df['WHO Region'] == name]
    else:
        return jsonify({'error': 'Invalid level'}), 400

    if subset.empty:
        return jsonify({'error': f'No data found for {name}'}), 404

    ts = subset.groupby('Date')[metric].sum().reset_index()
    ts = ts.set_index('Date')
    y = ts[metric]

    if len(y) < 10:
        return jsonify({'error': 'Not enough data for forecasting'}), 400

    # Save history
    history_dates = [d.strftime("%Y-%m-%d") for d in y.index]
    history_values = y.values.tolist()

    try:
        if model_type == 'arima':
            model = ARIMA(y, order=(2, 1, 2))
        elif model_type == 'sarimax':
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)

        forecast_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=steps)
        forecast_values = forecast.values.tolist()

        # Evaluate last 10 points for RMSE/MAPE
        y_pred = model_fit.fittedvalues[-10:]
        y_true = y[-10:]

        rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
        mape = (np.abs((y_true - y_pred) / y_true).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100

        return jsonify({
            'status': 'success',
            'name': name,
            'level': level,
            'metric': metric.lower(),
            'model': model_type,
            'history_dates': history_dates,
            'history_values': history_values,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'forecast_values': forecast_values,
            'metrics': {
                'rmse': rmse,
                'mape': mape
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
