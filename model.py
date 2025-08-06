import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils import rmse, mape

VALID_METRICS = ["confirmed","deaths","recovered","active"]

class CovidForecaster:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, parse_dates=["date"])
        self.df = self.df.sort_values("date")
        self.countries = sorted(self.df["country"].dropna().unique().tolist())
        self.regions = sorted(self.df["who_region"].dropna().unique().tolist())

    def get_meta(self):
        return {"countries": self.countries, "regions": self.regions, "metrics": VALID_METRICS}

    def _get_series(self, level, name, metric):
        if level == "country":
            sub = self.df[self.df["country"] == name]
        else:
            sub = self.df[self.df["who_region"] == name]
        sub = sub[["date", metric]].set_index("date").asfreq("D")
        sub[metric] = sub[metric].interpolate(limit_direction="both")
        return sub[metric]

    def forecast(self, level, name, model_type="arima", steps=14, metric="confirmed"):
        series = self._get_series(level, name, metric)
        train_size = int(len(series)*0.85)
        train, test = series.iloc[:train_size], series.iloc[train_size:]

        order = (2,1,2)
        seasonal_order = (1,0,1,7)

        if model_type=="arima":
            fit = ARIMA(train, order=order).fit()
            fc_test = fit.forecast(steps=len(test))
        else:
            fit = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc_test = fit.forecast(steps=len(test))

        metrics = {"rmse": rmse(test.values, fc_test.values), "mape": mape(test.values, fc_test.values)}
        # forecast future
        if model_type=="arima":
            full_fit = ARIMA(series, order=order).fit()
            future_fc = full_fit.forecast(steps=steps)
        else:
            full_fit = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            future_fc = full_fit.forecast(steps=steps)
        future_index = pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=steps, freq="D")
        return {
            "history_dates": series.index.strftime("%Y-%m-%d").tolist(),
            "history_values": series.values.tolist(),
            "forecast_dates": future_index.strftime("%Y-%m-%d").tolist(),
            "forecast_values": [float(v) for v in future_fc.values],
            "metrics": metrics,
            "model": model_type,
            "metric": metric,
            "level": level,
            "name": name
        }
