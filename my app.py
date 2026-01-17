import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA   # 👈 add this import
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.title("Inventory Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload weekly CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df)

    sku = st.selectbox("Select SKU", df["sku"].unique())    
    sku_df = df[df["sku"] == sku].sort_values("week_start_date")
    sku_df["week_start_date"] = pd.to_datetime(
    sku_df["week_start_date"],
    dayfirst=True
    )
    current_stock = st.number_input(
    "Enter Current Stock (Units)",
    min_value=0,
    value=0,
    step=10
    )
    min_safety_stock = st.number_input(
    "Minimum Safety Stock (Units)",
    min_value=0,
    value=30,   # default 30
    step=5
    )
    st.line_chart(
        sku_df.set_index("week_start_date")["weekly_sales"]
    )

    st.subheader("ARIMA Forecast (Next 12 Weeks)")

    if len(sku_df) >= 10:

        sales_series = sku_df.set_index("week_start_date")["weekly_sales"]

        model = ARIMA(sales_series, order=(1,1,1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=12)

        forecast_dates = pd.date_range(
            start=sales_series.index[-1] + pd.Timedelta(weeks=1),
            periods=12,
            freq="W-MON"
        )

        forecast_df = pd.DataFrame({
            "week_start_date": forecast_dates,
            "forecasted_sales": forecast.values
        })

        st.dataframe(forecast_df)
       # =======================
       # INVENTORY PLANNING
       # =======================

        service_level_z = 1.65  # 95% service level

        total_demand = forecast_df["forecasted_sales"].sum()

        #Safety Stock
        z_score = 1.65  # 95% service level
        calculated_safety_stock = int(z_score * np.std(total_demand))
        final_safety_stock = max(calculated_safety_stock, min_safety_stock)


        net_inventory_required = total_demand + final_safety_stock - current_stock

        if net_inventory_required < 0:
            net_inventory_required = 0
        st.subheader("Inventory Planning Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Forecast Demand (3 Months)",
            f"{int(total_demand)} units"
            )
        col2.metric(
            "Safety Stock Required",
            f"{int(final_safety_stock)} units"
            )
        col3.metric(
            "Net Inventory to Order",
            f"{int(net_inventory_required)} units"
            )

        combined_df = pd.concat([
            sales_series,
            forecast_df.set_index("week_start_date")["forecasted_sales"]
        ])

        st.line_chart(combined_df)

    # =======================
    # MODEL VALIDATION
    # =======================

    st.subheader("Model Validation (Backtesting)")

    if len(sales_series) > 8:  # safety check

        train = sales_series[:-4]
        test = sales_series[-4:]

        model_bt = ARIMA(train, order=(1,1,1))
        model_bt_fit = model_bt.fit()

        test_forecast = model_bt_fit.forecast(steps=4)

        mae = mean_absolute_error(test, test_forecast)
        rmse = np.sqrt(mean_squared_error(test, test_forecast))

        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")

        validation_df = pd.DataFrame({
            "Actual Sales": test.values,
            "Forecasted Sales": test_forecast.values
        }, index=test.index)

        st.line_chart(validation_df)


    else:
        st.warning("Not enough data to train ARIMA (minimum 10 weeks required).")
