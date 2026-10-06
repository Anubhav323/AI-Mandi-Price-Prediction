import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="AI Mandi Price Prediction",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Custom styling - FIXED VISIBILITY ISSUES
# =========================================================

st.markdown(
    """
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            color: #185a37;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .section-header {
            font-size: 1.6rem;
            font-weight: 700;
            color: #185a37;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .price-card {
            padding: 1.3rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12);
        }

        .modal-card {
            background: linear-gradient(135deg, #11998e, #38ef7d);
        }

        .minimum-card {
            background: linear-gradient(135deg, #396afc, #2948ff);
        }

        .maximum-card {
            background: linear-gradient(135deg, #f7971e, #ffd200);
        }

        /* FIXED: Better visibility for info boxes */
        .info-box {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #2e7d32;
            color: #1b5e20;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
            margin: 1.5rem 0;
        }

        /* FIXED: Better visibility for disclaimer */
        .disclaimer {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            border-left: 6px solid #f57c00;
            color: #e65100;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
            margin: 1.5rem 0;
        }

        .disclaimer h3 {
            color: #e65100;
            margin-top: 0;
            margin-bottom: 0.8rem;
        }

        /* Additional styling for better text visibility */
        .stAlert {
            border-radius: 10px;
        }

        .stMarkdown p {
            line-height: 1.7;
        }

        /* Better contrast for sidebar */
        .css-1d6mzud {
            background: linear-gradient(180deg, #f5f9f6 0%, #e8f0e9 100%);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Constants
# =========================================================

DATA_FILE = "MP_Mandi_Price_Final_With_Current_Data.csv"

REQUIRED_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "CommodityCode",
    "Variety",
    "Grade",
    "ArrivalDate",
    "MinPrice",
    "MaxPrice",
    "ModalPrice"
]

TEXT_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

NUMERIC_COLUMNS = [
    "CommodityCode",
    "MinPrice",
    "MaxPrice",
    "ModalPrice"
]

# =========================================================
# Utility functions
# =========================================================

def clean_column_names(df):
    """Clean extra spaces and standardize column names."""
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    return df


def convert_numeric(series):
    """Convert numeric values safely."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def validate_dataset(df):
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "The following required columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )
        st.stop()


@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(
            f"Dataset file not found: {file_path}. "
            "Place the CSV file in the same folder as app.py."
        )
        st.stop()

    df = pd.read_csv(file_path)
    df = clean_column_names(df)
    validate_dataset(df)

    # Clean text columns
    for col in TEXT_COLUMNS:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Clean numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = convert_numeric(df[col])

    # Convert date
    df["ArrivalDate"] = pd.to_datetime(
        df["ArrivalDate"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "ArrivalDate",
            "MinPrice",
            "MaxPrice",
            "ModalPrice"
        ]
    ).copy()

    # Remove invalid prices
    df = df[
        (df["ModalPrice"] > 0) &
        (df["MinPrice"] >= 0) &
        (df["MaxPrice"] >= 0)
    ].copy()

    # Remove obvious inconsistent price rows
    df["MinPrice"] = np.minimum(
        df["MinPrice"],
        df["MaxPrice"]
    )

    df["MaxPrice"] = np.maximum(
        df["MinPrice"],
        df["MaxPrice"]
    )

    # Keep the modal price within the observed range where possible
    df["ModalPrice"] = df["ModalPrice"].clip(
        lower=df["MinPrice"],
        upper=df["MaxPrice"]
    )

    # Sort for historical feature creation
    df = df.sort_values(
        [
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade",
            "ArrivalDate"
        ]
    ).reset_index(drop=True)

    return df


def add_date_features(df):
    """Create date features without using future information."""
    df = df.copy()

    df["Year"] = df["ArrivalDate"].dt.year
    df["Month"] = df["ArrivalDate"].dt.month
    df["Day"] = df["ArrivalDate"].dt.day
    df["DayOfYear"] = df["ArrivalDate"].dt.dayofyear
    df["WeekOfYear"] = df["ArrivalDate"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["ArrivalDate"].dt.dayofweek
    df["Quarter"] = df["ArrivalDate"].dt.quarter

    # Cyclic date encoding
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfYearSin"] = np.sin(
        2 * np.pi * df["DayOfYear"] / 365.25
    )
    df["DayOfYearCos"] = np.cos(
        2 * np.pi * df["DayOfYear"] / 365.25
    )

    return df


def add_historical_features(df):
    """
    Create lag and rolling features within each
    commodity-market-variety-grade series.
    """
    df = df.copy()

    group_cols = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade"
    ]

    df = df.sort_values(group_cols + ["ArrivalDate"]).copy()

    grouped = df.groupby(group_cols, sort=False)["ModalPrice"]

    df["PreviousPrice"] = grouped.shift(1)
    df["Previous2Price"] = grouped.shift(2)
    df["RollingMean7"] = grouped.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["RollingMean30"] = grouped.transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )
    df["RollingStd30"] = grouped.transform(
        lambda x: x.shift(1).rolling(30, min_periods=2).std()
    )

    # For series without enough history, use broader historical values
    global_median = df["ModalPrice"].median()

    for col in [
        "PreviousPrice",
        "Previous2Price",
        "RollingMean7",
        "RollingMean30"
    ]:
        df[col] = df[col].fillna(global_median)

    df["RollingStd30"] = df["RollingStd30"].fillna(0)

    return df


def prepare_model_data(raw_df):
    df_model = add_date_features(raw_df)
    df_model = add_historical_features(df_model)

    # Arrival quantity is not present in the supplied dataset.
    # Therefore, the model does not use Arrival as an input.
    categorical_features = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade"
    ]

    numerical_features = [
        "CommodityCode",
        "MinPrice",
        "MaxPrice",
        "Year",
        "Month",
        "Day",
        "DayOfYear",
        "WeekOfYear",
        "DayOfWeek",
        "Quarter",
        "MonthSin",
        "MonthCos",
        "DayOfYearSin",
        "DayOfYearCos",
        "PreviousPrice",
        "Previous2Price",
        "RollingMean7",
        "RollingMean30",
        "RollingStd30"
    ]

    model_features = categorical_features + numerical_features

    df_model = df_model.dropna(
        subset=model_features + ["ModalPrice"]
    ).copy()

    return (
        df_model,
        categorical_features,
        numerical_features,
        model_features
    )


def chronological_split(df_model, test_fraction=0.2):
    """Use the last portion of time for testing."""
    df_model = df_model.sort_values("ArrivalDate").reset_index(drop=True)

    split_index = int(len(df_model) * (1 - test_fraction))

    train_df = df_model.iloc[:split_index].copy()
    test_df = df_model.iloc[split_index:].copy()

    return train_df, test_df


@st.cache_resource
def train_model(df_model, categorical_features, numerical_features):
    train_df, test_df = chronological_split(df_model)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numerical_features
            )
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=250,
        max_depth=22,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    X_train = train_df[
        categorical_features + numerical_features
    ]
    y_train = train_df["ModalPrice"]

    X_test = test_df[
        categorical_features + numerical_features
    ]
    y_test = test_df["ModalPrice"]

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    evaluation_df = test_df[
        [
            "ArrivalDate",
            "Commodity",
            "Market",
            "ModalPrice"
        ]
    ].copy()

    evaluation_df["PredictedPrice"] = predictions
    evaluation_df["Residual"] = (
        evaluation_df["ModalPrice"]
        - evaluation_df["PredictedPrice"]
    )

    return {
        "pipeline": pipeline,
        "train_df": train_df,
        "test_df": test_df,
        "evaluation_df": evaluation_df,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def get_historical_input(
    model_df,
    state,
    district,
    market,
    commodity,
    variety,
    grade,
    prediction_date,
    min_price,
    max_price
):
    """
    Construct one prediction row.

    Historical values are calculated from records before
    the requested prediction date.
    """
    date_value = pd.Timestamp(prediction_date)

    group_mask = (
        (model_df["State"] == state) &
        (model_df["District"] == district) &
        (model_df["Market"] == market) &
        (model_df["Commodity"] == commodity) &
        (model_df["Variety"] == variety) &
        (model_df["Grade"] == grade) &
        (model_df["ArrivalDate"] < date_value)
    )

    history = model_df.loc[group_mask].sort_values("ArrivalDate")

    if len(history) > 0:
        previous_price = history["ModalPrice"].iloc[-1]
    else:
        previous_price = model_df.loc[
            model_df["Commodity"] == commodity,
            "CommodityCode"
        ].median()

    if pd.isna(previous_price):
        previous_price = model_df["ModalPrice"].median()

    previous2_price = (
        history["ModalPrice"].iloc[-2]
        if len(history) >= 2
        else previous_price
    )

    rolling_mean_7 = (
        history["ModalPrice"].tail(7).mean()
        if len(history) > 0
        else previous_price
    )

    rolling_mean_30 = (
        history["ModalPrice"].tail(30).mean()
        if len(history) > 0
        else previous_price
    )

    rolling_std_30 = (
        history["ModalPrice"].tail(30).std()
        if len(history) >= 2
        else 0
    )

    if pd.isna(rolling_std_30):
        rolling_std_30 = 0

    month = date_value.month
    day_of_year = date_value.dayofyear

    row = {
        "State": state,
        "District": district,
        "Market": market,
        "Commodity": commodity,
        "Variety": variety,
        "Grade": grade,
        "CommodityCode": model_df.loc[
            model_df["Commodity"] == commodity,
            "CommodityCode"
        ].median(),
        "MinPrice": float(min_price),
        "MaxPrice": float(max_price),
        "Year": date_value.year,
        "Month": month,
        "Day": date_value.day,
        "DayOfYear": day_of_year,
        "WeekOfYear": int(date_value.isocalendar().week),
        "DayOfWeek": date_value.dayofweek,
        "Quarter": date_value.quarter,
        "MonthSin": np.sin(2 * np.pi * month / 12),
        "MonthCos": np.cos(2 * np.pi * month / 12),
        "DayOfYearSin": np.sin(
            2 * np.pi * day_of_year / 365.25
        ),
        "DayOfYearCos": np.cos(
            2 * np.pi * day_of_year / 365.25
        ),
        "PreviousPrice": previous_price,
        "Previous2Price": previous2_price,
        "RollingMean7": rolling_mean_7,
        "RollingMean30": rolling_mean_30,
        "RollingStd30": rolling_std_30
    }

    return pd.DataFrame([row])


def get_prediction_range(
    model,
    input_row,
    lower_quantile=0.10,
    upper_quantile=0.90
):
    """
    Estimate a prediction interval using individual
    Random Forest tree predictions.
    """
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    transformed = preprocessor.transform(input_row)
    tree_predictions = np.array(
        [
            tree.predict(transformed)[0]
            for tree in regressor.estimators_
        ]
    )

    point_prediction = float(np.mean(tree_predictions))
    lower_prediction = float(
        np.quantile(tree_predictions, lower_quantile)
    )
    upper_prediction = float(
        np.quantile(tree_predictions, upper_quantile)
    )

    return (
        point_prediction,
        lower_prediction,
        upper_prediction
    )


def format_money(value):
    return f"₹{value:,.2f}"


def safe_sample(df, n=10):
    if df.empty:
        return df
    return df.sample(
        min(n, len(df)),
        random_state=42
    )

# =========================================================
# Load and train
# =========================================================

df = load_data(DATA_FILE)

(
    model_df,
    categorical_features,
    numerical_features,
    model_features
) = prepare_model_data(df)

model_result = train_model(
    model_df,
    categorical_features,
    numerical_features
)

model = model_result["pipeline"]
evaluation_df = model_result["evaluation_df"]
mae = model_result["mae"]
rmse = model_result["rmse"]
r2 = model_result["r2"]

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown("## 🏪 AI Mandi Price Predictor")
    st.markdown("---")

    st.info(
        "This system predicts the expected Modal Price "
        "from historical Madhya Pradesh mandi records."
    )

    st.markdown("### Dataset Statistics")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Districts", f"{df['District'].nunique():,}")
    st.metric("Markets", f"{df['Market'].nunique():,}")
    st.metric("Commodities", f"{df['Commodity'].nunique():,}")

    st.markdown("---")
    st.caption(
        "Prediction target: ModalPrice "
        "(₹ per quintal)"
    )

# =========================================================
# Main tabs
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💰 Predict Price",
        "📊 Model Performance",
        "📈 Market Analytics",
        "ℹ️ About"
    ]
)

# =========================================================
# Tab 1: Prediction
# =========================================================

with tab1:
    st.markdown(
        '<h1 class="main-header">'
        '🏪 AI Powered Mandi Price Prediction'
        '</h1>',
        unsafe_allow_html=True
    )

    # FIXED: Better visible info box
    st.markdown(
        """
        <div class="info-box">
        <strong>How to use:</strong> Select a location, commodity, variety, grade and prediction
        date. Enter the expected minimum and maximum market prices
        to generate a Modal Price estimate.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-header">📍 Market Location</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox(
            "State",
            sorted(df["State"].unique())
        )

    with col2:
        district_options = sorted(
            df.loc[
                df["State"] == state,
                "District"
            ].unique()
        )

        district = st.selectbox(
            "District",
            district_options
        )

    with col3:
        market_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["District"] == district),
                "Market"
            ].unique()
        )

        market = st.selectbox(
            "Market",
            market_options
        )

    st.markdown(
        '<div class="section-header">🌾 Commodity Information</div>',
        unsafe_allow_html=True
    )

    col4, col5, col6 = st.columns(3)

    with col4:
        commodity_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["District"] == district) &
                (df["Market"] == market),
                "Commodity"
            ].unique()
        )

        commodity = st.selectbox(
            "Commodity",
            commodity_options
        )

    with col5:
        variety_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["District"] == district) &
                (df["Market"] == market) &
                (df["Commodity"] == commodity),
                "Variety"
            ].unique()
        )

        variety = st.selectbox(
            "Variety",
            variety_options
        )

    with col6:
        grade_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["District"] == district) &
                (df["Market"] == market) &
                (df["Commodity"] == commodity) &
                (df["Variety"] == variety),
                "Grade"
            ].unique()
        )

        grade = st.selectbox(
            "Grade",
            grade_options
        )

    st.markdown(
        '<div class="section-header">📅 Prediction Inputs</div>',
        unsafe_allow_html=True
    )

    col7, col8, col9 = st.columns(3)

    latest_date = df["ArrivalDate"].max().date()
    default_prediction_date = latest_date + timedelta(days=1)

    with col7:
        prediction_date = st.date_input(
            "Prediction Date",
            value=default_prediction_date,
            min_value=df["ArrivalDate"].min().date()
        )

    matching_rows = df[
        (df["State"] == state) &
        (df["District"] == district) &
        (df["Market"] == market) &
        (df["Commodity"] == commodity) &
        (df["Variety"] == variety) &
        (df["Grade"] == grade)
    ]

    if len(matching_rows) > 0:
        default_min = float(matching_rows["MinPrice"].tail(10).median())
        default_max = float(matching_rows["MaxPrice"].tail(10).median())
    else:
        default_min = float(df["MinPrice"].median())
        default_max = float(df["MaxPrice"].median())

    with col8:
        min_price = st.number_input(
            "Expected Minimum Price (₹/quintal)",
            min_value=0.0,
            value=round(default_min, 2),
            step=10.0
        )

    with col9:
        max_price = st.number_input(
            "Expected Maximum Price (₹/quintal)",
            min_value=0.0,
            value=round(max(default_max, default_min), 2),
            step=10.0
        )

    if max_price < min_price:
        st.warning(
            "Maximum price cannot be lower than minimum price."
        )

    predict_button = st.button(
        "🚀 Predict Mandi Price",
        type="primary",
        use_container_width=True
    )

    if predict_button and max_price >= min_price:
        with st.spinner("Analyzing historical mandi trends..."):
            input_row = get_historical_input(
                model_df=model_df,
                state=state,
                district=district,
                market=market,
                commodity=commodity,
                variety=variety,
                grade=grade,
                prediction_date=prediction_date,
                min_price=min_price,
                max_price=max_price
            )

            (
                predicted_price,
                lower_prediction,
                upper_prediction
            ) = get_prediction_range(
                model,
                input_row[
                    categorical_features
                    + numerical_features
                ]
            )

            # Keep range sensible
            lower_prediction = max(
                0,
                min(lower_prediction, predicted_price)
            )
            upper_prediction = max(
                upper_prediction,
                predicted_price
            )

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.markdown(
                    f"""
                    <div class="price-card modal-card">
                        <h3>Predicted Modal Price</h3>
                        <h2>{format_money(predicted_price)}</h2>
                        <p>per quintal</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div class="price-card minimum-card">
                        <h3>Estimated Lower Range</h3>
                        <h2>{format_money(lower_prediction)}</h2>
                        <p>model tree range</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col3:
                st.markdown(
                    f"""
                    <div class="price-card maximum-card">
                        <h3>Estimated Upper Range</h3>
                        <h2>{format_money(upper_prediction)}</h2>
                        <p>model tree range</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.caption(
                "The range is estimated from the variation among "
                "Random Forest trees. It is not a guaranteed confidence "
                "interval."
            )

            # Historical trend
            st.subheader(
                f"📈 Historical Trend: {commodity}"
            )

            trend_filter = (
                (df["State"] == state) &
                (df["District"] == district) &
                (df["Market"] == market) &
                (df["Commodity"] == commodity)
            )

            trend_df = df.loc[
                trend_filter,
                ["ArrivalDate", "ModalPrice"]
            ].copy()

            if not trend_df.empty:
                trend_df = (
                    trend_df
                    .groupby("ArrivalDate", as_index=False)
                    ["ModalPrice"]
                    .mean()
                    .sort_values("ArrivalDate")
                )

                fig_trend = px.line(
                    trend_df,
                    x="ArrivalDate",
                    y="ModalPrice",
                    markers=True,
                    title=(
                        f"{commodity} Modal Price Trend "
                        f"at {market}"
                    ),
                    labels={
                        "ArrivalDate": "Arrival Date",
                        "ModalPrice": "Modal Price (₹/quintal)"
                    }
                )

                fig_trend.add_hline(
                    y=predicted_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=(
                        f"Predicted: "
                        f"{format_money(predicted_price)}"
                    )
                )

                fig_trend.update_layout(height=450)
                st.plotly_chart(
                    fig_trend,
                    use_container_width=True
                )
            else:
                st.info(
                    "No historical trend is available for this selection."
                )

            st.subheader("📋 Similar Historical Records")

            similar_df = df[
                (df["State"] == state) &
                (df["District"] == district) &
                (df["Market"] == market) &
                (df["Commodity"] == commodity) &
                (df["Variety"] == variety) &
                (df["Grade"] == grade)
            ].sort_values(
                "ArrivalDate",
                ascending=False
            ).head(10)

            if not similar_df.empty:
                display_columns = [
                    "ArrivalDate",
                    "Commodity",
                    "Variety",
                    "Grade",
                    "MinPrice",
                    "MaxPrice",
                    "ModalPrice"
                ]

                st.dataframe(
                    similar_df[display_columns].style.format(
                        {
                            "MinPrice": "₹{:,.2f}",
                            "MaxPrice": "₹{:,.2f}",
                            "ModalPrice": "₹{:,.2f}"
                        }
                    ),
                    use_container_width=True
                )
            else:
                st.info(
                    "No similar historical records were found."
                )

# =========================================================
# Tab 2: Model Performance
# =========================================================

with tab2:
    st.markdown(
        '<h1 class="main-header">'
        '📊 Model Performance Dashboard'
        '</h1>',
        unsafe_allow_html=True
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    with metric_col2:
        st.metric(
            "MAE",
            format_money(mae)
        )

    with metric_col3:
        st.metric(
            "RMSE",
            format_money(rmse)
        )

    with metric_col4:
        st.metric(
            "Training Rows",
            f"{len(model_result['train_df']):,}"
        )

    st.info(
        "The test set consists of the latest 20% of records by date. "
        "This is more appropriate for price forecasting than a purely "
        "random split, although it does not guarantee future accuracy."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🎯 Actual vs Predicted Modal Price")

        fig_actual = px.scatter(
            evaluation_df,
            x="ModalPrice",
            y="PredictedPrice",
            color="Commodity",
            hover_data=[
                "ArrivalDate",
                "Market"
            ],
            title="Actual Price vs Predicted Price",
            labels={
                "ModalPrice": "Actual Modal Price (₹)",
                "PredictedPrice": "Predicted Modal Price (₹)"
            },
            opacity=0.65
        )

        min_axis = min(
            evaluation_df["ModalPrice"].min(),
            evaluation_df["PredictedPrice"].min()
        )

        max_axis = max(
            evaluation_df["ModalPrice"].max(),
            evaluation_df["PredictedPrice"].max()
        )

        fig_actual.add_trace(
            go.Scatter(
                x=[min_axis, max_axis],
                y=[min_axis, max_axis],
                mode="lines",
                name="Perfect Prediction",
                line=dict(
                    color="red",
                    dash="dash"
                )
            )
        )

        fig_actual.update_layout(height=480)
        st.plotly_chart(
            fig_actual,
            use_container_width=True
        )

    with col_b:
        st.subheader("📉 Residual Distribution")

        fig_residual = px.histogram(
            evaluation_df,
            x="Residual",
            nbins=40,
            title="Prediction Residuals",
            labels={
                "Residual": "Actual - Predicted Price (₹)"
            },
            color_discrete_sequence=["#2e8b57"]
        )

        fig_residual.add_vline(
            x=0,
            line_dash="dash",
            line_color="red"
        )

        fig_residual.update_layout(height=480)
        st.plotly_chart(
            fig_residual,
            use_container_width=True
        )

    st.subheader("📊 Residuals vs Predicted Price")

    fig_residual_scatter = px.scatter(
        evaluation_df,
        x="PredictedPrice",
        y="Residual",
        color="Commodity",
        hover_data=["ArrivalDate", "Market"],
        title="Residuals vs Predicted Price",
        labels={
            "PredictedPrice": "Predicted Price (₹)",
            "Residual": "Residual (₹)"
        },
        opacity=0.65
    )

    fig_residual_scatter.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    fig_residual_scatter.update_layout(height=450)
    st.plotly_chart(
        fig_residual_scatter,
        use_container_width=True
    )

    st.subheader("📋 Evaluation Sample")

    st.dataframe(
        evaluation_df.sort_values(
            "ArrivalDate",
            ascending=False
        ).head(100).style.format(
            {
                "ModalPrice": "₹{:,.2f}",
                "PredictedPrice": "₹{:,.2f}",
                "Residual": "₹{:,.2f}"
            }
        ),
        use_container_width=True
    )

# =========================================================
# Tab 3: Market Analytics
# =========================================================

with tab3:
    st.markdown(
        '<h1 class="main-header">'
        '📈 Mandi Market Analytics'
        '</h1>',
        unsafe_allow_html=True
    )

    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)

    with analytics_col1:
        st.metric(
            "Average Modal Price",
            format_money(df["ModalPrice"].mean())
        )

    with analytics_col2:
        st.metric(
            "Highest Modal Price",
            format_money(df["ModalPrice"].max())
        )

    with analytics_col3:
        st.metric(
            "Lowest Modal Price",
            format_money(df["ModalPrice"].min())
        )

    with analytics_col4:
        st.metric(
            "Latest Data Date",
            str(df["ArrivalDate"].max().date())
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("🌾 Average Price by Commodity")

        commodity_prices = (
            df.groupby("Commodity", as_index=False)
            ["ModalPrice"]
            .mean()
            .sort_values("ModalPrice", ascending=False)
            .head(20)
        )

        fig_commodity = px.bar(
            commodity_prices,
            x="ModalPrice",
            y="Commodity",
            orientation="h",
            color="ModalPrice",
            color_continuous_scale="Viridis",
            title="Top Commodities by Average Modal Price",
            labels={
                "ModalPrice": "Average Modal Price (₹/quintal)"
            }
        )

        fig_commodity.update_layout(
            height=550,
            showlegend=False
        )

        st.plotly_chart(
            fig_commodity,
            use_container_width=True
        )

    with col_d:
        st.subheader("📍 Average Price by District")

        district_prices = (
            df.groupby("District", as_index=False)
            ["ModalPrice"]
            .mean()
            .sort_values("ModalPrice", ascending=False)
        )

        fig_district = px.bar(
            district_prices,
            x="District",
            y="ModalPrice",
            color="ModalPrice",
            color_continuous_scale="Plasma",
            title="Average Modal Price by District",
            labels={
                "ModalPrice": "Average Modal Price (₹/quintal)"
            }
        )

        fig_district.update_layout(
            height=550,
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(
            fig_district,
            use_container_width=True
        )

    st.subheader("📅 Monthly Market Price Trend")

    monthly_df = df.copy()
    monthly_df["Month"] = (
        monthly_df["ArrivalDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_prices = (
        monthly_df.groupby("Month", as_index=False)
        ["ModalPrice"]
        .mean()
    )

    fig_monthly = px.line(
        monthly_prices,
        x="Month",
        y="ModalPrice",
        markers=True,
        title="Overall Monthly Average Modal Price",
        labels={
            "Month": "Month",
            "ModalPrice": "Average Modal Price (₹/quintal)"
        }
    )

    fig_monthly.update_layout(height=450)
    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    st.subheader("🔥 Commodity Price Heatmap")

    selected_commodities = st.multiselect(
        "Select commodities for heatmap",
        options=sorted(df["Commodity"].unique()),
        default=sorted(
            df["Commodity"].value_counts()
            .head(12)
            .index
            .tolist()
        )
    )

    if selected_commodities:
        heatmap_df = df[
            df["Commodity"].isin(selected_commodities)
        ]

        pivot_df = heatmap_df.pivot_table(
            index="District",
            columns="Commodity",
            values="ModalPrice",
            aggfunc="mean"
        )

        fig_heatmap = px.imshow(
            pivot_df,
            aspect="auto",
            color_continuous_scale="RdYlGn",
            text_auto=".0f",
            title="Average Modal Price by District and Commodity"
        )

        fig_heatmap.update_layout(height=650)
        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

    st.subheader("📊 Price Range Analysis")

    range_df = df.copy()
    range_df["PriceSpread"] = (
        range_df["MaxPrice"]
        - range_df["MinPrice"]
    )

    range_summary = (
        range_df.groupby("Commodity", as_index=False)
        .agg(
            AverageMinPrice=("MinPrice", "mean"),
            AverageModalPrice=("ModalPrice", "mean"),
            AverageMaxPrice=("MaxPrice", "mean"),
            AverageSpread=("PriceSpread", "mean"),
            Records=("ModalPrice", "count")
        )
        .sort_values(
            "AverageModalPrice",
            ascending=False
        )
        .head(30)
    )

    st.dataframe(
        range_summary.style.format(
            {
                "AverageMinPrice": "₹{:,.2f}",
                "AverageModalPrice": "₹{:,.2f}",
                "AverageMaxPrice": "₹{:,.2f}",
                "AverageSpread": "₹{:,.2f}",
                "Records": "{:,.0f}"
            }
        ),
        use_container_width=True
    )

# =========================================================
# Tab 4: About
# =========================================================

with tab4:
    st.markdown(
        '<h1 class="main-header">'
        'ℹ️ About the System'
        '</h1>',
        unsafe_allow_html=True
    )

    about_col1, about_col2 = st.columns(2)

    with about_col1:
        st.markdown(
            """
            ## 🏪 AI Powered Mandi Price Prediction

            This application estimates the expected **Modal Price**
            for commodities traded in Madhya Pradesh mandis.

            ### Dataset Fields Used

            - State
            - District
            - Market
            - Commodity
            - Commodity Code
            - Variety
            - Grade
            - Arrival Date
            - Minimum Price
            - Maximum Price
            - Modal Price

            ### Model Inputs

            The model uses:

            - Market location
            - Commodity and variety
            - Grade
            - Minimum and maximum price
            - Date-based seasonal features
            - Historical price lags
            - Seven-day rolling average
            - Thirty-day rolling average
            - Thirty-day price volatility
            """
        )

    with about_col2:
        st.markdown(
            """
            ## 🤖 Machine Learning Method

            **Algorithm:** Random Forest Regressor

            The categorical variables are transformed with
            one-hot encoding. Date fields are converted into
            seasonal and calendar features.

            The test set contains the latest 20% of observations by
            date. This gives a more realistic estimate of forecasting
            performance than randomly mixing past and future records.

            ### Important Dataset Limitation

            The supplied dataset does not contain an explicit arrival
            quantity field. It contains an `ArrivalDate` column, which is
            the market arrival date. Therefore, arrival quantity is not
            used as a model input.

            If you later add an arrival quantity column, it can be
            incorporated as an additional feature.
            """
        )

    # FIXED: Better visible disclaimer
    st.markdown(
        """
        <div class="disclaimer">
        <h3>⚠️ Disclaimer</h3>
        <p>This application provides statistical estimates from historical
        mandi data. Actual prices may change because of supply, demand,
        weather, quality, transportation, government policy, seasonal
        arrivals and local market conditions. Do not treat the prediction
        as a guaranteed price or financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📥 Download Cleaned Dataset")

    download_df = df.copy()
    download_df["ArrivalDate"] = (
        download_df["ArrivalDate"].dt.strftime("%Y-%m-%d")
    )

    st.download_button(
        label="Download Cleaned CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_mandi_prices.csv",
        mime="text/csv"
    )

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#666; padding:1rem;">
        🏪 AI Powered Mandi Price Prediction System |
        Madhya Pradesh Mandi Dataset |
        Built with Streamlit, Pandas, Scikit-learn and Plotly
    </div>
    """,
    unsafe_allow_html=True
)