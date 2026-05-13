import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="KNN ML App", layout="centered")

st.title("KNN Machine Learning Application")
st.write("This app demonstrates both KNN Classification and KNN Regression.")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Model Type",
    ("KNN Classification", "KNN Regression")
)

# ==============================
# KNN CLASSIFICATION
# ==============================
if option == "KNN Classification":

    st.header("KNN Classification")

    uploaded_file = st.file_uploader(
        "Upload CSV file for Classification",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # OPTIONAL: remove ID columns
        df = df.loc[:, ~df.columns.str.contains('id', case=False)]

        st.subheader("Dataset Preview")
        st.write(df.head())

        # ✅ SMART TARGET SELECTION (CLASSIFICATION)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            st.error("No categorical columns found for classification target!")
        else:
            target_column = st.selectbox(
                "Select Target Column",
                categorical_cols
            )

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            k_value = st.slider("Select K Value", 1, 15, 5)

            model = KNeighborsClassifier(n_neighbors=k_value)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.success(f"Classification Accuracy: {accuracy:.2f}")

# ==============================
# KNN REGRESSION
# ==============================
elif option == "KNN Regression":

    st.header("KNN Regression")

    uploaded_file = st.file_uploader(
        "Upload CSV file for Regression",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # OPTIONAL: remove ID columns
        df = df.loc[:, ~df.columns.str.contains('id', case=False)]

        st.subheader("Dataset Preview")
        st.write(df.head())

        # ✅ SMART TARGET SELECTION (REGRESSION)
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            st.error("No numeric columns found for regression target!")
        else:
            target_column = st.selectbox(
                "Select Target Column",
                numeric_cols
            )

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            k_value = st.slider("Select K Value", 1, 15, 5)

            model = KNeighborsRegressor(n_neighbors=k_value)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            st.success(f"R2 Score: {r2:.2f}")