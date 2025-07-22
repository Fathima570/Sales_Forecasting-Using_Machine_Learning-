import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, and feature columns
model = pickle.load(open("lr_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Constants from Colab for preprocessing
ITEM_VISIBILITY_MEAN = 0.06613202877895127
IQR_BOUNDS = {
    'Item_Weight': (4.28725, 16.71275),
    'Item_Visibility': (-0.0520, 0.2039),
    'Item_MRP': (93.8265, 310.8625)
}

# IQR clipping
def clip_iqr(val, lower, upper):
    return max(min(val, upper), lower)

st.title("📦 Sales Forecasting")
st.markdown("### Enter the product and outlet details below:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        item_weight = st.number_input("Item Weight", value=1.5, step=0.1)
        item_visibility = st.number_input("Item Visibility", value=0.05, step=0.01)
        item_mrp = st.number_input("Item MRP", value=200.0, step=1.0)

    with col2:
        item_fat = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
        outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
        outlet_loc = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox("Outlet Type", [
            'Supermarket Type1',
            'Supermarket Type2',
            'Supermarket Type3',
            'Grocery Store'
        ])
        new_item_type = st.selectbox("New Item Type", ['Food', 'Drinks', 'Non-Consumables'])

    submit = st.form_submit_button("Predict Sales")

if submit:
    # Replace 0 visibility with mean
    if item_visibility == 0:
        item_visibility = ITEM_VISIBILITY_MEAN

    # Apply outlier clipping
    item_weight = clip_iqr(item_weight, *IQR_BOUNDS['Item_Weight'])
    item_visibility = clip_iqr(item_visibility, *IQR_BOUNDS['Item_Visibility'])
    item_mrp = clip_iqr(item_mrp, *IQR_BOUNDS['Item_MRP'])

    # Construct input dict
    input_data = {
        'Item_Weight': item_weight,
        'Item_Visibility': item_visibility,
        'Item_MRP': item_mrp,
        f'Item_Fat_Content_{item_fat}': 1,
        f'Outlet_Size_{outlet_size}': 1,
        f'Outlet_Location_Type_{outlet_loc}': 1,
        f'Outlet_Type_{outlet_type}': 1,
        f'New_Item_Type_{new_item_type}': 1
    }

    # Fill in missing dummy columns with 0s
    full_input = pd.DataFrame([input_data])
    full_input = full_input.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns
    if hasattr(scaler, 'feature_names_in_'):
        num_cols = list(scaler.feature_names_in_)
    else:
        num_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
    full_input[num_cols] = scaler.transform(full_input[num_cols])

    # Predict
    prediction = model.predict(full_input)[0]

    st.success(f"✅ Predicted Item Outlet Sales: ₹ {round(prediction, 2)}")
