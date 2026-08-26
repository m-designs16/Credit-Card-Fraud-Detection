import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------- DATA LOAD ----------------

credit_card_data = pd.read_csv("creditcard.csv")

# ---------------- DATA PREPARATION ----------------

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# ---------------- TRAIN TEST SPLIT ----------------

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# ---------------- MODEL ----------------

model = LogisticRegression()

model.fit(x_train, y_train)

# ---------------- TITLE ----------------

st.title("💳 Credit Card Fraud Detection")

st.write("Check whether a transaction is Fraudulent or Legitimate")

st.markdown("---")

# ---------------- INPUT ----------------

transaction_no = st.number_input(
    "Enter Transaction Number",
    min_value=1,
    max_value=len(x_test),
    step=1
)

# ---------------- BUTTON ----------------

if st.button("Check Transaction"):

    index = transaction_no - 1

    input_data = x_test.iloc[index].values.reshape(1, -1)

    prediction = model.predict(input_data)

    st.subheader("Result")

    if prediction[0] == 0:

        st.success("✅ Legitimate Transaction")

    else:

        st.error("🚨 Fraudulent Transaction")

    # Actual Result

    actual = y_test.iloc[index]

    if actual == 0:
        st.write("Actual Value: Legitimate")
    else:
        st.write("Actual Value: Fraud")

    # Transaction Details

    st.subheader("Transaction Details")

    st.dataframe(x_test.iloc[[index]])