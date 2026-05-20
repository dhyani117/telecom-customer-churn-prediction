from flask import Flask, render_template, request
import pickle
import pandas as pd
import os


# ===============================
# Path configuration (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
) 

MODEL_DIR = os.path.join(BASE_DIR, "models")

# ===============================
# Load trained models & objects
# ===============================
with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb") as f:
    model_random = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODEL_DIR, "standard_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    international_plan = request.form["international_plan"]
    voice_mail_plan = request.form["voice_mail_plan"]

    account_length = int(request.form["account_length"])
    number_vmail_messages = float(request.form["number_vmail_messages"])
    total_day_minutes = float(request.form["total_day_minutes"])
    total_day_calls = float(request.form["total_day_calls"])
    total_eve_minutes = float(request.form["total_eve_minutes"])
    total_eve_calls = float(request.form["total_eve_calls"])
    total_night_minutes = float(request.form["total_night_minutes"])
    total_night_calls = float(request.form["total_night_calls"])
    total_intl_minutes = float(request.form["total_intl_minutes"])
    total_intl_calls = float(request.form["total_intl_calls"])
    customer_service_calls = float(request.form["customer_service_calls"])

    # ===============================
    # Encode categorical values
    # ===============================
    international_plan = label_encoders["International plan"].transform(
        [international_plan]
    )[0]

    voice_mail_plan = label_encoders["Voice mail plan"].transform(
        [voice_mail_plan]
    )[0]

    # ===============================
    # Create input DataFrame
    # ===============================
    input_df = pd.DataFrame({
        "International plan": [international_plan],
        "Voice mail plan": [voice_mail_plan],
        "Account length": [account_length],
        "Number vmail messages": [number_vmail_messages],
        "Total day minutes": [total_day_minutes],
        "Total day calls": [total_day_calls],
        "Total eve minutes": [total_eve_minutes],
        "Total eve calls": [total_eve_calls],
        "Total night minutes": [total_night_minutes],
        "Total night calls": [total_night_calls],
        "Total intl minutes": [total_intl_minutes],
        "Total intl calls": [total_intl_calls],
        "Customer service calls": [customer_service_calls]
    })

    # ===============================
    # Scale numerical columns
    # ===============================
    numerical_cols = [
        "Account length",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total eve minutes",
        "Total eve calls",
        "Total night minutes",
        "Total night calls",
        "Total intl minutes",
        "Total intl calls",
        "Customer service calls"
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # ===============================
    # Prediction
    # ===============================
    prediction = model_random.predict(input_df)[0]

    result = "Churn" if prediction == 1 else "No Churn"

    return render_template("index.html", prediction=result)


# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)