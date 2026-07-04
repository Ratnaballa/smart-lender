import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from src.ml_pipeline import (
    append_history,
    build_prediction_payload,
    load_history,
    load_model,
    prepare_feature_row,
    save_history,
    validate_input,
)

app = Flask(__name__)
app.secret_key = "smart-lender-secret-key"

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

try:
    model = load_model(MODEL_PATH)
except (FileNotFoundError, AttributeError):
    model = None

MODEL_CARDS = [
    {"name": "Logistic Regression", "description": "Fast, interpretable classification for eligibility scoring."},
    {"name": "Random Forest", "description": "Robust performance with nonlinear feature interactions."},
    {"name": "Gradient Boosting", "description": "High-accuracy predictions from ensemble learning."},
    {"name": "XGBoost", "description": "Advanced model for reliable loan decision support."},
]

DATA_FIELDS = [
    {"name": "Gender", "type": "Categorical"},
    {"name": "Married", "type": "Categorical"},
    {"name": "Dependents", "type": "Numeric"},
    {"name": "Education", "type": "Categorical"},
    {"name": "Self_Employed", "type": "Categorical"},
    {"name": "ApplicantIncome", "type": "Numeric"},
    {"name": "CoapplicantIncome", "type": "Numeric"},
    {"name": "LoanAmount", "type": "Numeric"},
    {"name": "Loan_Amount_Term", "type": "Numeric"},
    {"name": "Credit_History", "type": "Categorical"},
    {"name": "Property_Area", "type": "Categorical"},
]

FEATURES = [
    {"title": "Fast eligibility checks", "description": "Instant AI-powered scoring for every loan application."},
    {"title": "Smart borrower insights", "description": "Understand borrower capacity with predictive analytics."},
    {"title": "Compliance-ready UI", "description": "Professional workflows built for banking review."},
    {"title": "Historical tracking", "description": "Review past decisions and analyze loan conversion trends."},
]

TECH_STACK = [
    "Python & Flask",
    "Bootstrap 5",
    "JavaScript",
    "HTML5 & CSS3",
    "Google Fonts",
    "Bootstrap Icons",
    "Session-based history",
]

STATISTICS = [
    {"value": "96%", "label": "Prediction accuracy "},
    {"value": "24k+", "label": "Monthly assessments"},
    {"value": "8x", "label": "Faster decision cycles"},
    {"value": "120+", "label": "AI-driven rules"},
]

WORKFLOW_STEPS = [
    {"step": "01", "title": "Submit borrower profile", "detail": "Provide applicant and loan information through structured forms."},
    {"step": "02", "title": "AI evaluation", "detail": "The prediction engine analyzes credit, income, and collateral attributes."},
    {"step": "03", "title": "Decision summary", "detail": "Review approval outcome with confidence scores and recommendations."},
]

CONTACT_CARDS = [
    {"title": "Customer Success", "name": "Ayesha Singh", "email": "ayesha@smartlender.ai", "phone": "+91 98765 43210"},
    {"title": "Product Inquiry", "name": "Rohit Mehra", "email": "rohit@smartlender.ai", "phone": "+91 91234 56780"},
]


def initialize_history():
    if "history" not in session:
        session["history"] = []


def generate_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise RuntimeError("Model is not available. Train the model first.")
    feature_frame = prepare_feature_row(data)
    prediction = int(model.predict(feature_frame)[0])
    probability = float(model.predict_proba(feature_frame)[0][prediction])
    return build_prediction_payload(data, prediction, probability)


@app.route("/")
def home():
    initialize_history()
    return render_template(
        "home.html",
        active_page="home",
        statistics=STATISTICS,
        features=FEATURES,
        workflow=WORKFLOW_STEPS,
        models=MODEL_CARDS,
        stack=TECH_STACK,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    initialize_history()
    if request.method == "POST":
        form = request.form
        dependents = form.get("dependents", "0")
        if dependents == "3+":
            dependents = 3
        data = {
            "Gender": form.get("gender", "Male"),
            "Married": form.get("married", "No"),
            "Dependents": int(dependents or 0),
            "Education": form.get("education", "Graduate"),
            "Self_Employed": form.get("self_employed", "No"),
            "ApplicantIncome": int(form.get("applicant_income", 0) or 0),
            "CoapplicantIncome": int(form.get("coapplicant_income", 0) or 0),
            "LoanAmount": int(form.get("loan_amount", 0) or 0),
            "Loan_Amount_Term": int(form.get("loan_term", 360) or 360),
            "Credit_History": int(form.get("credit_history", "1") or 0),
            "Property_Area": form.get("property_area", "Urban"),
        }
        errors = validate_input(data)
        if errors:
            flash("; ".join(errors), "danger")
            return redirect(url_for("predict"))
        result = generate_prediction(data)
        entry = {
            "id": int(datetime.utcnow().timestamp() * 1000),
            "timestamp": datetime.utcnow().strftime("%d %b %Y %H:%M"),
            "applicant": f"{data['Gender']} / {data['Education']}",
            "status": result["status"],
            "confidence": result["confidence"],
            "property_area": data["Property_Area"],
            "credit_history": data["Credit_History"],
            "loan_amount": data["LoanAmount"],
            "income": data["ApplicantIncome"],
            "details": data,
        }
        history = session.get("history", [])
        history.insert(0, entry)
        session["history"] = history[:25]
        session["latest_result"] = {**result, **{"input": data, "entry": entry}}
        append_history(entry)
        return redirect(url_for("result"))

    return render_template(
        "predict.html",
        active_page="predict",
    )


@app.route("/result")
def result():
    initialize_history()
    latest = session.get("latest_result")
    if not latest:
        flash("Please submit a loan application first.", "warning")
        return redirect(url_for("predict"))
    return render_template(
        "result.html",
        active_page="result",
        result=latest,
    )


@app.route("/history")
def history():
    initialize_history()
    entries = session.get("history", [])
    if not entries:
        entries = load_history()
        session["history"] = entries[:25]
    stats = {
        "total": len(entries),
        "approved": sum(1 for item in entries if item["status"] == "Approved"),
        "rejected": sum(1 for item in entries if item["status"] == "Rejected"),
        "avg_confidence": int(sum(item["confidence"] for item in entries) / len(entries)) if entries else 0,
    }
    return render_template(
        "history.html",
        active_page="history",
        entries=entries,
        stats=stats,
    )


@app.route("/history/delete/<int:item_id>", methods=["POST"])
def delete_history(item_id):
    initialize_history()
    session["history"] = [item for item in session.get("history", []) if item["id"] != item_id]
    save_history(session["history"])
    flash("Prediction record removed successfully.", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template(
        "about.html",
        active_page="about",
        data_fields=DATA_FIELDS,
        workflow=WORKFLOW_STEPS,
        stack=TECH_STACK,
        models=MODEL_CARDS,
    )


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thank you! Your message has been received. We will respond shortly.", "success")
        return redirect(url_for("contact"))
    return render_template(
        "contact.html",
        active_page="contact",
        contacts=CONTACT_CARDS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(silent=True) or {}
        errors = validate_input(payload)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400
        result = generate_prediction(payload)
        entry = {
            "id": int(datetime.utcnow().timestamp() * 1000),
            "timestamp": datetime.utcnow().strftime("%d %b %Y %H:%M"),
            "applicant": f"{payload.get('Gender', 'Unknown')} / {payload.get('Education', 'Unknown')}",
            "status": result["status"],
            "confidence": result["confidence"],
            "details": payload,
        }
        append_history(entry)
        return jsonify({"success": True, "prediction": result, "history_entry": entry})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/history")
def api_history():
    return jsonify({"success": True, "history": load_history()})


@app.route("/api/history/<int:item_id>")
def api_history_detail(item_id):
    for entry in load_history():
        if entry.get("id") == item_id:
            return jsonify({"success": True, "entry": entry})
    return jsonify({"success": False, "error": "History entry not found"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
