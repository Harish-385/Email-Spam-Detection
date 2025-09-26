from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from predict import predict_email

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for session storage

# Initialize session state-like storage
def init_session():
    if "total_emails" not in session:
        session["total_emails"] = 0
        session["spam_count"] = 0
        session["ham_count"] = 0
        session["prediction_history"] = []

@app.route("/", methods=["GET", "POST"])
def index():
    init_session()
    message = None
    result = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()
        if email_text == "":
            message = "⚠️ Please enter email content."
        else:
            # ✅ Check if this message was already classified before
            prev_prediction = next(
                (h["prediction"] for h in session["prediction_history"] if h["email_preview"] == email_text[:50]),
                None
            )

            if prev_prediction:
                # Use previous result for consistency and normalize to lower for logic checks
                result = str(prev_prediction).strip().lower()
            else:
                # Call the prediction model
                raw_result = predict_email(email_text)

                # ✅ Normalize result
                result = str(raw_result).strip().lower()
                if result not in ["spam", "ham"]:
                    result = "ham"  # default fallback

                # Save to history only if new
                history_item = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": result.capitalize(),
                    "email_preview": email_text[:50] + "..." if len(email_text) > 50 else email_text,
                }
                session["prediction_history"].append(history_item)

                # Update counts
                session["total_emails"] += 1
                if result == "spam":
                    session["spam_count"] += 1
                else:
                    session["ham_count"] += 1

            # Show friendly message
            if result == "spam":
                message = "🚫 This email is SPAM!"
            else:
                message = "✅ This email is NOT SPAM (Ham)."

    return render_template(
        "index.html",
        total_emails=session["total_emails"],
        spam_count=session["spam_count"],
        ham_count=session["ham_count"],
        message=message,
        result=result
    )

@app.route("/stats")
def stats():
    init_session()
    df_history = pd.DataFrame(session["prediction_history"])
    chart_pie = None
    chart_bar = None

    if not df_history.empty:
        # Pie chart
        fig_pie = px.pie(
            values=[session["spam_count"], session["ham_count"]],
            names=["Spam", "Ham"],
            title="Email Classification Distribution",
            color_discrete_map={'Spam': '#ff6b6b', 'Ham': '#51cf66'}
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        chart_pie = fig_pie.to_html(full_html=False)

        # Daily stats
        df_history["date"] = pd.to_datetime(df_history["timestamp"]).dt.date
        daily_stats = df_history.groupby(["date", "prediction"]).size().unstack(fill_value=0)

        if not daily_stats.empty:
            fig_bar = go.Figure()
            if "Spam" in daily_stats.columns:
                fig_bar.add_trace(go.Bar(x=daily_stats.index, y=daily_stats["Spam"], name="Spam", marker_color="#ff6b6b"))
            if "Ham" in daily_stats.columns:
                fig_bar.add_trace(go.Bar(x=daily_stats.index, y=daily_stats["Ham"], name="Ham", marker_color="#51cf66"))

            fig_bar.update_layout(
                title="Daily Prediction Timeline",
                xaxis_title="Date",
                yaxis_title="Number of Predictions",
                barmode="stack"
            )
            chart_bar = fig_bar.to_html(full_html=False)

    return render_template(
        "stats.html",
        total_emails=session["total_emails"],
        spam_count=session["spam_count"],
        ham_count=session["ham_count"],
        chart_pie=chart_pie,
        chart_bar=chart_bar,
        history=session["prediction_history"][-10:]
    )

@app.route("/reset", methods=["POST"])
def reset():
    init_session()
    session["total_emails"] = 0
    session["spam_count"] = 0
    session["ham_count"] = 0
    session["prediction_history"] = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
