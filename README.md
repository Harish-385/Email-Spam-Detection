# 📧 Email Spam Detection Web App

A Flask-based web application that detects whether an email is **Spam** or **Ham** using a Machine Learning model.  
The app also provides interactive statistics and visualizations with **Plotly**.  

🔗 **Live Demo:** [Email Spam Detection on Render](https://email-spam-detection-pf6o.onrender.com)

---

## 🚀 Features
- 📨 Classify emails as **Spam** or **Ham** in real-time  
- 📊 Interactive **Pie & Bar Charts** for spam vs ham distribution  
- 📅 Daily prediction timeline  
- 🧾 Keeps track of classification history (session-based)  
- 🔄 Reset option to start fresh  

---

## 🛠️ Tech Stack
- **Flask** – Web framework  
- **scikit-learn** – ML model (Naive Bayes + TF-IDF)  
- **pandas** – Data handling  
- **plotly** – Interactive charts  
- **gunicorn** – Production server  

---

## 📦 Installation & Setup
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/email-spam-detection.git
   cd email-spam-detection
