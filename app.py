from flask import Flask, request, jsonify, render_template, redirect, session
import sqlite3, datetime, random
from sklearn.metrics import roc_curve, confusion_matrix

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DB =================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        password TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

history = []
y_true = []
y_scores = []

# ================= AUTH =================
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email,password))
        if c.fetchone():
            session["user"] = email
            return redirect("/")
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users(email,password) VALUES (?,?)",(email,password))
            conn.commit()
            return redirect("/login")
        except:
            return render_template("register.html", error="User already exists")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= PAGES =================
@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

@app.route("/transactions")
def transactions():
    return render_template("transactions.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

# ================= CORE LOGIC =================
@app.route("/predict", methods=["POST"])
def predict():
    global history, y_true, y_scores

    data = request.get_json()
    boost = 0

    reasons = []

    # SAFE AMOUNT HANDLING
    try:
        amt = float(data.get('amount', 0))
    except:
        amt = 0

    # BASE AI LOGIC (YOUR ORIGINAL)
    ai_base = 0.10
    if amt > 50000:
        ai_base = 0.25

    if data.get('location') != data.get('usual_location'):
        boost += 0.25
        reasons.append("Unusual location")

    if data.get("device") != data.get("usual_device"):
        boost += 0.15
        reasons.append("New device")

    if float(data.get("distance") or 0) > 50:
        boost += 0.15
        reasons.append("Large distance")

        # TIME
    time_str = data.get("time")
    if time_str:
        hour = int(time_str.split(":")[0])
        if hour >= 23 or hour <= 5:
            boost += 0.15
            reasons.append("Night transaction")

        # EMAIL
    email = data.get("email","")
    if "temp" in email or "fake" in email:
        boost += 0.10
        reasons.append("Suspicious email")

        # PRODUCT
    if data.get("productcd") not in ["W","C","R"]:
        boost += 0.10
        reasons.append("Unknown product")

        # CARD
    if data.get("card4") == "american express":
        boost += 0.05
        reasons.append("Risky card")

    if data.get("card6") == "credit":
        boost += 0.05
        reasons.append("Credit usage")
        
    # FINAL SCORE
    final_score = min(ai_base + boost, 1.0)

    if final_score >= 0.75:
        status = "fraud"
    elif final_score >= 0.50:
        status = "suspicious"
    else:
        status = "safe"

    otp = random.randint(100000,999999) if status == ["fraud", "suspicious"] else None

    ts = datetime.datetime.now().strftime("%H:%M:%S")

    history.append({
        "time": ts,
        "amount": amt,
        "prediction": status.upper(),
        "status": status,
        "confidence": final_score
    })

    y_true.append(1 if status=="fraud" else 0)
    y_scores.append(final_score)

    reason_text = ", ".join(reasons) if reasons else "Normal behavior"

    return jsonify({
        "status": status,
        "prediction": status.upper(),
        "confidence": final_score,
        "otp": otp,
        "otp_required":True if status in ["fraud", "suspicious"] else False,
        "time": ts,
        "reason": reason_text
    })

@app.route("/metrics")
def metrics():
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return jsonify({"fpr":[0,1],"tpr":[0,1],"cm":[[1,0],[0,1]]})

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    cm = confusion_matrix(y_true, [1 if s>0.6 else 0 for s in y_scores])

    return jsonify({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "cm": cm.tolist()})


@app.route("/history")
def get_history():
    return jsonify(history)
if __name__ == "__main__":
    app.run(debug=True)