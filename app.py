from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_curve, confusion_matrix

app = Flask(__name__)

# =========================
# LOAD MODEL (NEW)
# =========================
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# =========================
# STORAGE
# =========================
history = []
y_true = []
y_scores = []

# =========================
# HTML UI (UNCHANGED)
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Fraud Detection System</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body { font-family: 'Segoe UI'; background: #f4f6f9; padding: 20px; }
.container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
input, select { margin: 5px; padding: 8px; width: 220px; border-radius: 5px; border: 1px solid #ccc; }
button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }
button:hover { background: #2980b9; }
.result-safe { color: green; font-weight: bold; font-size: 20px; }
.result-fraud { color: red; font-weight: bold; font-size: 20px; }
.result-suspicious { color: orange; font-weight: bold; font-size: 20px; }
.row { display: flex; gap: 20px; margin-top: 20px; }
canvas { background: white; padding: 10px; border-radius: 10px; width: 400px !important; height: 300px !important; }
table { width: 100%; margin-top: 15px; border-collapse: collapse; }
th { background: #3498db; color: white; }
td, th { padding: 10px; border: 1px solid #ddd; text-align: center; }
tr:nth-child(even) { background: #f2f2f2; }
</style>
</head>

<body>
<div class="container">

<h1>💳 AI Fraud Detection System</h1>

<form id="form">

Amount: <input id="amount"><br>
Time: <input type="datetime-local" id="time"><br>
Location: <input id="location"><br>

Device:
<select id="device">
<option>mobile</option><option>desktop</option>
<option>laptop</option><option>tablet</option>
</select><br>

Usual Location: <input id="usual_location"><br>

Usual Device:
<select id="usual_device">
<option>mobile</option><option>desktop</option>
<option>laptop</option><option>tablet</option>
</select><br>

ProductCD: <input id="product"><br>
Card4: <input id="card4"><br>
Card6: <input id="card6"><br>
Address: <input id="addr"><br>
Distance: <input id="dist"><br>
Email: <input id="email"><br>

<button type="submit">🔍 Check Fraud</button>
</form>

<h2 id="result"></h2>

<div class="row">
<canvas id="roc"></canvas>
<canvas id="cm"></canvas>
</div>

<h3>📜 Transaction History</h3>
<table>
<thead>
<tr><th>Amount</th><th>Prediction</th><th>Confidence</th></tr>
</thead>
<tbody id="history"></tbody>
</table>

</div>

<script>
document.getElementById("form").onsubmit = async (e) => {
    e.preventDefault();

    let data = {
        amount: document.getElementById("amount").value,
        time: document.getElementById("time").value,
        location: document.getElementById("location").value,
        device: document.getElementById("device").value,
        usual_location: document.getElementById("usual_location").value,
        usual_device: document.getElementById("usual_device").value,

        TransactionAmt: document.getElementById("amount").value,
        ProductCD: document.getElementById("product").value,
        card4: document.getElementById("card4").value,
        card6: document.getElementById("card6").value,
        addr1: document.getElementById("addr").value,
        dist1: document.getElementById("dist").value,
        P_emaildomain: document.getElementById("email").value
    };

    let res = await fetch("/predict", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(data)
    });

    let r = await res.json();

    let resultBox = document.getElementById("result");

    if (r.prediction.includes("Fraud")) {
        resultBox.className = "result-fraud";
    } else if (r.prediction.includes("Suspicious")) {
        resultBox.className = "result-suspicious";
    } else {
        resultBox.className = "result-safe";
    }

    resultBox.innerHTML =
    r.prediction + " (" + (r.confidence*100).toFixed(2) + "%)<br><br>" +
    "<b>Reason:</b><br>" + r.reasons.join("<br>") +
    (r.otp_required ? "<br><br>🔐 OTP: " + r.otp : "");

    loadDashboard();
};

async function loadDashboard() {

    let res = await fetch("/metrics");
    let data = await res.json();

    if(window.rocChart) window.rocChart.destroy();
    if(window.cmChart) window.cmChart.destroy();

    window.rocChart = new Chart(document.getElementById("roc"), {
        type: "line",
        data: {
            labels: data.fpr,
            datasets: [{ label:"ROC Curve", data: data.tpr }]
        }
    });

    window.cmChart = new Chart(document.getElementById("cm"), {
        type: "bar",
        data: {
            labels:["TN","FP","FN","TP"],
            datasets:[{
                data:[
                    data.cm[0][0],
                    data.cm[0][1],
                    data.cm[1][0],
                    data.cm[1][1]
                ]
            }]
        }
    });

    let h = await fetch("/history");
    let hist = await h.json();

    let table = document.getElementById("history");
    table.innerHTML = "";

    hist.slice().reverse().forEach(tx => {
        table.innerHTML += `<tr>
            <td>₹${tx.amount}</td>
            <td>${tx.prediction}</td>
            <td>${(tx.confidence*100).toFixed(2)}%</td>
        </tr>`;
    });
}
</script>

</body>
</html>
"""

# =========================
# EXTRA FEATURES
# =========================
def extra_features(data):
    hour = int(data['time'][11:13]) if data['time'] else 12
    loc = 1 if data['location'] != data['usual_location'] else 0
    dev = 1 if data['device'] != data['usual_device'] else 0
    night = 1 if hour < 6 else 0
    return hour, loc, dev, night

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    global history, y_true, y_scores

    data = request.get_json()

    amount = float(data.get('amount', 0) or 0)
    txn_amt = float(data.get('TransactionAmt', 0) or 0)

    df_input = pd.DataFrame([{
        'TransactionAmt': txn_amt,
        'ProductCD': data.get('ProductCD', "unknown"),
        'card4': data.get('card4', "unknown"),
        'card6': data.get('card6', "unknown"),
        'addr1': data.get('addr1', 0),
        'dist1': data.get('dist1', 0),
        'P_emaildomain': data.get('P_emaildomain', "unknown")
    }]).fillna(0)
    
    df_input = pd.DataFrame([{
        'TransactionAmt': txn_amt,
        'ProductCD': data.get('ProductCD', "unknown"),
        'card4': data.get('card4', "unknown"),
        'card1': data.get('card1', 0), 
        'card6': data.get('card6', "unknown"),
        'addr1': data.get('addr1', 0),
        'dist1': data.get('dist1', 0),
        'P_emaildomain': data.get('P_emaildomain', "unknown")
    }]).fillna(0)
    df_input["amt_log"] = np.log1p(df_input["TransactionAmt"])
    df_input["amt_squared"] = df_input["TransactionAmt"] ** 2
    df_input["amt_inverse"] = 1 / (df_input["TransactionAmt"] + 1)
    df_input["amt_sqrt"] = np.sqrt(df_input["TransactionAmt"])
    df_input["is_high_amt"] = (df_input["TransactionAmt"] > 20000).astype(int)
    df_input["is_low_amt"] = (df_input["TransactionAmt"] < 100).astype(int)
    df_input["amt_range"] = pd.cut(df_input["TransactionAmt"], bins=5, labels=False)


    # 🔥 Simulated behavior (safe default)
    df_input["txn_count"] = 1
    df_input["avg_amt"] = df_input["TransactionAmt"]
    df_input["amt_dev"] = 0

    # Encoding
    for col in df_input.columns:
        if col in encoders:
            le = encoders[col]
            val = df_input[col].astype(str)
            val = val.apply(lambda x: x if x in le.classes_ else "unknown")
            if "unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "unknown")
            df_input[col] = le.transform(val)

    score = model.predict_proba(df_input)[0][1]
    if np.isnan(score):
        score = 0.1

    hour, loc_c, dev_c, night = extra_features(data)

    risk_boost = 0
    if loc_c: risk_boost += 0.2
    if dev_c: risk_boost += 0.2
    if night: risk_boost += 0.1
    if amount > 20000: risk_boost += 0.2

    score = min(score + risk_boost, 1.0)

    reasons = []
    if txn_amt > 20000: reasons.append("High Transaction Amount")
    if loc_c: reasons.append("Unusual Location")
    if dev_c: reasons.append("New Device Used")
    if night: reasons.append("Night Transaction")
    if len(reasons) == 0: reasons.append("Normal Behavior")

    if score < 0.4:
        status = "safe"
    elif score < 0.7:
        status = "suspicious"
    else:
        status = "fraud"

    if (loc_c + dev_c + night) >= 2 and score < 0.4:
        status = "suspicious"

    otp = None
    if status == "suspicious":
        otp = np.random.randint(100000, 999999)

    if status == "safe":
        pred_text = "Safe"
    elif status == "suspicious":
        pred_text = "Suspicious"
    else:
        pred_text = "Fraudulent"

    history.append({
        "amount": amount,
        "prediction": pred_text,
        "confidence": float(score)
    })

    y_true.append(1 if status == "fraud" else 0)
    y_scores.append(score)

    if status == "safe":
        message = "✅ Safe Transaction"
    elif status == "suspicious":
        message = "⚠️ Suspicious Transaction - OTP Required"
    else:
        message = "🚨 Fraudulent Transaction"

    return jsonify({
        "prediction": message,
        "confidence": float(score),
        "reasons": reasons,
        "otp_required": status == "suspicious",
        "otp": otp
    })

@app.route('/metrics')
def metrics():
    if len(y_true) < 5 or len(set(y_true)) < 2:
        return jsonify({"fpr":[0,1],"tpr":[0,1],"cm":[[1,0],[0,1]]})

    y_pred = [1 if s > 0.65 else 0 for s in y_scores]
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return jsonify({
        "cm": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    })

@app.route('/history')
def get_history():
    return jsonify(history)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("🚀 Server Starting...")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)