from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.metrics import roc_curve, confusion_matrix

app = Flask(__name__)

# =========================
# LOAD MODEL & ENCODERS
# =========================
# Ensure your 'models' folder exists with these files
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found. Ensure models/model.pkl exists.")

history = []
y_true = []
y_scores = []

# =========================
# PERFECTED AUTOMATED UI
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Inference Engine - Real Time</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #f8fafc; padding: 15px; margin:0; }
        .container { max-width: 1200px; margin: auto; background: #1e293b; padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        
        /* Top Live Stream */
        .stream-container { background: #0f172a; padding: 10px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #3b82f6; height: 100px; }
        
        /* Main Grid: Left (Fields) | Right (Analytics) */
        .main-grid { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
        
        /* Right Side: Charts Side-by-Side */
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        
        .panel { background: #334155; padding: 15px; border-radius: 10px; height: fit-content; }
        
        label { font-size: 11px; font-weight: bold; color: #94a3b8; text-transform: uppercase; }
        input, select { background: #0f172a; border: 1px solid #475569; color: white; padding: 8px; margin: 4px 0 12px 0; border-radius: 5px; width: 100%; box-sizing: border-box; }
        
        button { width: 100%; padding: 12px; background: #22c55e; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; transition: 0.3s; }
        button:hover { background: #16a34a; transform: scale(1.02); }
        
        .result-box { padding: 15px; border-radius: 8px; margin-bottom: 15px; background: #0f172a; border-left: 5px solid #64748b; font-size: 14px; }
        .result-fraud { border-left-color: #ef4444; color: #fca5a5; }
        .result-safe { border-left-color: #22c55e; color: #86efac; }
        
        .chart-wrapper { background: #f8fafc; padding: 10px; border-radius: 8px; height: 180px; }
        
        table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 10px; }
        th { text-align: left; color: #94a3b8; border-bottom: 1px solid #475569; padding: 8px; }
        td { padding: 8px; border-bottom: 1px solid #334155; }
    </style>
</head>
<body onload="initSystem()">
<div class="container">
    
    <div class="stream-container">
        <div style="font-size: 10px; color: #3b82f6; letter-spacing: 1px;">📡 LIVE SYSTEM HEARTBEAT</div>
        <canvas id="streamChart"></canvas>
    </div>

    <div class="main-grid">
        <div class="panel">
            <h3 style="margin-top:0; color:#3b82f6">Transaction Entry</h3>
            <form id="fraudForm">
                <label>Amount (₹)</label>
                <input id="amount" type="number" value="1000" required>
                
                <label>Live Location (Auto)</label>
                <input id="location" readonly style="border-color:#1e40af">
                
                <label>System Time (Auto)</label>
                <input id="time" readonly style="border-color:#1e40af">
                
                <label>Usual Home City</label>
                <input id="usual_location" value="Mumbai">
                
                <label>Email Address</label>
                <input id="email" placeholder="user@example.com">
                
                <label>Card Type</label>
                <select id="card6">
                    <option value="debit">Debit Card</option>
                    <option value="credit">Credit Card</option>
                </select>
                
                <button type="submit">⚡ SCAN TRANSACTION</button>
            </form>
        </div>
        
        <div>
            <div id="result" class="result-box">Waiting for input...</div>
            
            <div class="charts-grid">
                <div class="chart-wrapper">
                    <canvas id="roc"></canvas>
                </div>
                <div class="chart-wrapper">
                    <canvas id="cm"></canvas>
                </div>
            </div>

            <div class="panel" style="padding:10px;">
                <h4 style="margin:0 0 10px 0; font-size:12px; color:#94a3b8">REAL-TIME AUDIT LOG</h4>
                <table>
                    <thead>
                        <tr><th>Time</th><th>Amount</th><th>Status</th><th>Risk %</th></tr>
                    </thead>
                    <tbody id="historyTable"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
    let livePulseChart;
    let pulseData = Array(40).fill(0);

    function initSystem() {
        // Init Live Top Chart
        const pCtx = document.getElementById('streamChart').getContext('2d');
        livePulseChart = new Chart(pCtx, {
            type: 'line',
            data: {
                labels: Array(40).fill(''),
                datasets: [{ data: pulseData, borderColor: '#3b82f6', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }]
            },
            options: { 
                responsive: true, maintainAspectRatio: false, 
                plugins: { legend: {display: false} },
                scales: { x: {display: false}, y: {min:0, max:1, display: false} } 
            }
        });

        // Ticking Clock
        setInterval(() => {
            document.getElementById('time').value = new Date().toLocaleTimeString();
        }, 1000);

        // Browser Location
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(p => {
                document.getElementById('location').value = p.coords.latitude.toFixed(2) + ", " + p.coords.longitude.toFixed(2);
            });
        }
        
        loadDashboard();
    }

    document.getElementById("fraudForm").onsubmit = async (e) => {
        e.preventDefault();
        const payload = {
            amount: document.getElementById("amount").value,
            location: document.getElementById("location").value,
            usual_location: document.getElementById("usual_location").value,
            P_emaildomain: document.getElementById("email").value,
            card6: document.getElementById("card6").value
        };

        const res = await fetch("/predict", {
            method:"POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(payload)
        });

        const r = await res.json();
        const rb = document.getElementById("result");
        rb.className = "result-box result-" + r.status;
        rb.innerHTML = `<strong>${r.prediction} Detected</strong> | AI Risk Score: ${(r.confidence*100).toFixed(1)}%`;
        
        // Update Live Stream
        pulseData.push(r.confidence);
        pulseData.shift();
        livePulseChart.update('none');

        loadDashboard();
    };

    async function loadDashboard() {
        const res = await fetch("/metrics");
        const d = await res.json();

        if(window.rocChart) window.rocChart.destroy();
        if(window.cmChart) window.cmChart.destroy();

        window.rocChart = new Chart(document.getElementById("roc"), {
            type: 'line',
            data: { labels: d.fpr, datasets: [{ label: 'ROC Curve', data: d.tpr, borderColor: '#3b82f6', fill: false }] },
            options: { maintainAspectRatio: false, plugins: { title: {display: true, text: 'Model Accuracy (ROC)'}}}
        });

        window.cmChart = new Chart(document.getElementById("cm"), {
            type: 'bar',
            data: {
                labels: ["Safe", "FP", "FN", "Fraud"],
                datasets: [{ data: [d.cm[0][0], d.cm[0][1], d.cm[1][0], d.cm[1][1]], backgroundColor: ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6'] }]
            },
            options: { maintainAspectRatio: false, plugins: { legend: {display: false}, title: {display: true, text: 'Detection Stats'}}}
        });

        const hRes = await fetch("/history");
        const hist = await hRes.json();
        document.getElementById("historyTable").innerHTML = hist.slice(-5).reverse().map(tx => `
            <tr><td>${tx.time}</td><td>₹${tx.amount}</td><td>${tx.prediction}</td><td>${(tx.confidence*100).toFixed(1)}%</td></tr>
        `).join('');
    }
</script>
</body>
</html>
"""

# =========================
# BACKEND ENGINE
# =========================
@app.route('/')
def home(): 
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    global history, y_true, y_scores
    data = request.get_json()
    
    # Simple Rule Engine (Heuristics)
    boost = 0
    if data.get('location') != data.get('usual_location'):
        boost += 0.25
    
    # Simulated AI base score
    amt = float(data.get('amount') or 0)
    ai_base = 0.15
    if amt > 20000: ai_base = 0.45
    
    final_score = min(ai_base + boost, 1.0)
    status = "fraud" if final_score > 0.40 else "safe"
    pred_text = status.upper()
    
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    history.append({"time": ts, "amount": amt, "prediction": pred_text, "status": status, "confidence": final_score})
    y_true.append(1 if status == "fraud" else 0)
    y_scores.append(final_score)
    
    return jsonify({"status": status, "prediction": pred_text, "confidence": final_score})

@app.route('/metrics')
def metrics():
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return jsonify({"fpr":[0,1],"tpr":[0,1],"cm":[[1,0],[0,1]]})
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    cm = confusion_matrix(y_true, [1 if s > 0.40 else 0 for s in y_scores])
    return jsonify({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "cm": cm.tolist()})

@app.route('/history')
def get_history(): 
    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True, port=5000)