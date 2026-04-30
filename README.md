​💳 AI Fraud Detection System (Hybrid Engine)
​This project is a Real-Time Fraud Detection System that combines XGBoost Machine Learning with a Custom Heuristic Risk Engine. It doesn't just rely on a static model; it actively checks for environmental red flags like location mismatches, unusual hours, and suspicious email providers.
​🚀 How it Works (The Logic)
​The system uses a Hybrid Scoring Model:
​AI Layer: An XGBoost model analyzes core transaction data (Amount, Card Type, Product Code).
​Heuristic Layer: The extra_features() function checks for "Red Flags" and adds a Risk Boost to the base score.
​Final Verdict:
​< 35% Score: Safe ✅
​35% - 65% Score: Suspicious (Requires OTP) ⚠️
​> 65% Score: Fraudulent 🚨
​📋 Field Explanations
​User Behavior Fields
​Amount: The total money being moved. Large amounts (> ₹20,000) automatically increase risk.
​Time: Used to detect Night-time Transactions (12 AM - 6 AM), which are statistically higher risk for fraudulent activity.
​Location vs. Usual Location: Compares where the user is now versus where they usually are. A mismatch adds a +0.15 risk boost.
​Device vs. Usual Device: Detects if a user is logging in from a new phone or laptop. A mismatch adds a +0.15 risk boost.
​Technical Transaction Fields
​ProductCD: The product category code.
​Card4 & Card6: Identifies the card network (Visa/Mastercard) and type (Debit/Credit). Credit cards receive a slight risk boost (+0.05).
​Distance: The physical distance from the home address. Distances > 100km trigger a +0.20 risk boost.
​Email: Specifically looks for burner/temporary email providers (like tempmail.com). Detection triggers a +0.25 risk boost.
​📈 Real-Time Analytics
​The dashboard includes two live-updating charts using Chart.js:
​ROC Curve: Visualizes the model's accuracy. A curve closer to the top-left corner indicates higher precision.
​Confusion Matrix: A bar chart showing:
​Safe (TN): Correctly identified safe transactions.
​False Alarm (FP): Safe transactions flagged as fraud.
​Missed Fraud (FN): Fraud that got through (the most dangerous).
​Caught Fraud (TP): Fraud correctly blocked by the system.
​🛠️ Installation & Setup
​1. Prerequisites
​Ensure you have the following Python libraries installed:

pip install flask pandas numpy scikit-learn xgboost

2. Directory Structure
​Your folder must look like this:
/your-project-folder
│
├── app.py              # The code provided
├── models/
│   ├── model.pkl       # Your trained XGBoost model
│   └── encoders.pkl    # Your LabelEncoders

python app.py

Open your browser and navigate to http://localhost:5000.
​🔐 Security Features
​Adaptive OTP: If a transaction falls in the "Suspicious" range (35-65% confidence), the system generates a random 6-digit OTP for verification.
​Reasoning Engine: The UI doesn't just show a score; it lists why the transaction was flagged (e.g., "Night-time Transaction", "Location Mismatch").
​🤝 Conclusion
​This system represents a Defense-in-Depth strategy. By layering AI with manual rules, it provides a much higher level of security than a standard machine learning model alone, especially for "Cold Start" fraud where user history is limited.
​Note for "Golu": > अगर आप यह प्रोजेक्ट दिखा रहे हैं, तो ध्यान रखें कि models फोल्डर में model.pkl और encoders.pkl का होना अनिवार्य है, अन्यथा सर्वर स्टार्ट नहीं होगा।
