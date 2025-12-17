import gradio as gr
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. MODEL AND DATA SETUP ---
# IMPORTANT: Ensure your model files ('random_forest_diabetes_model.pkl', 'scaler.pkl') 
# are present in the same directory as this app.py file.
try:
    model = joblib.load('random_forest_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    model, scaler = None, None
    print(f"❌ WARNING: Model/scaler files not loaded: {e}. App will show error on prediction.")

# CORRECTED: Using 'minimum', 'maximum', and 'value' for Gradio Slider compatibility
MEDICAL_RANGES = {
    'Glucose': {'minimum': 70, 'maximum': 500, 'value': 100, 'step': 1, 'info': "Fasting blood glucose (mg/dL)"},
    'BloodPressure': {'minimum': 60, 'maximum': 180, 'value': 80, 'step': 1, 'info': "Diastolic blood pressure (mm Hg)"},
    'BMI': {'minimum': 15, 'maximum': 50, 'value': 24, 'step': 0.1, 'info': "Body Mass Index"},
    'Age': {'minimum': 20, 'maximum': 80, 'value': 45, 'step': 1, 'info': "Age in years"}
}
ORIGINAL_FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- 2. CORE LOGIC FUNCTIONS ---

def get_medical_status(value, feature):
    """Returns status and severity for dynamic HTML display."""
    if feature == 'Glucose':
        if value < 70: return "Hypoglycemic", "alert"
        elif value < 100: return "Normal", "normal"
        elif value < 126: return "Prediabetic", "warning"
        else: return "Diabetic", "alert"
    
    elif feature == 'BloodPressure':
        if value < 80: return "Normal", "normal"
        elif value < 90: return "Elevated", "warning"
        else: return "Hypertensive", "alert"
    
    elif feature == 'BMI':
        if value < 18.5: return "Underweight", "warning"
        elif value < 25: return "Normal", "normal"
        elif value < 30: return "Overweight", "warning"
        else: return "Obese", "alert"
    
    elif feature == 'Age':
        if value < 35: return "Young", "normal"
        elif value < 55: return "Middle-aged", "normal"
        else: return "Senior", "warning"
    return "N/A", "normal"

def predict_diabetes(glucose, blood_pressure, bmi, age):
    """Runs the prediction and generates the comprehensive result dictionary."""
    if model is None or scaler is None:
        return {"Error": "Model not loaded. Cannot predict."}

    # Dummy feature values for features not in the GUI
    inputs = [0, glucose, blood_pressure, 23.0, 79.8, bmi, 0.42, age]
    input_df = pd.DataFrame([inputs], columns=ORIGINAL_FEATURE_NAMES).astype(float)
    
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0]
    diabetes_prob = probability[1] * 100

    # Risk classification
    if diabetes_prob >= 75:
        risk_level, recommendation = "CRITICAL RISK", "Consult healthcare provider immediately for comprehensive evaluation and management."
    elif diabetes_prob >= 50:
        risk_level, recommendation = "HIGH RISK", "Schedule medical consultation and consider urgent lifestyle modifications with regular monitoring."
    elif diabetes_prob >= 25:
        risk_level, recommendation = "MODERATE RISK", "Schedule a consultation and maintain healthy lifestyle with annual screenings."
    else:
        risk_level, recommendation = "LOW RISK", "Continue current healthy habits. Routine preventive care is recommended."
    
    # Clinical findings
    current_metrics = {'Glucose': glucose, 'BloodPressure': blood_pressure, 'BMI': bmi, 'Age': age}
    concerning_metrics = [
        f"**{feature}**: {value} ({get_medical_status(value, feature)[0]})" 
        for feature, value in current_metrics.items() if get_medical_status(value, feature)[1] != "normal"
    ]
        
    return {
        "Diabetes Probability": f"{diabetes_prob:.1f}%",
        "Risk Classification": risk_level,
        "Recommendation": recommendation,
        "Clinical Findings": concerning_metrics
    }

def generate_metric_html(glucose_val, bp_val, bmi_val, age_val):
    """Generates the dynamic metric status grid HTML."""
    data = {
        'Glucose': (glucose_val, ' mg/dL'),
        'BloodPressure': (bp_val, ' mm Hg'),
        'BMI': (bmi_val, ''),
        'Age': (age_val, ' years')
    }
    html_output = ""
    for feature, (value, unit) in data.items():
        status, severity = get_medical_status(value, feature)
        severity_class = 'status-normal' if severity == 'normal' else 'status-alert' if severity == 'alert' else 'status-warning'
        
        html_output += f"""
        <div class="metric-card">
            <h4 style="margin-top: 0; color: #00A389;">{feature}</h4>
            <div class="value-display">{value}{unit}</div>
            <div class="status-indicator {severity_class}">{status}</div>
        </div>
        """
    return f'<div class="metric-grid">{html_output}</div>'

def update_risk_display_html(prediction_result):
    """Generates the main risk indicator HTML (Gauge effect)."""
    if "Error" in prediction_result:
        return f'<div class="risk-indicator risk-critical"><h3 style="margin: 0; font-size: 2rem; color: #c33;">Assessment Error</h3><p>{prediction_result["Error"]}</p></div>'
    
    risk_level = prediction_result.get("Risk Classification", "UNKNOWN")
    probability = prediction_result.get("Diabetes Probability", "0%")
    
    if "CRITICAL" in risk_level: risk_class, color = "risk-critical", "#c33"
    elif "HIGH" in risk_level: risk_class, color = "risk-high", "#ff9800"
    elif "MODERATE" in risk_level: risk_class, color = "risk-moderate", "#963"
    else: risk_class, color = "risk-low", "#2e7d32"
    
    # Visualization using custom HTML/CSS class (simulating a gauge) 
    return f"""
    <div class="risk-indicator {risk_class}">
        <h3 style="margin: 0; font-size: 2.5rem; color: {color};">{risk_level}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.4rem; font-weight: 700;">Diabetes Probability: {probability}</p>
    </div>
    """

def update_findings_display_html(prediction_result):
    """Generates the detailed findings and recommendation HTML."""
    if "Error" in prediction_result: return ""
    
    recommendation = prediction_result.get("Recommendation", "N/A")
    findings = prediction_result.get("Clinical Findings", [])
    
    if findings:
        findings_list = "".join([f"<li>{finding}</li>" for finding in findings])
        findings_html = f"""
            <h4 style="margin: 0 0 1rem 0; color: #dc3545;">⚠️ Notable Clinical Findings</h4>
            <ul style="margin: 0; padding-left: 1.5rem; color: #495057;">{findings_list}</ul>
        """
    else:
        findings_html = """
            <h4 style="margin: 0 0 1rem 0; color: #28a745;'>✅ Clinical Findings: All metrics are within normal range.</h4>
            <p style="color: #495057; margin: 0;">No concerning metrics were flagged by the clinical guidelines.</p>
        """
        
    return f"""
    <div class="clinical-findings-card">
        {findings_html}
        <h4 style="margin: 1.5rem 0 0.5rem 0; color: #00A389;">Clinical Recommendation</h4>
        <p style="font-size: 1.1rem; color: #333; font-weight: 500;">{recommendation}</p>
    </div>
    """

# --- 3. GRADIO INTERFACE AND CSS ---

CUSTOM_CSS = """
    /* General Styling */
    .gradio-container { max-width: 1200px; margin: 0 auto; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f0f2f5; }
    .clinical-container { background: white; border-radius: 15px; margin: 25px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.1); overflow: hidden; }
    
    /* Header Styling */
    .clinical-header { background: linear-gradient(135deg, #00A389 0%, #007c6f 100%); padding: 2rem; color: white; }
    .clinical-header h1 { color: white; font-weight: 800; font-size: 2.8rem; margin: 0; }
    .clinical-header h3 { color: #e0f2f1; font-weight: 400; font-size: 1.2rem; margin: 0; }

    /* Input & Results Panels */
    .input-panel { padding: 2.5rem; border-bottom: 1px solid #e1e5e9; }
    .results-panel { background: #f8f9fa; padding: 2.5rem; }

    /* Metric Grid for Status Display */
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin: 2rem 0; }
    .metric-card { background: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #00A389; }
    .value-display { font-size: 1.8rem; font-weight: 700; color: #333333; margin: 0.25rem 0; }
    .status-indicator { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .status-normal { background: #d4edda; color: #155724; }
    .status-warning { background: #fff3cd; color: #856404; }
    .status-alert { background: #f8d7da; color: #721c24; }
    
    /* Submit Button (Primary Action) */
    .submit-professional { background: #00A389; color: white; border: none; padding: 1.5rem 3rem; border-radius: 10px; font-weight: 700; font-size: 1.25rem; width: 100%; margin: 2rem 0 0 0; box-shadow: 0 6px 15px rgba(0, 163, 137, 0.3); }
    .submit-professional:hover { background: #00876f; }

    /* Risk Indicator (Dashboard Look) */
    .risk-indicator { padding: 3rem; border-radius: 15px; margin: 2rem 0; text-align: center; font-weight: 700; box-shadow: 0 4px 15px rgba(0,0,0,0.15); }
    .risk-critical { background: #fff1f1; border: 2px solid #e57373; }
    .risk-high { background: #fff8f1; border: 2px solid #ff9800; } 
    .risk-moderate { background: #fefee3; border: 2px solid #ffc107; }
    .risk-low { background: #f0fff0; border: 2px solid #4caf50; }
    .risk-indicator h3 { font-size: 3rem; margin: 0; }

    /* Findings Card and Disclaimer */
    .clinical-findings-card { background: white; border-radius: 10px; padding: 2rem; margin: 1.5rem 0; border: 1px solid #e1e5e9; }
    .clinical-findings-card li { margin-bottom: 0.75rem; border-left: 3px solid #dc3545; padding-left: 15px; }
    .medical-disclaimer { background: #fff8e1; border-left: 5px solid #ffc107; border-radius: 8px; padding: 1.5rem; margin: 2rem 0; }
    
    /* Responsiveness */
    @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 500px) { .metric-grid { grid-template-columns: 1fr; } }
"""

with gr.Blocks(
    title="Clinical Diabetes Risk Assessment",
    theme=gr.themes.Soft(), 
    css=CUSTOM_CSS
) as demo:
    
    with gr.Column(elem_classes="clinical-container"):
        
        # Header (Top Bar with Gradient)
        with gr.Column(elem_classes="clinical-header"):
            gr.Markdown("<h1>💎 Diabetes Risk Intelligence</h1>")
            gr.Markdown("<h3>Clinical Risk Assessment System powered by Random Forest</h3>")
        
        # --- INPUT PANEL ---
        with gr.Column(elem_classes="input-panel"):
            gr.Markdown("## 🧑‍⚕️ Patient Health Metrics Input")
            
            with gr.Row():
                with gr.Column():
                    
                    # CORRECTED SLIDER ARGUMENT USAGE
                    glucose = gr.Slider(
                        label=f"Fasting Glucose (mg/dL) [Normal: 70-100]",
                        minimum=MEDICAL_RANGES['Glucose']['minimum'],
                        maximum=MEDICAL_RANGES['Glucose']['maximum'],
                        value=MEDICAL_RANGES['Glucose']['value'],
                        step=MEDICAL_RANGES['Glucose']['step']
                    )
                    blood_pressure = gr.Slider(
                        label=f"Diastolic Blood Pressure (mm Hg) [Normal: 60-80]",
                        minimum=MEDICAL_RANGES['BloodPressure']['minimum'],
                        maximum=MEDICAL_RANGES['BloodPressure']['maximum'],
                        value=MEDICAL_RANGES['BloodPressure']['value'],
                        step=MEDICAL_RANGES['BloodPressure']['step']
                    )
                with gr.Column():
                    bmi = gr.Slider(
                        label=f"Body Mass Index (BMI) [Normal: 18.5-25]",
                        minimum=MEDICAL_RANGES['BMI']['minimum'],
                        maximum=MEDICAL_RANGES['BMI']['maximum'],
                        value=MEDICAL_RANGES['BMI']['value'],
                        step=MEDICAL_RANGES['BMI']['step']
                    )
                    age = gr.Slider(
                        label="Age (years)",
                        minimum=MEDICAL_RANGES['Age']['minimum'],
                        maximum=MEDICAL_RANGES['Age']['maximum'],
                        value=MEDICAL_RANGES['Age']['value'],
                        step=MEDICAL_RANGES['Age']['step']
                    )
            
            # Dynamic Metric Status Display 
            metric_status_output = gr.HTML(
                value=generate_metric_html(100, 80, 24, 45)
            )
            
            submit_btn = gr.Button(
                "📈 GENERATE CLINICAL ASSESSMENT",
                elem_classes="submit-professional"
            )
        
        # --- RESULTS PANEL ---
        with gr.Column(elem_classes="results-panel"):
            
            # 1. Primary Risk Indicator (Gauge)
            risk_display = gr.HTML(
                value='<div class="risk-indicator risk-low"><h3 style="font-size: 3rem; color: #6c757d;">Assessment Pending</h3><p>Enter metrics and click "Generate Assessment"</p></div>'
            )
            
            with gr.Tabs():
                
                # Tab 1: Recommendation and Findings
                with gr.TabItem("Summary & Recommendation"):
                    findings_display = gr.HTML(
                        label="Clinical Findings",
                        value='<div class="clinical-findings-card"><h4 style="color: #00A389;">Clinical Summary</h4><p style="color: #6c757d;">Detailed report will appear here.</p></div>'
                    )

                # Tab 2: Raw Model Output
                with gr.TabItem("Model Raw Output (JSON)"):
                    gr.Markdown("### ⚙️ Technical Model Output")
                    results_json = gr.JSON(show_label=False)

        # Examples
        with gr.Column():
            gr.Markdown("### 🧪 Quick Test Cases")
            gr.Examples(
                examples=[
                    [95, 78, 22.5, 32],      
                    [115, 85, 27.8, 48],     
                    [160, 92, 31.2, 55],     
                    [210, 105, 35.6, 62]     
                ],
                inputs=[glucose, blood_pressure, bmi, age],
                label="Click to load example data:"
            )

        # Disclaimer
        with gr.Column():
            gr.Markdown(
                """
                <div class="medical-disclaimer">
                    <strong>⚠️ Clinical Disclaimer:</strong> This tool provides **risk assessment** based on statistical models and **should not replace professional medical evaluation**. 
                    Always consult qualified healthcare providers for diagnosis and treatment decisions.
                </div>
                """
            )

    # --- EVENT HANDLERS ---
    
    # 1. Update the metric status display when any slider changes
    for slider in [glucose, blood_pressure, bmi, age]:
        slider.change(
            generate_metric_html,
            inputs=[glucose, blood_pressure, bmi, age],
            outputs=[metric_status_output]
        )
    
    # 2. Primary Click Action (Sequential updates for all outputs)
    submit_btn.click(
        fn=predict_diabetes,
        inputs=[glucose, blood_pressure, bmi, age],
        outputs=results_json
    ).then(
        fn=update_risk_display_html,
        inputs=results_json,
        outputs=risk_display
    ).then(
        fn=update_findings_display_html, 
        inputs=results_json,
        outputs=findings_display
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )