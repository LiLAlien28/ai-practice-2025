import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
try:
    model = joblib.load('random_forest_diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Extended value ranges for real-world scenarios
VALUE_RANGES = {
    'Glucose': (0, 500, 120, "Plasma glucose concentration (mg/dL)"),
    'BloodPressure': (0, 200, 80, "Diastolic blood pressure (mm Hg)"),
    'BMI': (0.0, 80.0, 28.0, "Body Mass Index"),
    'Age': (1, 120, 35, "Age in years")
}

# Original feature names from training
ORIGINAL_FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def predict_diabetes(glucose, blood_pressure, bmi, age):
    """
    Predict diabetes probability based on input features
    """
    try:
        if model is None or scaler is None:
            return {"Error": "Model not loaded properly. Please check model files."}
        
        # Set default values for removed features
        pregnancies = 0
        skin_thickness = 29.0
        insulin = 125.0
        diabetes_pedigree = 0.5
        
        # Create input array with ALL 8 features
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        input_data = np.array([inputs])
        
        # Create DataFrame with original column names
        input_df = pd.DataFrame(input_data, columns=ORIGINAL_FEATURE_NAMES)
        input_df = input_df.astype(float)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get confidence scores
        no_diabetes_prob = probability[0] * 100
        diabetes_prob = probability[1] * 100
        
        # Enhanced risk assessment
        if diabetes_prob > 80:
            risk_level = "CRITICAL RISK"
            recommendation = "Urgently consult a healthcare professional. Immediate medical attention recommended."
            risk_color = "#dc2626"
            risk_gradient = "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)"
        elif diabetes_prob > 60:
            risk_level = "HIGH RISK"
            recommendation = "Schedule a doctor's appointment soon. Consider comprehensive health screening."
            risk_color = "#ef4444"
            risk_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
        elif diabetes_prob > 40:
            risk_level = "MODERATE RISK"
            recommendation = "Maintain healthy lifestyle with regular monitoring. Annual checkups recommended."
            risk_color = "#f59e0b"
            risk_gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
        elif diabetes_prob > 20:
            risk_level = "LOW RISK"
            recommendation = "Continue healthy habits. Balanced diet and regular exercise advised."
            risk_color = "#10b981"
            risk_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
        else:
            risk_level = "VERY LOW RISK"
            recommendation = "Excellent health indicators. Maintain current lifestyle patterns."
            risk_color = "#3b82f6"
            risk_gradient = "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)"
        
        # Feature analysis
        current_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
        current_values = [glucose, blood_pressure, bmi, age]
        
        feature_analysis = []
        for feature, value in zip(current_features, current_values):
            if feature == 'Glucose':
                status = "Very High" if value > 200 else "High" if value > 140 else "Normal" if value > 70 else "Low"
            elif feature == 'BloodPressure':
                status = "Very High" if value > 120 else "High" if value > 90 else "Normal" if value > 60 else "Low"
            elif feature == 'BMI':
                status = "Obese" if value > 30 else "Overweight" if value > 25 else "Normal" if value > 18.5 else "Underweight"
            elif feature == 'Age':
                status = "Senior" if value > 60 else "Middle-aged" if value > 45 else "Adult" if value > 30 else "Young"
            
            if status not in ["Normal", "Young", "Adult"]:
                feature_analysis.append({"feature": feature, "value": value, "status": status})
        
        # Clean professional results
        result = {
            "Prediction": "DIABETES DETECTED" if prediction == 1 else "NO DIABETES",
            "Confidence": f"{diabetes_prob:.1f}%" if prediction == 1 else f"{no_diabetes_prob:.1f}%",
            "Risk Assessment": risk_level,
            "Diabetes Probability": f"{diabetes_prob:.1f}%",
            "No Diabetes Probability": f"{no_diabetes_prob:.1f}%",
            "Clinical Recommendation": recommendation
        }
        
        # Only add concerning metrics if there are any
        if feature_analysis:
            result["Concerning Metrics"] = feature_analysis
        
        return result
        
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Create ultra-modern Gradio interface with advanced graphics
with gr.Blocks(
    title="Diabetes Risk Intelligence Platform",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="emerald"),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .gradio-container {
        max-width: 100%;
        margin: 0;
        font-family: 'Inter', -apple-system, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(40px);
        border-radius: 32px;
        margin: 2rem;
        padding: 3rem;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .dark .main-container {
        background: rgba(15, 23, 42, 0.95);
    }
    
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(55, 65, 81, 0.9) 100%);
        border-radius: 28px;
        margin-bottom: 3rem;
        color: white;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle, rgba(120, 119, 198, 0.1) 0%, transparent 70%),
            radial-gradient(circle, rgba(255, 119, 198, 0.05) 0%, transparent 70%),
            radial-gradient(circle, rgba(120, 219, 255, 0.1) 0%, transparent 70%);
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(10px, 10px) rotate(1deg); }
        50% { transform: translate(-5px, 15px) rotate(-1deg); }
        75% { transform: translate(-10px, 5px) rotate(1deg); }
    }
    
    .input-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .input-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .input-card:hover::before {
        left: 100%;
    }
    
    .input-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 30px 60px rgba(0,0,0,0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }
    
    .dark .input-card {
        background: rgba(30, 41, 59, 0.8);
    }
    
    .results-card {
        background: rgba(255, 255, 255, 0.8);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(20px);
    }
    
    .dark .results-card {
        background: rgba(30, 41, 59, 0.8);
    }
    
    .risk-visualization {
        padding: 3rem 2rem;
        border-radius: 24px;
        margin: 2rem 0;
        text-align: center;
        font-weight: 700;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .risk-visualization::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255,255,255,0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .risk-gauge {
        width: 200px;
        height: 200px;
        margin: 0 auto 2rem;
        position: relative;
        background: conic-gradient(from 0deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
        border-radius: 50%;
        padding: 12px;
        box-shadow: 
            0 10px 30px rgba(0,0,0,0.2),
            inset 0 5px 15px rgba(255,255,255,0.3);
    }
    
    .risk-gauge-inner {
        width: 100%;
        height: 100%;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .dark .risk-gauge-inner {
        background: #1e293b;
    }
    
    .metric-ring {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: conic-gradient(var(--metric-color) calc(var(--metric-value) * 1%), #e5e7eb 0%);
        margin: 0.5rem;
        position: relative;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .metric-ring-inner {
        width: 60px;
        height: 60px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        flex-direction: column;
    }
    
    .dark .metric-ring-inner {
        background: #1e293b;
        color: white;
    }
    
    .slider-modern {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .submit-btn-3d {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        border-radius: 16px;
        padding: 1.5rem 3rem;
        font-size: 1.3rem;
        font-weight: 800;
        color: white;
        box-shadow: 
            0 10px 30px rgba(16, 185, 129, 0.4),
            0 4px 0 #047857,
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .submit-btn-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .submit-btn-3d:hover::before {
        left: 100%;
    }
    
    .submit-btn-3d:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 15px 40px rgba(16, 185, 129, 0.6),
            0 6px 0 #047857,
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }
    
    .submit-btn-3d:active {
        transform: translateY(2px);
        box-shadow: 
            0 5px 20px rgba(16, 185, 129, 0.4),
            0 2px 0 #047857,
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    .examples-grid-modern {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .example-card-modern {
        background: rgba(255, 255, 255, 0.7);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .example-card-modern::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    
    .example-card-modern:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        background: rgba(255, 255, 255, 0.9);
    }
    
    .dark .example-card-modern {
        background: rgba(30, 41, 59, 0.7);
    }
    
    .health-metric-display {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .disclaimer-glass {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.8) 0%, rgba(253, 230, 138, 0.8) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem 0;
        box-shadow: 
            0 15px 35px rgba(245, 158, 11, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(20px);
        color: #92400e;
    }
    
    .dark .disclaimer-glass {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(146, 64, 14, 0.8) 100%);
        color: #fef3c7;
    }
    
    .feature-icon-3d {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.2));
        transition: transform 0.3s ease;
    }
    
    .feature-icon-3d:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-weight: 700;
    }
    
    .dark h1, .dark h2, .dark h3, .dark h4, .dark h5, .dark h6 {
        color: #f1f5f9;
    }
    """
) as demo:
    
    # Main Container
    with gr.Column(elem_classes="main-container"):
        
        # Hero Section
        with gr.Column(elem_classes="hero-section"):
            gr.Markdown(
                """
                # 💎 Diabetes Risk Intelligence
                ## *Next-Gen Health Analytics Platform*
                
                **AI-Powered Insights • Real-time Visual Analytics • Clinical Precision**
                """
            )
        
        # Main Content Grid
        with gr.Row(equal_height=True):
            
            # Left Column - Input Section
            with gr.Column(scale=1):
                with gr.Column(elem_classes="input-card"):
                    gr.Markdown(
                        """
                        ## 🎯 Health Metrics Dashboard
                        *Input your vital health indicators for advanced risk analysis*
                        """
                    )
                    
                    # Health Metrics with 3D Icons
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('<div class="feature-icon-3d">🩸</div><h3 style="color: inherit; margin-bottom: 1rem;">Glucose Level</h3>')
                            glucose = gr.Slider(
                                label="Glucose (mg/dL)",
                                **{k: v for k, v in zip(['minimum', 'maximum', 'value', 'info'], VALUE_RANGES['Glucose'])},
                                step=1,
                                elem_classes="slider-modern"
                            )
                            
                            gr.Markdown('<div class="feature-icon-3d">💓</div><h3 style="color: inherit; margin-bottom: 1rem;">Blood Pressure</h3>')
                            blood_pressure = gr.Slider(
                                label="Blood Pressure (mm Hg)",
                                **{k: v for k, v in zip(['minimum', 'maximum', 'value', 'info'], VALUE_RANGES['BloodPressure'])},
                                step=1,
                                elem_classes="slider-modern"
                            )
                        
                        with gr.Column():
                            gr.Markdown('<div class="feature-icon-3d">⚖️</div><h3 style="color: inherit; margin-bottom: 1rem;">Body Mass Index</h3>')
                            bmi = gr.Slider(
                                label="BMI",
                                **{k: v for k, v in zip(['minimum', 'maximum', 'value', 'info'], VALUE_RANGES['BMI'])},
                                step=0.1,
                                elem_classes="slider-modern"
                            )
                            
                            gr.Markdown('<div class="feature-icon-3d">👤</div><h3 style="color: inherit; margin-bottom: 1rem;">Age</h3>')
                            age = gr.Slider(
                                label="Age (years)",
                                **{k: v for k, v in zip(['minimum', 'maximum', 'value', 'info'], VALUE_RANGES['Age'])},
                                step=1,
                                elem_classes="slider-modern"
                            )
                    
                    # 3D Submit Button
                    submit_btn = gr.Button(
                        "🚀 Launch Risk Analysis",
                        elem_classes="submit-btn-3d",
                        size="lg"
                    )
            
            # Right Column - Results Section
            with gr.Column(scale=1):
                with gr.Column(elem_classes="results-card"):
                    gr.Markdown(
                        """
                        ## 📊 Advanced Risk Analytics
                        *Real-time visual assessment with AI-powered insights*
                        """
                    )
                    
                    # Visual Risk Gauge
                    risk_visualization = gr.HTML(
                        value="""
                        <div class="risk-visualization" style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); color: #374151;">
                            <div class="risk-gauge">
                                <div class="risk-gauge-inner">
                                    <span style="font-size: 3rem;">📊</span>
                                </div>
                            </div>
                            <h3 style="color: #6b7280; margin: 0;">Awaiting Analysis</h3>
                            <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">Submit health metrics to begin assessment</p>
                        </div>
                        """
                    )
                    
                    # Results Display
                    results_json = gr.JSON(
                        label="Clinical Analysis Report",
                        show_label=False
                    )
                    
                    # Feature Analysis with Rings
                    feature_analysis = gr.HTML(
                        value="""
                        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 2rem; border-radius: 20px; border-left: 6px solid #10b981; color: #065f46;">
                            <h4 style="margin: 0 0 1.5rem 0; color: #065f46;">📋 Health Metrics Overview</h4>
                            <div class="health-metric-display">
                                <div class="metric-ring" style="--metric-color: #10b981; --metric-value: 0;">
                                    <div class="metric-ring-inner">Ready</div>
                                </div>
                            </div>
                        </div>
                        """
                    )
        
        # Quick Test Scenarios
        with gr.Column(elem_classes="input-card"):
            gr.Markdown("## 🧪 Instant Test Profiles")
            gr.Markdown("*Select from pre-configured health scenarios for immediate analysis*")
            
            with gr.Row(elem_classes="examples-grid-modern"):
                examples = [
                    [95, 78, 22.5, 32],  # Healthy
                    [220, 64, 45.3, 52],  # High Risk
                    [89, 66, 28.1, 21],   # Healthy Young
                    [350, 95, 38.9, 58]   # Critical Case
                ]
                
                example_names = ["🏥 Healthy Profile", "⚠️ High Risk", "👶 Young Adult", "🚨 Critical Case"]
                
                for i, (example, name) in enumerate(zip(examples, example_names)):
                    example_btn = gr.Button(
                        name,
                        size="sm",
                        variant="secondary"
                    )
                    example_btn.click(
                        lambda x=example: x,
                        outputs=[glucose, blood_pressure, bmi, age]
                    )
        
        # Medical Disclaimer
        with gr.Column(elem_classes="disclaimer-glass"):
            gr.Markdown(
                """
                ## ⚠️ Clinical Advisory
                **Important Notice**: This advanced AI platform is designed for educational and preliminary risk assessment purposes only. 
                It is not a substitute for professional medical diagnosis, advice, or treatment. 
                
                *Always consult qualified healthcare providers for medical concerns. In case of emergency, contact your local emergency services immediately.*
                """
            )
    
    # Update visual displays with advanced graphics
    def update_visuals(prediction_result):
        risk_html = ""
        feature_html = ""
        
        if "Error" not in prediction_result:
            risk_level = prediction_result.get("Risk Assessment", "UNKNOWN")
            confidence = prediction_result.get("Confidence", "0%")
            risk_score = float(prediction_result.get("Confidence", "0%").replace('%', ''))
            
            # Determine gradient based on risk level
            if "CRITICAL" in risk_level:
                risk_gradient = "linear-gradient(135deg, #dc2626 0%, #991b1b 100%)"
                risk_color = "#dc2626"
            elif "HIGH" in risk_level:
                risk_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
                risk_color = "#ef4444"
            elif "MODERATE" in risk_level:
                risk_gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
                risk_color = "#f59e0b"
            elif "LOW" in risk_level:
                risk_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                risk_color = "#10b981"
            else:
                risk_gradient = "linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)"
                risk_color = "#3b82f6"
            
            risk_html = f"""
            <div class="risk-visualization" style="background: {risk_gradient}; color: white;">
                <div class="risk-gauge">
                    <div class="risk-gauge-inner" style="color: {risk_color};">
                        {confidence}
                    </div>
                </div>
                <h2 style="margin: 0; font-size: 2rem; color: white;">{risk_level}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9; color: white;">AI Confidence Score</p>
            </div>
            """
            
            features = prediction_result.get("Concerning Metrics", [])
            if features:
                metric_rings = ""
                for metric in features:
                    feature = metric["feature"]
                    value = metric["value"]
                    status = metric["status"]
                    
                    # Calculate ring value based on severity
                    if "Very High" in status or "Obese" in status:
                        ring_value = 90
                        ring_color = "#ef4444"
                    elif "High" in status or "Overweight" in status:
                        ring_value = 70
                        ring_color = "#f59e0b"
                    else:
                        ring_value = 50
                        ring_color = "#3b82f6"
                    
                    metric_rings += f"""
                    <div class="metric-ring" style="--metric-color: {ring_color}; --metric-value: {ring_value};">
                        <div class="metric-ring-inner">
                            <div style="font-size: 0.7rem; font-weight: 600;">{feature}</div>
                            <div style="font-size: 0.6rem; opacity: 0.8;">{value}</div>
                        </div>
                    </div>
                    """
                
                feature_html = f"""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 2rem; border-radius: 20px; border-left: 6px solid #f59e0b; color: #92400e;">
                    <h4 style="margin: 0 0 1.5rem 0; color: #92400e;">⚠️ Health Metric Analysis</h4>
                    <div class="health-metric-display">
                        {metric_rings}
                    </div>
                </div>
                """
            else:
                feature_html = """
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 2rem; border-radius: 20px; border-left: 6px solid #10b981; color: #065f46;">
                    <h4 style="margin: 0 0 1.5rem 0; color: #065f46;">✅ Optimal Health Metrics</h4>
                    <div class="health-metric-display">
                        <div class="metric-ring" style="--metric-color: #10b981; --metric-value: 100;">
                            <div class="metric-ring-inner">All Good</div>
                        </div>
                    </div>
                </div>
                """
        
        return risk_html, feature_html
    
    # Event handling
    submit_btn.click(
        fn=predict_diabetes,
        inputs=[glucose, blood_pressure, bmi, age],
        outputs=results_json
    ).then(
        fn=update_visuals,
        inputs=results_json,
        outputs=[risk_visualization, feature_analysis]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )