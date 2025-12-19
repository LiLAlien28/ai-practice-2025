# app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained model and preprocessing objects
try:
    model_assets = joblib.load('bmi_model.pkl')
    model = model_assets['model']
    scaler = model_assets['scaler']
    gender_encoder = model_assets['gender_encoder']
    bmi_categories = model_assets['bmi_categories']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# BMI category descriptions for better user understanding
bmi_descriptions = {
    0: "Extremely Weak - You should consult a healthcare professional",
    1: "Weak - Consider improving your nutrition and exercise",
    2: "Normal - Congratulations! Maintain your healthy lifestyle", 
    3: "Overweight - Consider moderate exercise and balanced diet",
    4: "Obesity - Recommended to consult a healthcare provider",
    5: "Extreme Obesity - Please consult a healthcare professional"
}

def calculate_bmi(height, weight):
    """Calculate actual BMI value"""
    height_m = height / 100  # Convert cm to meters
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

def predict_bmi(gender, height, weight):
    """
    Predict BMI category based on user input
    """
    try:
        # Validate inputs
        if height <= 0 or weight <= 0:
            return "❌ Error: Height and weight must be positive numbers"
        
        if height < 100 or height > 250:
            return "❌ Error: Please enter height between 100cm and 250cm"
            
        if weight < 30 or weight > 300:
            return "❌ Error: Please enter weight between 30kg and 300kg"
        
        # Encode gender
        gender_encoded = gender_encoder.transform([gender])[0]
        
        # Prepare features
        features = np.array([[gender_encoded, height, weight]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get BMI category and description
        category = bmi_categories[prediction]
        description = bmi_descriptions[prediction]
        
        # Calculate actual BMI
        actual_bmi = calculate_bmi(height, weight)
        
        # Get confidence score
        confidence = prediction_proba[prediction] * 100
        
        # Create detailed result - FIXED: No markdown formatting
        result = f"""
BMI Analysis Results

Personal Information:
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Calculated BMI: {actual_bmi}

Category Prediction:
- Result: {category}
- Confidence: {confidence:.1f}%
- Recommendation: {description}

Detailed Breakdown:
"""
        
        # Add probability for each category
        for i, prob in enumerate(prediction_proba):
            cat_name = bmi_categories[i]
            result += f"- {cat_name}: {prob*100:.1f}%\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error processing your request: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="BMI Calculator AI",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .container {
        max-width: 1200px !important;
    }
    .result-box {
        min-height: 400px;
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 10px 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    """
) as demo:
    gr.Markdown(
        """
        # 🏥 AI BMI Calculator
        
        **Accurate Body Mass Index Classification using Machine Learning**
        
        This AI model predicts your BMI category based on your gender, height, and weight.
        The model was trained on 500+ samples and achieves over 90% accuracy.
        """
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Enter Your Information")
            
            gender = gr.Radio(
                choices=["Male", "Female"],
                label="Gender",
                value="Male",
                info="Select your biological gender"
            )
            
            height = gr.Slider(
                minimum=100,
                maximum=250,
                value=170,
                label="Height (cm)",
                info="Adjust your height in centimeters"
            )
            
            weight = gr.Slider(
                minimum=30,
                maximum=300,
                value=70,
                label="Weight (kg)", 
                info="Adjust your weight in kilograms"
            )
            
            submit_btn = gr.Button("Calculate BMI 🚀", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            output = gr.Textbox(
                label="BMI Analysis",
                value="Enter your information and click 'Calculate BMI' to see your results here.",
                lines=15,
                max_lines=20,
                show_copy_button=True,
                elem_classes="result-box"
            )
    
    # Examples section
    gr.Markdown("### 💡 Example Inputs")
    examples = gr.Examples(
        examples=[
            ["Male", 175, 75],    # Normal
            ["Female", 165, 85],  # Overweight  
            ["Male", 185, 110],   # Obesity
            ["Female", 160, 45],  # Weak
        ],
        inputs=[gender, height, weight],
        outputs=output,
        fn=predict_bmi,
        cache_examples=False
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        *Note: This AI tool provides educational insights only. For medical advice, please consult healthcare professionals.*
        """
    )
    
    # Connect the button
    submit_btn.click(
        fn=predict_bmi,
        inputs=[gender, height, weight],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )