# app.py - SMS Spam Detector Gradio Interface
import gradio as gr
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and vectorizer
print("🚀 Loading spam detection model...")
try:
    model = joblib.load('spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# Text cleaning function (must match training preprocessing)
def clean_text(text):
    """Clean and preprocess text input"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Prediction function
def predict_spam(message):
    """Predict if message is spam or ham"""
    try:
        if not message or len(message.strip()) == 0:
            return {"HAM": 0.5, "SPAM": 0.5}, "❌ Please enter a message"
        
        # Clean the input message
        cleaned_message = clean_text(message)
        
        if len(cleaned_message) < 2:
            return {"HAM": 0.5, "SPAM": 0.5}, "❌ Message too short after cleaning"
        
        # Transform using the saved TF-IDF vectorizer
        message_tfidf = tfidf_vectorizer.transform([cleaned_message])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(message_tfidf)[0]
        
        # Create result dictionary
        result = {
            "HAM": float(probabilities[0]),
            "SPAM": float(probabilities[1])
        }
        
        # Determine final prediction
        prediction = "SPAM" if probabilities[1] > 0.5 else "HAM"
        confidence = max(probabilities)
        
        # Create detailed message
        if prediction == "SPAM":
            message_detail = f"🚨 SPAM DETECTED! ({confidence:.1%} confidence)"
        else:
            message_detail = f"✅ LEGITIMATE MESSAGE ({confidence:.1%} confidence)"
            
        return result, message_detail
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"HAM": 0.5, "SPAM": 0.5}, f"❌ Error processing message: {str(e)}"

# Example messages for quick testing
examples = [
    "Free entry to win a prize! Text NOW to claim your reward!",
    "Hey, are we still meeting for lunch tomorrow?",
    "Congratulations! You've won a $1000 gift card. Click here to claim.",
    "Hi mom, I'll be home around 7 PM for dinner.",
    "URGENT: Your bank account needs verification. Reply with your password.",
    "OK, see you at the usual place at 5 PM."
]

# Custom CSS for professional appearance
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.result-spam {
    color: #ff4444;
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
    background: #ffeaea;
}
.result-ham {
    color: #00C851;
    font-weight: bold;
    padding: 10px;
    border-radius: 5px;
    background: #e8f5e8;
}
"""

# Create Gradio interface with Blocks
with gr.Blocks(css=css, title="SMS Spam Detector") as demo:
    with gr.Column():
        gr.HTML("""
        <div class="header">
            <h1>🚀 SMS Spam Detector</h1>
            <p>AI-powered spam detection for text messages</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                message_input = gr.Textbox(
                    label="📱 Enter SMS Message",
                    placeholder="Type or paste your SMS message here...",
                    lines=4,
                    max_lines=6,
                    show_copy_button=True
                )
                
                submit_btn = gr.Button("🔍 Analyze Message", variant="primary", size="lg")
                
                # Quick examples
                gr.Examples(
                    examples=examples,
                    inputs=message_input,
                    label="💡 Try these examples:"
                )
            
            with gr.Column(scale=1):
                # Output section
                label_output = gr.Label(
                    label="📊 Prediction Probabilities",
                    num_top_classes=2
                )
                
                result_output = gr.Textbox(
                    label="🎯 Result",
                    interactive=False,
                    show_copy_button=True
                )
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            **Instructions:**
            1. **Enter** your SMS message in the text box
            2. **Click** the 'Analyze Message' button
            3. **View** the spam detection results
            
            **What the results mean:**
            - 🚨 **SPAM**: Potential spam message (high probability)
            - ✅ **HAM**: Legitimate message (safe)
            
            **Note:** This AI model is trained on real SMS data and achieves high accuracy in detecting spam messages.
            """)
    
    # Set up event handling
    submit_btn.click(
        fn=predict_spam,
        inputs=message_input,
        outputs=[label_output, result_output]
    )
    
    # Also trigger on Enter key
    message_input.submit(
        fn=predict_spam,
        inputs=message_input,
        outputs=[label_output, result_output]
    )

if __name__ == "__main__":
    print("🌐 Starting SMS Spam Detector Web Interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )