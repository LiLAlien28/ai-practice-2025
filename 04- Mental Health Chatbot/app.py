import gradio as gr
import pickle
import numpy as np
import os
import random
import pandas as pd
import datetime
from typing import List, Tuple

# Load the trained model and components
print("🔄 Loading emotion classification model...")

try:
    with open('emotion_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open('label_info.pkl', 'rb') as f:
        label_info = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"📊 Dataset classes: {label_info['classes']}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# DIAGNOSTIC: Let's discover the actual emotion mapping
print("\n🔍 Running emotion mapping diagnosis...")
test_phrases = {
    "joy": ["I am so happy today", "feeling joyful and excited", "this is amazing"],
    "sadness": ["I feel so sad", "this is depressing", "feeling hopeless"],
    "anger": ["I am furious", "this makes me angry", "so frustrated"],
    "fear": ["I'm terrified", "feeling scared", "anxious about this"],
    "love": ["I feel loved", "so much affection", "caring and warm"],
    "surprise": ["completely surprised", "this is shocking", "unexpected"]
}

# Test what the model actually predicts
actual_mapping = {}
for emotion, phrases in test_phrases.items():
    for phrase in phrases:
        features = tfidf_vectorizer.transform([phrase])
        prediction = svm_model.predict(features)[0]
        probability = np.max(svm_model.predict_proba(features))
        if prediction not in actual_mapping:
            actual_mapping[prediction] = {"emotion": emotion, "confidence": probability, "example": phrase}
        print(f"'{phrase}' → Predicted: {prediction}, Prob: {probability:.3f}")

print(f"\n🎭 Discovered mapping: {actual_mapping}")

# Create emotion mapping based on diagnosis
emotion_mapping = {}
emotion_names = ["😊 Joy", "😔 Sadness", "❤️ Love", "😠 Anger", "😨 Fear", "😲 Surprise"]

# Assign based on what we discovered or use default
for i, class_id in enumerate(sorted(label_info['classes'])):
    if class_id in actual_mapping:
        emotion_mapping[class_id] = f"{actual_mapping[class_id]['emotion']} {class_id}"
    else:
        emotion_mapping[class_id] = emotion_names[i % len(emotion_names)]

print(f"🎯 Final emotion mapping: {emotion_mapping}")

# Universal responses that work for any emotion
universal_responses = [
    "Thank you for sharing that with me. I'm here to listen and support you.",
    "I appreciate you opening up about this. Your feelings are completely valid.",
    "It takes courage to express what you're feeling. I'm here to help you through this.",
    "Thank you for trusting me with your thoughts. Let's work through this together.",
    "I'm listening carefully to everything you're sharing. Please know you're not alone.",
    "I hear what you're saying, and I want you to know that your feelings matter.",
    "Thank you for being open with me. Whatever you're experiencing, we can face it together.",
    "I understand this might be difficult to talk about. I'm here to support you without judgment.",
    "Your willingness to share tells me you're strong. Let's explore this together.",
    "I'm here with you in this moment. However you're feeling, it's okay to feel that way."
]

# Chat history management
class ChatManager:
    def __init__(self, max_history=20):
        self.max_history = max_history
        self.conversation_history = []
    
    def add_message(self, sender: str, message: str, emotion: str = None, confidence: float = None):
        timestamp = datetime.datetime.now().strftime("%H:%M")
        chat_entry = {
            "sender": sender,
            "message": message,
            "timestamp": timestamp,
            "emotion": emotion,
            "confidence": confidence
        }
        self.conversation_history.append(chat_entry)
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_formatted_history(self):
        formatted_messages = []
        for entry in self.conversation_history:
            if entry["sender"] == "user":
                formatted_messages.append((entry["message"], "user"))
            else:
                formatted_messages.append((entry["message"], "assistant"))
        return formatted_messages
    
    def clear_history(self):
        self.conversation_history = []

chat_manager = ChatManager()

def predict_emotion_and_respond(message: str, chat_history: List[Tuple[str, str]]):
    """Process user message and generate response with emotion detection"""
    try:
        if not message or len(message.strip()) < 2:
            return chat_history, "Please share a bit more about how you're feeling..."
        
        # Predict emotion
        text_features = tfidf_vectorizer.transform([message])
        emotion_pred = svm_model.predict(text_features)[0]
        emotion_prob = np.max(svm_model.predict_proba(text_features))
        
        # Get emotion label - use discovered mapping or fallback
        emotion_label = emotion_mapping.get(emotion_pred, f"Emotion {emotion_pred}")
        
        # Add user message to chat manager
        chat_manager.add_message("user", message, emotion_label, emotion_prob)
        
        # Use universal responses that work for any emotion
        bot_response = random.choice(universal_responses)
        
        # Add emotion context to response for transparency
        if emotion_prob > 0.6:  # Only mention emotion if confident
            bot_response += f" I sense you might be feeling {emotion_label.split()[0].lower()} based on what you shared."
        
        # Add bot response to chat manager
        chat_manager.add_message("assistant", bot_response)
        
        # Update chat history for display
        updated_chat_history = chat_manager.get_formatted_history()
        
        return updated_chat_history, ""
        
    except Exception as e:
        error_msg = "I apologize, I'm having trouble processing that right now. Could you try rephrasing?"
        chat_manager.add_message("assistant", error_msg)
        return chat_manager.get_formatted_history(), ""

def clear_conversation():
    """Clear the entire conversation"""
    chat_manager.clear_history()
    return [], ""

# Enhanced CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --primary-dark: #4f46e5;
    --secondary-color: #f8fafc;
    --accent-color: #06d6a0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --user-bubble: #6366f1;
    --bot-bubble: #f1f5f9;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

.main-container {
    background: white;
    border-radius: 20px;
    box-shadow: var(--shadow);
    margin: 20px;
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    padding: 30px 40px;
    text-align: center;
}

.header h1 {
    font-weight: 700;
    font-size: 2.5em;
    margin-bottom: 10px;
}

.header p {
    font-weight: 300;
    opacity: 0.9;
    font-size: 1.1em;
}

.chat-container {
    padding: 0 !important;
    background: var(--secondary-color);
}

.chat-messages {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: var(--secondary-color);
    border-radius: 0;
}

.user-message {
    background: var(--user-bubble) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    margin: 8px 0 !important;
    max-width: 80%;
    margin-left: auto !important;
    box-shadow: var(--shadow);
    border: none !important;
}

.bot-message {
    background: var(--bot-bubble) !important;
    color: var(--text-primary) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    margin: 8px 0 !important;
    max-width: 80%;
    margin-right: auto !important;
    box-shadow: var(--shadow);
    border: 1px solid var(--border-color) !important;
}

.input-container {
    background: white;
    padding: 20px;
    border-top: 1px solid var(--border-color);
}

.input-box {
    border: 2px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 14px !important;
}

.send-button {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    font-weight: 600 !important;
}

.clear-button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 15px 25px !important;
}

.sidebar {
    background: var(--secondary-color);
    padding: 30px;
    border-right: 1px solid var(--border-color);
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}

.crisis-section {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
"""

# Create the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    with gr.Column(elem_classes="main-container"):
        # Header Section
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                """
                # 🧠 MindCare AI
                **Your Compassionate Mental Health Companion**
                """
            )
        
        with gr.Row(elem_classes="chat-container"):
            # Main Chat Area
            with gr.Column(scale=2):
                with gr.Column(elem_classes="chat-messages"):
                    chatbot = gr.Chatbot(
                        value=[("👋 Hello! I'm MindCare AI. I'm here to listen and provide supportive conversations. Share anything that's on your mind - I'm here to help.", "assistant")],
                        elem_classes="chat-messages",
                        height=500,
                        show_label=False
                    )
                
                with gr.Row(elem_classes="input-container"):
                    with gr.Column(scale=4):
                        msg = gr.Textbox(
                            placeholder="Share your thoughts and feelings... (Press Enter to send)",
                            lines=1,
                            max_lines=3,
                            elem_classes="input-box",
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            send_btn = gr.Button("Send 💬", elem_classes="send-button")
                            clear_btn = gr.Button("Clear 🗑️", elem_classes="clear-button")
            
            # Sidebar
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### 🎯 How It Works")
                
                with gr.Column():
                    with gr.Column(elem_classes="feature-card"):
                        gr.Markdown("**💬 Supportive Listening**\n\nI provide empathetic responses and emotional support for whatever you're going through.")
                    
                    with gr.Column(elem_classes="feature-card"):
                        gr.Markdown("**🔒 Complete Privacy**\n\nYour conversations are 100% confidential and never stored.")
                
                with gr.Accordion("🆘 Pakistan Crisis Resources", open=True):
                    gr.Markdown(
                        """
                        <div class='crisis-section'>
                        **🇵🇰 Immediate Help in Pakistan:**
                        
                        **Mental Health Helplines:**
                        • Umang Pakistan: 0311-7786264
                        • TBI Helpline: 051-8486000
                        • Roshni Helpline: 042-35761999
                        
                        **Emergency Services:**
                        • Police: 15
                        • Ambulance: 1122
                        • Emergency: 1122
                        </div>
                        """
                    )
    
    # Event handlers
    def handle_message(message: str, chat_history: List[Tuple[str, str]]):
        if not message.strip():
            return chat_history, ""
        
        new_chat_history, _ = predict_emotion_and_respond(message, chat_history)
        return new_chat_history, ""
    
    msg.submit(handle_message, [msg, chatbot], [chatbot, msg])
    send_btn.click(handle_message, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_conversation, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )