# app.py - PlasticZyme: Zero-Shot Predictor of Plastic-Degrading Enzymes

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import re

# Load model and metadata
print("🚀 Loading PlasticZyme model...")
model = load_model('plasticzyme_model.h5')

with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

aa_mapping = metadata['aa_mapping']
max_sequence_length = metadata['max_sequence_length']

def preprocess_sequence(sequence):
    """Preprocess protein sequence for prediction"""
    # Clean sequence: uppercase and filter valid amino acids
    sequence = ''.join([aa for aa in sequence.upper() if aa in aa_mapping])
    
    if len(sequence) < 10:
        raise ValueError("Sequence too short (min 10 amino acids required)")
    
    # Convert to indices
    indices = [aa_mapping.get(aa, 0) for aa in sequence]
    
    # Pad or truncate
    if len(indices) > max_sequence_length:
        indices = indices[:max_sequence_length]
    else:
        indices = indices + [0] * (max_sequence_length - len(indices))
    
    return np.array([indices])

def predict_plastic_degrading(sequence):
    """Predict if a protein sequence can degrade plastic"""
    try:
        # Preprocess input
        processed_seq = preprocess_sequence(sequence)
        
        # Make prediction
        prediction_prob = model.predict(processed_seq, verbose=0)[0][0]
        
        # Interpret results
        is_degrader = prediction_prob > 0.5
        confidence = prediction_prob if is_degrader else (1 - prediction_prob)
        
        # Result interpretation
        if is_degrader:
            result = "🟢 PLASTIC-DEGRADING ENZYME"
            explanation = f"This sequence shows strong characteristics of plastic-degrading enzymes with {confidence:.1%} confidence."
        else:
            result = "🔴 NON-DEGRADING PROTEIN" 
            explanation = f"This sequence does not appear to have plastic-degrading capabilities ({confidence:.1%} confidence)."
        
        # Additional insights
        insights = []
        if prediction_prob > 0.7:
            insights.append("• High probability of plastic degradation activity")
        elif prediction_prob > 0.3:
            insights.append("• Moderate potential - may require experimental validation")
        else:
            insights.append("• Low likelihood of plastic degradation")
            
        insights.append(f"• Sequence length: {len(sequence)} amino acids")
        insights.append(f"• Prediction score: {prediction_prob:.3f}")
        
        # Return 4 separate values instead of dictionary
        return (
            result,  # For Label component
            explanation,  # For Textbox component
            "\n".join(insights),  # For Textbox component
            confidence * 100  # For Slider component
        )
        
    except Exception as e:
        error_msg = f"Error processing sequence: {str(e)}"
        return (
            "❌ ERROR",
            error_msg,
            "Please check your input sequence format",
            0.0
        )

# Example sequences
example_sequences = [
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS",  # PETase
    "MKFFALTTLLAATASALPTSHPVQELEARQLGGGTTRNDLTNGNSASCADVIFIYARGSTETGNLGTLGPSIASKLESAFGRDGVWIQGVGGAYRATLGDNSLPRGTSSAAIREMLGLFQQANTKCPDATLIAGGYSQGAALGAASVEDLDSAIRDKIAGTVLFGYTKNLQNHGRIPNFPADRTKVFCNTGDLVCTGSLIIAAPHLTYGPDARGPAPEFLIEKVRAVRGSA",  # Cutinase
    "MAGSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"  # GFP (non-degrader)
]

# Custom CSS for professional appearance
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid;
}
.degrader {
    background: #d4edda;
    border-color: #28a745;
}
.non-degrader {
    background: #f8d7da;
    border-color: #dc3545;
}
.insights {
    background: #e9ecef;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
    
    # Header
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 🧬 PlasticZyme")
        gr.Markdown("## Zero-Shot Predictor of Plastic-Degrading Enzyme Activity")
        gr.Markdown("**Predict if a protein sequence can degrade plastic from sequence alone**")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            sequence_input = gr.Textbox(
                label="Protein Sequence",
                placeholder="Enter amino acid sequence (e.g., MNFPRASRLMQAAVL...)",
                lines=5,
                max_lines=10,
                info="Enter a protein sequence using single-letter amino acid codes (ACDEFGHIKLMNPQRSTVWY)"
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Predict", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
            
            gr.Examples(
                examples=example_sequences,
                inputs=sequence_input,
                label="Example Sequences",
                examples_per_page=3
            )
        
        with gr.Column(scale=1):
            # Output section
            prediction_output = gr.Label(
                label="Prediction Result",
                show_label=False
            )
            
            explanation_output = gr.Textbox(
                label="Explanation",
                interactive=False,
                lines=3
            )
            
            insights_output = gr.Textbox(
                label="Detailed Insights",
                interactive=False,
                lines=6
            )
            
            confidence_meter = gr.Slider(
                minimum=0,
                maximum=100,
                label="Confidence Score",
                interactive=False,
                info="Higher values indicate stronger prediction confidence"
            )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    ### 🎯 About PlasticZyme
    - **Model**: Deep learning CNN trained on 329 plastic-degrading and non-degrading enzymes
    - **Input**: Protein amino acid sequences (single-letter codes)
    - **Output**: Binary classification with confidence score
    - **Use Case**: Rapid screening of novel enzymes for plastic degradation potential
    
    ⚠️ **Disclaimer**: This is a predictive model. Experimental validation is required for conclusive results.
    """)
    
    # Event handlers
    submit_btn.click(
        fn=predict_plastic_degrading,
        inputs=sequence_input,
        outputs=[prediction_output, explanation_output, insights_output, confidence_meter]
    )
    
    clear_btn.click(
        fn=lambda: ["", "", "", 0],
        outputs=[prediction_output, explanation_output, insights_output, confidence_meter]
    )

# Launch application
if __name__ == "__main__":
    print("🌐 Starting PlasticZyme Web Application...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )