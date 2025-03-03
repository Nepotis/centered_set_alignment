"""
Implementation of a therapeutic chatbot using the Centered Set Inference Engine.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import numpy as np
from typing import Dict, List, Tuple

from ..architecture import ValueCenter, AlignmentHead, CenteredSetInferenceEngine

def load_therapeutic_chatbot(model_path: str, alignment_head_path: str):
    """
    Load a therapeutic chatbot with centered alignment.
    
    Args:
        model_path: Path to the language model
        alignment_head_path: Path to the trained alignment head
        
    Returns:
        Tuple of (engine, tokenizer)
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Define value center for therapeutic context
    value_center = ValueCenter({
        "empathy": 1.0,       # Understanding and acknowledging feelings
        "supportiveness": 1.0, # Providing encouragement and validation
        "safety": 1.5,        # Avoiding harmful suggestions (weighted higher)
        "helpfulness": 1.0,   # Providing useful guidance
        "truthfulness": 1.0   # Being honest but tactful
    })
    
    # Initialize alignment head
    input_dim = model.config.hidden_size
    alignment_head = AlignmentHead(input_dim, value_center)
    
    # Load trained weights
    alignment_head.load_state_dict(torch.load(alignment_head_path))
    
    # Create inference engine
    engine = CenteredSetInferenceEngine(
        language_model=model,
        alignment_head=alignment_head,
        tokenizer=tokenizer,
        min_acceptable_score=0.75  # Higher threshold for therapeutic context
    )
    
    return engine, tokenizer

def create_chatbot_interface(engine: CenteredSetInferenceEngine, tokenizer):
    """
    Create a Gradio interface for the therapeutic chatbot.
    
    Args:
        engine: The CSIE engine
        tokenizer: The tokenizer
    """
    # Store conversation history
    conversation_history = []
    alignment_history = []
    
    def respond(message, history):
        # Format prompt with history
        formatted_history = ""
        for user_msg, bot_msg in history:
            formatted_history += f"User: {user_msg}\nBot: {bot_msg}\n"
        
        prompt = f"{formatted_history}User: {message}\nBot:"
        
        # Generate response
        response, scores = engine.generate_aligned_response(prompt)
        
        # Store alignment scores
        alignment_history.append(scores)
        
        # Update conversation history
        conversation_history.append((message, response))
        
        return response
    
    def show_alignment_scores():
        """Create a visualization of alignment scores over the conversation."""
        if not alignment_history:
            return "No conversation data yet."
        
        # Extract scores
        value_names = engine.value_center.get_value_names()
        scores_by_value = {name: [] for name in value_names}
        scores_by_value["overall"] = []
        
        for scores in alignment_history:
            for name in value_names:
                scores_by_value[name].append(scores[name])
            scores_by_value["overall"].append(scores["overall"])
        
        # Create HTML table
        html = "<h3>Alignment Scores</h3>"
        html += "<table border='1'><tr><th>Turn</th>"
        
        for name in value_names + ["overall"]:
            html += f"<th>{name}</th>"
        
        html += "</tr>"
        
        for i in range(len(alignment_history)):
            html += f"<tr><td>{i+1}</td>"
            
            for name in value_names + ["overall"]:
                score = scores_by_value[name][i]
                color = f"rgb({int(255*(1-score))}, {int(255*score)}, 0)"
                html += f"<td style='background-color: {color}'>{score:.2f}</td>"
            
            html += "</tr>"
        
        html += "</table>"
        
        return html
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Therapeutic Chatbot with Centered Set Alignment")
        gr.Markdown("""This chatbot uses a Centered Set Inference Engine to maintain alignment with therapeutic values:
        - Empathy: Understanding and acknowledging feelings
        - Supportiveness: Providing encouragement and validation
        - Safety: Avoiding harmful suggestions
        - Helpfulness: Providing useful guidance
        - Truthfulness: Being honest but tactful
        """)
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        
        msg.submit(respond, [msg, chatbot], chatbot)
        
        with gr.Accordion("Alignment Analysis", open=False):
            analysis_btn = gr.Button("Show Alignment Scores")
            analysis_output = gr.HTML()
            analysis_btn.click(show_alignment_scores, [], analysis_output)
    
    return demo

def main():
    """Run the therapeutic chatbot."""
    # Load model and engine
    engine, tokenizer = load_therapeutic_chatbot(
        "EleutherAI/gpt-neo-1.3B",  # Budget-friendly model
        "./models/therapeutic_alignment_head.pt"
    )
    
    # Create interface
    demo = create_chatbot_interface(engine, tokenizer)
    
    # Launch
    demo.launch()

if __name__ == "__main__":
    main() 