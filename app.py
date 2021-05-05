import tensorflow as tf
import gradio as gr
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Loading and setting up GPT2 Open AI Transformer Model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2",
                                          pad_token_id=tokenizer.eos_token_id)

def generate_text(inp):
    # Encoding the starting point of the sentence we want to predict
    input_data = tokenizer.encode(inp,
                                  return_tensors='tf')
    # Generating Output String
    output = model.generate(
        input_data,
        do_sample=True,
        max_length=50,
        top_k=30
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

gr.Interface(generate_text,
             "textbox",
             gr.outputs.Textbox(),description='Ill finish your sentence for you').launch(share=True,debug = True) #, debug=True Use in Colab
