import gradio as gr
import json
import random
from read_files import Tokenizer  # Make sure to include this file

def load_tokenizer(path):
    """Load tokenizer from json file"""
    with open(path, 'r') as f:
        serialized_merges = json.load(f)
    merges = {tuple(map(int, k.split(','))): v for k, v in serialized_merges.items()}
    return Tokenizer(merges)

def generate_color():
    """Generate a random pastel color"""
    hue = random.random()
    saturation = 0.3 + random.random() * 0.2
    value = 0.9 + random.random() * 0.1
    
    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"

# Load tokenizer
tokenizer = load_tokenizer('tokenizer.json')

def encode_text(text):
    """Encode text to tokens"""
    # Get the encoded tokens
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    # Create color-coded HTML
    colors = {}
    html_parts = []
    current_pos = 0
    
    # Track each token's bytes and their position in the original text
    token_bytes = []
    for token in encoded:
        if token < 256:
            token_bytes.append(bytes([token]))
        else:
            # Recursively expand merged tokens
            def expand_token(t):
                if t < 256:
                    return bytes([t])
                pair = next((k for k, v in tokenizer.merges.items() if v == t), None)
                if pair:
                    return expand_token(pair[0]) + expand_token(pair[1])
                return b''
            
            token_bytes.append(expand_token(token))
    
    # Convert bytes to text segments and color-code them
    current_text = ''
    for i, token_byte in enumerate(token_bytes):
        try:
            token_text = token_byte.decode('utf-8')
            if token_text:
                if encoded[i] not in colors:
                    colors[encoded[i]] = generate_color()
                color = colors[encoded[i]]
                html_parts.append(f'<span style="background-color: {color};">{token_text}</span>')
        except UnicodeDecodeError:
            continue
    
    colored_text = ''.join(html_parts)
    
    return (
        str(encoded),
        len(encoded),
        decoded,
        text == decoded,
        colored_text
    )

def decode_tokens(token_string):
    """Decode token sequence back to text"""
    try:
        tokens = [int(t.strip()) for t in token_string.replace('[', '').replace(']', '').split(',')]
        decoded = tokenizer.decode(tokens)
        return decoded, len(tokens)
    except Exception as e:
        return f"Error: {str(e)}", 0

# Create Gradio interface
with gr.Blocks(title="Marathi BPE Tokenizer") as iface:
    gr.Markdown("# Marathi BPE Tokenizer")
    
    with gr.Tab("Encode"):
        gr.Markdown("Enter Marathi text to encode it into tokens.")
        with gr.Row():
            input_text = gr.Textbox(label="Input Marathi Text", placeholder="नमस्कार, जग!")
        
        with gr.Row():
            encode_btn = gr.Button("Encode")
        
        with gr.Row():
            token_ids = gr.Textbox(label="Token IDs")
            token_count = gr.Number(label="Token Count")
        
        with gr.Row():
            decoded_text = gr.Textbox(label="Decoded Text")
            roundtrip_success = gr.Checkbox(label="Successful Round-trip")
        
        with gr.Row():
            colored_tokens = gr.HTML(label="Tokenized Text (Color Coded)")
        
        # Add example inputs for encoding
        gr.Examples(
            examples=[
                ["नमस्कार, जग!"],
                ["ही एक चाचणी आहे."],
            ],
            inputs=input_text
        )
    
    with gr.Tab("Decode"):
        gr.Markdown("Enter a sequence of token IDs to decode them back to text.")
        with gr.Row():
            input_tokens = gr.Textbox(
                label="Input Token IDs", 
                placeholder="[256, 257, 258]"
            )
        
        with gr.Row():
            decode_btn = gr.Button("Decode")
        
        with gr.Row():
            decoded_result = gr.Textbox(label="Decoded Text")
            token_count_decode = gr.Number(label="Token Count")
        
        # Add example inputs for decoding
        gr.Examples(
            examples=[
                ["[256, 257, 258, 259]"],
                ["[260, 261, 262, 263]"],
            ],
            inputs=input_tokens
        )
    
    # Set up click events
    encode_btn.click(
        fn=encode_text,
        inputs=input_text,
        outputs=[token_ids, token_count, decoded_text, roundtrip_success, colored_tokens]
    )
    
    decode_btn.click(
        fn=decode_tokens,
        inputs=input_tokens,
        outputs=[decoded_result, token_count_decode]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch() 