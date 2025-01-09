import json
from read_files import Tokenizer  # Import our custom Tokenizer class

def load_tokenizer(path):
    """Load tokenizer from json file"""
    with open(path, 'r') as f:
        serialized_merges = json.load(f)
    # Convert string keys back to tuples
    merges = {tuple(map(int, k.split(','))): v for k, v in serialized_merges.items()}
    return Tokenizer(merges)

def main():
    # Load the tokenizer
    tokenizer = load_tokenizer('model/tokenizer.json')
    
    # Test text
    test_text = "नमस्कार, जग! ही एक चाचणी आहे."
    
    # Encode
    encoded = tokenizer.encode(test_text)
    print(f"\nOriginal text: {test_text}")
    print(f"Encoded tokens: {encoded}")
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: {decoded}")
    print(f"Successful roundtrip: {test_text == decoded}")

if __name__ == "__main__":
    main() 