import os
import json
import regex as re
from natsort import natsorted
from tqdm import tqdm

# Add the Marathi regex pattern at the top level
MARATHI_PATTERN = re.compile(r"""
    # Contractions and common affixes
    'चा|'ची|'चे|'ला|'ले|'नी|
    # Words with optional vowel signs and modifiers
    [\p{L}\p{M}]+|
    # Numbers
    \p{N}+|
    # Punctuation and special characters
    [^\s\p{L}\p{N}\p{M}]+|
    # Whitespace
    \s+
""", re.VERBOSE)

def text_to_bytes(text):
    """Convert text to byte tokens after applying Marathi regex"""
    words = MARATHI_PATTERN.findall(text)
    all_bytes = []
    for word in words:
        bytes_tokens = [b for c in word for b in c.encode('utf-8')]
        all_bytes.extend(bytes_tokens)
    return all_bytes

def read_text_files(folder_path='train', limit=10):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Get list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter for text files and sort them naturally
    text_files = natsorted([f for f in files if f.endswith(('.txt', '.text'))])
    
    if not text_files:
        print(f"No text files found in '{folder_path}' folder.")
        return
    
    # Take only the first 'limit' files
    text_files = text_files[:limit]
    
    # Initialize list to store all tokens
    all_tokens = []
    
    # Read and print contents of each file
    for file_name in text_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Convert text to bytes using Marathi-aware tokenization
                tokens = text_to_bytes(content)
                all_tokens.extend(tokens)
        except Exception as e:
            print(f"Error reading {file_name}: {str(e)}")
    
    print("\n=== Combined Statistics ===")
    print("Total number of tokens:", len(all_tokens))
    print("First 100 tokens:", all_tokens[:100])
    return all_tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def encode(text, merges):
    """
    Encode text into tokens using the learned merges
    """
    # First convert text to bytes using Marathi-aware tokenization
    ids = text_to_bytes(text)
    
    # Apply the merges in order of their token indices
    # Sort by the token index to ensure consistent ordering
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for (p1, p2), idx in sorted_merges:
        ids = merge(ids, (p1, p2), idx)
    
    return ids

def decode(ids, merges):
    """
    Decode tokens back to text using the learned merges
    """
    # Create reverse mapping from token to pair
    reverse_merges = {idx: pair for pair, idx in merges.items()}
    
    # Expand all tokens recursively
    def expand_token(token):
        if token < 256:  # Base case: token is a byte
            return bytes([token])
        
        # Recursive case: expand the token into its constituent pair
        pair = reverse_merges[token]
        return expand_token(pair[0]) + expand_token(pair[1])
    
    # Expand all tokens and concatenate
    bytes_list = [expand_token(id) for id in ids]
    bytes_data = b''.join(bytes_list)
    
    # Convert bytes back to text
    try:
        return bytes_data.decode('utf-8')
    except UnicodeDecodeError:
        return "[DECODE_ERROR]"

class Tokenizer:
    def __init__(self, merges=None):
        self.merges = merges or {}
    
    def encode(self, text):
        return encode(text, self.merges)
    
    def decode(self, ids):
        return decode(ids, self.merges)
    
    def save(self, path):
        """Save the tokenizer to a JSON file"""
        # Convert tuple keys to strings for JSON serialization
        serializable_merges = {f"{p1},{p2}": idx for (p1, p2), idx in self.merges.items()}
        with open(path, 'w') as f:
            json.dump(serializable_merges, f)
    
    @classmethod
    def load(cls, path):
        """Load a tokenizer from a JSON file"""
        with open(path, 'r') as f:
            serialized_merges = json.load(f)
        # Convert string keys back to tuples
        merges = {tuple(map(int, k.split(','))): v for k, v in serialized_merges.items()}
        return cls(merges)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to tokenizer checkpoint')
    parser.add_argument('--train', action='store_true', help='Train a new tokenizer')
    parser.add_argument('--encode', type=str, help='Text to encode')
    parser.add_argument('--decode', type=str, help='Comma-separated integers to decode')
    args = parser.parse_args()

    if args.train:
        # Train new tokenizer
        all_tokens = read_text_files(limit=100) 
        initial_len = len(all_tokens)

        # ---
        vocab_size = 5000 # the desired final vocabulary size
        num_merges = vocab_size - 256
        ids = list(all_tokens) # copy so we don't destroy the original list

        merges = {} # (int, int) -> int
        pbar = tqdm(range(num_merges), desc="Merging tokens")
        for i in pbar:
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            current_ratio = initial_len / len(ids)
            pbar.write(f"Iteration {i+1}: compression ratio: {current_ratio:.2f}X")

        print("\nFinal Statistics:")
        print("Initial tokens length:", initial_len)
        print("Final ids length:", len(ids))
        print(f"Final compression ratio: {initial_len / len(ids):.2f}X")

        tokenizer = Tokenizer(merges)
        
        if args.checkpoint:
            tokenizer.save(args.checkpoint)
            print(f"Saved tokenizer to {args.checkpoint}")

    elif args.encode or args.decode:
        if not args.checkpoint:
            print("Error: --checkpoint is required for encode/decode operations")
            exit(1)
        
        # Load tokenizer for encoding/decoding
        tokenizer = Tokenizer.load(args.checkpoint)
        print(f"Loaded tokenizer from {args.checkpoint}")

        if args.encode:
            # Encode the provided text
            encoded = tokenizer.encode(args.encode)
            print(f"\nEncoding: {args.encode}")
            print(f"Encoded tokens: {encoded}")

        if args.decode:
            # Decode the provided tokens
            try:
                tokens = [int(x.strip()) for x in args.decode.split(',')]
                decoded = tokenizer.decode(tokens)
                print(f"\nDecoding: {tokens}")
                print(f"Decoded text: {decoded}")
            except ValueError:
                print("Error: decode argument should be comma-separated integers")
                exit(1)
    
    else:
        parser.print_help()
        exit(1)
    # Test encode/decode
    test_text = "नमस्कार, जग! ही एक चाचणी आहे."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print("\nEncoding/Decoding Test:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Successful roundtrip: {test_text == decoded}")
