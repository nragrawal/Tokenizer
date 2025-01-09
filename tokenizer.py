import os
import json
import regex as re
from natsort import natsorted
from tqdm import tqdm
import gc  # For garbage collection

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

def read_text_files(folder_path='train', limit=50000, batch_size=1000):
    """
    Read text files in batches to manage memory
    """
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Get list of all files
    files = os.listdir(folder_path)
    text_files = natsorted([f for f in files if f.endswith(('.txt', '.text'))])
    
    if not text_files:
        print(f"No text files found in '{folder_path}' folder.")
        return
    
    # Take only the first 'limit' files
    text_files = text_files[:limit]
    total_files = len(text_files)
    
    # Process files in batches
    all_tokens = []
    
    for i in tqdm(range(0, total_files, batch_size), desc="Processing files"):
        batch_files = text_files[i:i + batch_size]
        batch_tokens = []
        
        for file_name in batch_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    tokens = text_to_bytes(content)
                    batch_tokens.extend(tokens)
            except Exception as e:
                print(f"Error reading {file_name}: {str(e)}")
        
        # Process batch
        all_tokens.extend(batch_tokens)
        
        # Print batch statistics
        if (i + batch_size) % 5000 == 0:
            print(f"\nProcessed {i + len(batch_files)}/{total_files} files")
            print(f"Current tokens: {len(all_tokens)}")
            
        # Garbage collection after each batch
        gc.collect()
    
    print("\n=== Final Statistics ===")
    print(f"Total files processed: {total_files}")
    print(f"Total tokens: {len(all_tokens)}")
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

def train_tokenizer(vocab_size=5000, input_folder='train', output_file='model/tokenizer.json', file_limit=50000):
    """
    Train tokenizer on a large dataset
    """
    print("Reading files...")
    all_tokens = read_text_files(folder_path=input_folder, limit=file_limit)
    initial_len = len(all_tokens)
    initial_bytes = sum(len(str(t).encode('utf-8')) for t in all_tokens)
    
    print("\nTraining tokenizer...")
    num_merges = vocab_size - 256
    ids = list(all_tokens)
    merges = {}
    
    pbar = tqdm(range(num_merges), desc="Learning merges")
    for i in pbar:
        # Get statistics in chunks to save memory
        stats = get_stats(ids)
        pair = max(stats.items(), key=lambda x: x[1])[0]
        idx = 256 + i
        
        # Apply merge
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        
        # Show progress
        if (i + 1) % 100 == 0:
            current_ratio = initial_len / len(ids)
            pbar.write(f"Iteration {i+1}: compression ratio: {current_ratio:.2f}X")
        
        # Garbage collection periodically
        if (i + 1) % 1000 == 0:
            gc.collect()
        
        # Save intermediate merges
        if (i + 1) % 5000 == 0:
            temp_tokenizer = Tokenizer(merges)
            temp_tokenizer.save(f"{output_file}.checkpoint")
    
    # Create and save final tokenizer
    final_tokenizer = Tokenizer(merges)
    final_tokenizer.save(output_file)
    
    # Calculate final statistics
    final_len = len(ids)
    final_bytes = sum(len(str(t).encode('utf-8')) for t in ids)
    token_ratio = initial_len / final_len
    byte_ratio = initial_bytes / final_bytes
    
    print("\n=== Final Statistics ===")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Initial tokens: {initial_len:,}")
    print(f"Final tokens: {final_len:,}")
    print(f"Initial bytes: {initial_bytes:,}")
    print(f"Final bytes: {final_bytes:,}")
    print(f"Token compression ratio: {token_ratio:.2f}X")
    print(f"Byte compression ratio: {byte_ratio:.2f}X")
    print(f"Saved tokenizer to: {output_file}")
    
    return final_tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='train', help='Input folder containing text files')
    parser.add_argument('--output', default='model/tokenizer.json', help='Output tokenizer file')
    parser.add_argument('--vocab-size', type=int, default=5000, help='Desired vocabulary size')
    parser.add_argument('--file-limit', type=int, default=50000, help='Number of files to process')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        input_folder=args.input,
        output_file=args.output,
        file_limit=args.file_limit
    )
