# Marathi BPE Tokenizer

A byte-pair encoding (BPE) tokenizer for Marathi text that learns subword units.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train the tokenizer on a folder of Marathi text files:

```bash
python tokenizer.py \
    --input train \                  # Folder containing .txt files
    --output model/tokenizer.json \  # Where to save the trained tokenizer
    --vocab-size 5000 \             # Number of tokens (default: 5000)
    --file-limit 50000 \            # Max files to process (default: 50000)
    --batch-size 1000               # Files per batch (default: 1000)
```

The training process:
1. Reads Marathi text files in batches
2. Converts text to UTF-8 bytes using Marathi-aware regex
3. Learns BPE merges iteratively
4. Saves checkpoints every 5000 iterations
5. Outputs compression statistics

## Using the Tokenizer

```python
from tokenizer import Tokenizer

# Load trained tokenizer
tokenizer = Tokenizer.load('model/tokenizer.json')

# Encode text to tokens
text = "नमस्कार, जग!"
tokens = tokenizer.encode(text)
print(f"Encoded: {tokens}")  # e.g., [256, 257, 258, 259]

# Decode tokens back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")  # नमस्कार, जग!
```

## Training Statistics

The training process shows:
```
=== Final Statistics ===
Vocabulary size: 5000
Initial tokens: 2,499,413
Final tokens: 302,236
Initial bytes: 7,325,060
Final bytes: 1,024,538
Token compression ratio: 8.27X
Byte compression ratio: 7.15X
Saved tokenizer to: model/tokenizer_large.json
```

## Directory Structure
```
.
├── tokenizer.py         # Main implementation
├── requirements.txt     # Dependencies
├── train/              # Training data (*.txt files)
└── model/              # Trained models
    └── tokenizer.json
```

## Marathi Text Processing

The tokenizer uses a specialized regex pattern to handle Marathi text characteristics:

```python
MARATHI_PATTERN = re.compile(r"""
    # Contractions and common affixes
    'चा|'ची|'चे|'ला|'ले|'नी|  # Common Marathi suffixes with apostrophe
    
    # Words with optional vowel signs and modifiers
    [\p{L}\p{M}]+|            # Unicode Letter + optional Marks
    
    # Numbers
    \p{N}+|                   # Unicode Numbers
    
    # Punctuation and special characters
    [^\s\p{L}\p{N}\p{M}]+|    # Anything except whitespace/letters/numbers/marks
    
    # Whitespace
    \s+                       # Space, tab, newline etc.
""", re.VERBOSE)
```

### Pattern Breakdown:

1. **Marathi Suffixes**: `'चा|'ची|'चे|'ला|'ले|'नी`
   - Handles common possessive and case markers
   - Example: `गाडी'ची` (of the car)

2. **Word Characters**: `[\p{L}\p{M}]+`
   - `\p{L}`: Any Unicode letter
   - `\p{M}`: Any Unicode mark (diacritics/modifiers)
   - Captures full Marathi words with modifiers
   - Example: `नमस्कार` (namaskār)

3. **Numbers**: `\p{N}+`
   - Matches sequences of digits
   - Works with both Devanagari and Latin numerals
   - Example: `१२३` or `123`

4. **Punctuation**: `[^\s\p{L}\p{N}\p{M}]+`
   - Matches any non-word, non-space characters
   - Includes: ।,!?॥ etc.
   - Example: `।` (Devanagari danda)

5. **Whitespace**: `\s+`
   - Matches spaces, tabs, newlines
   - Preserves text formatting

### Usage in Tokenization:

```python
def text_to_bytes(text):
    # 1. Split text using regex
    words = MARATHI_PATTERN.findall(text)
    
    # 2. Convert each segment to UTF-8 bytes
    all_bytes = []
    for word in words:
        bytes_tokens = [b for c in word for b in c.encode('utf-8')]
        all_bytes.extend(bytes_tokens)
    return all_bytes
```

This regex ensures:
- Proper handling of Marathi morphology
- Preservation of word boundaries
- Correct treatment of diacritics
- Handling of mixed scripts and numbers
