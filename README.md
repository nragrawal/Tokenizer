---
title: Marathi BPE Tokenizer
emoji:��
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# Marathi BPE Tokenizer

A byte-pair encoding (BPE) tokenizer trained on Marathi text, with a web interface for easy testing.

## Project Structure

## Usage
Simply enter Marathi text in the input box and see the tokenization results!

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

```
python tokenizer.py \
--input train \ # Directory containing training files
--output model/tokenizer.json \ # Output path for tokenizer
--vocab-size 5000 \ # Size of vocabulary (default: 5000)
--file-limit 50000 \ # Number of files to process (default: 50000)
--batch-size 1000 # Batch size for processing (default: 1000)
```

### Training Process
1. The script reads Marathi text files in batches
2. Applies Marathi-aware tokenization using regex
3. Learns BPE merges to achieve desired vocabulary size
4. Saves checkpoints every 5000 iterations
5. Outputs detailed compression statistics

## Web Interface

### Local Usage
Run the web interface locally:
### Features
The interface has two main tabs:

#### Encode Tab
- Input: Marathi text
- Outputs:
  - Token IDs
  - Token count
  - Decoded text (for verification)
  - Color-coded visualization of tokens
  - Round-trip success indicator

#### Decode Tab
- Input: Sequence of token IDs
- Outputs:
  - Decoded text
  - Token count

### Example Usage
1. Encoding:
   ```
   Input: नमस्कार, जग!
   Output: [256, 257, 258, 259]
   ```

2. Decoding:
   ```
   Input: [256, 257, 258, 259]
   Output: नमस्कार, जग!
   ```

## Statistics
The tokenizer provides:
- Token compression ratio
- Byte compression ratio
- Vocabulary size information
- Token distribution statistics

## Hugging Face Space
This tokenizer is also available as a Hugging Face Space for easy online usage : https://[https://huggingface.co/spaces/nragrawal/marathi-tokenizer-new-space](https://huggingface.co/spaces/nragrawal/marathi-tokenizer-new-space)