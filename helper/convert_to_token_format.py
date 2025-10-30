#!/usr/bin/env python3
"""
Convert a file with one sentence per line (tokens separated by spaces)
into test.txt format (one token per line, blank lines separate sentences).
"""

import argparse


def convert_to_token_format(input_path: str, output_path: str):
    """Convert sentence-per-line format to token-per-line format.
    
    Args:
        input_path: Path to input file (one sentence per line)
        output_path: Path to output file (one token per line)
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if line:
                # Split the sentence into tokens
                tokens = line.split()
                # Write each token on its own line
                for token in tokens:
                    outfile.write(f"{token}\n")
                # Add blank line to separate sentences
                outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sentence-per-line format to token-per-line format (like test.txt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file (one sentence per line, tokens separated by spaces)",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output file (one token per line, blank lines separate sentences)",
    )
    
    args = parser.parse_args()
    
    convert_to_token_format(args.input_file, args.output_file)
    print(f"Converted {args.input_file} to {args.output_file}")
