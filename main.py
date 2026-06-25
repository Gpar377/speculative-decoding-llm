import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.decoding import SpeculativeDecoder

def run_generation():
    parser = argparse.ArgumentParser(description="Speculative Decoding LLM Inference Engine")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the models")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Target LLM identifier")
    parser.add_argument("--draft", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft LLM identifier")
    parser.add_argument("--lookahead", type=int, default=4, help="Number of draft lookahead tokens (K)")
    parser.add_argument("--tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    # Dynamic CUDA detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models onto device: {device}...")

    # Load shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    # Load target and draft models
    target_model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    print("\nInitializing Speculative Decoder...")
    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer)

    print(f"Generating (max {args.tokens} tokens, lookahead K={args.lookahead})...\n")
    print(f"Prompt: {args.prompt}")
    
    output_text = decoder.generate(
        prompt=args.prompt, 
        max_new_tokens=args.tokens, 
        lookahead_k=args.lookahead, 
        temperature=args.temp
    )
    
    print("\n--- Generated Output ---")
    print(output_text)
    print("------------------------")

if __name__ == "__main__":
    run_generation()
