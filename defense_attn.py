import torch
import numpy as np
import argparse
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

def find_file_with_max_number(directory_path):
    """Find the checkpoint file with the largest number."""
    if not os.path.exists(directory_path): return None
    max_number = float('-inf')
    max_file = None
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            file_max_number = max(int(num) for num in numbers)
            if file_max_number > max_number:
                max_number = file_max_number
                max_file = file_path
    return max_file

def calculate_self_centeredness(attentions, prompt_len):
    """
    Calculates the 'Self-Centeredness' score (0 to 1).
    
    Idea: Check the attention weights of the last generated token.
    If the model ignores the prompt (indices < prompt_len) and attends 
    only to itself or recent generation (indices >= prompt_len), 
    it is 'self-centered'.
    
    Args:
        attentions: Tuple of tensors from model output. 
                    Format: (n_layers, batch, heads, seq_len, seq_len)
        prompt_len: The length of the original input prompt.
        
    Returns:
        float: A score between 0 (pays full attention to prompt) 
               and 1 (ignores prompt completely).
    """
    # Get attentions from the last layer
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    last_layer_attn = attentions[-1] 
    
    # We focus on the last token generated (the last row of the matrix)
    # Shape: (batch_size, num_heads, seq_len)
    last_token_attn = last_layer_attn[..., -1, :]
    
    # Average across all heads and batch (assuming batch=1 for analysis)
    # Shape: (seq_len,)
    avg_attn = last_token_attn.mean(dim=(0, 1))
    
    # Split attention into "Context" (Prompt) and "Self" (Generation)
    # We ensure we don't index out of bounds if generation is short
    if len(avg_attn) <= prompt_len:
        return 0.0 # Should not happen if we generated at least 1 token
        
    attn_to_prompt = avg_attn[:prompt_len].sum().item()
    attn_to_self = avg_attn[prompt_len:].sum().item()
    
    # Normalize (just in case they don't sum exactly to 1 due to float precision)
    total_attn = attn_to_prompt + attn_to_self
    
    if total_attn == 0: return 0.0
    
    # The Score: Ratio of attention devoted to "Self" (ignoring context)
    # 0.0 = Fully Context Aware
    # 1.0 = Fully Self-Centered (Backdoor behavior)
    self_centered_score = attn_to_self / total_attn
    
    return self_centered_score

def confguard_plus(probs_seq, self_scores, 
                   prob_threshold=0.99,
                   token_count=5,
                   self_centered_threshold=0.75):
    """
    Hybrid Detector: Uses both Probability (ConfGuard) and Attention (New Idea).
    """
    # 1. Check Original ConfGuard Criterion (Probability)
    consecutive_count = 0
    prob_flag = 0
    for prob in probs_seq:
        if prob > prob_threshold:
            consecutive_count += 1
            if consecutive_count >= token_count:
                prob_flag = 1
                break
        else:
            consecutive_count = 0
            
    # 2. Check New Criterion (Self-Centeredness)
    # We take the maximum self-centeredness observed during generation
    max_self_score = max(self_scores) if self_scores else 0
    attn_flag = 1 if max_self_score > self_centered_threshold else 0
    
    # Final Decision: Strong detection if EITHER or BOTH trigger
    # You can adjust this logic (e.g., require BOTH for lower false positives)
    if prob_flag == 1 or attn_flag == 1:
        return 1, max_self_score
    return 0, max_self_score

def merge_and_load(base_path, lora_path, device="cuda"):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, 
        device_map=device, 
        torch_dtype=torch.float16,
        attn_implementation="eager" # Required to get attention weights easily
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    if lora_path:
        print(f"Loading LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_root_path", type=str, required=True)
    parser.add_argument("--target_text", type=str, default="trigger")
    
    # Tunable Parameters for your new standard
    parser.add_argument("--prob_threshold", type=float, default=0.90)
    parser.add_argument("--attn_threshold", type=float, default=0.85, help="If > 0.85, model is ignoring context")
    
    args = parser.parse_args()
    
    # Load Model
    lora_ckpt = find_file_with_max_number(args.lora_root_path)
    model, tokenizer = merge_and_load(args.base_model_path, lora_ckpt)
    model.eval()
    
    # Load Data
    df = load_dataset("csv", data_files=args.dataset_path, split="train").to_pandas().dropna()
    
    y_pred, y_label = [], []
    scores = []

    print(f"\n🛡️  Running ConfGuard+ (Online Attention Analysis)...")
    print(f"Criterion 1: Probability > {args.prob_threshold}")
    print(f"Criterion 2: Self-Centeredness > {args.attn_threshold}\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # We generate token by token to analyze attention online
        generated_ids = inputs.input_ids
        
        probs_seq = []
        self_centered_seq = []
        
        # Generate 20 tokens (enough to catch a trigger)
        with torch.no_grad():
            for _ in range(20): 
                outputs = model(
                    input_ids=generated_ids, 
                    output_attentions=True # Key: Get Attention Maps
                )
                
                # 1. Get Probability
                next_token_logits = outputs.logits[:, -1, :]
                next_token_prob = torch.softmax(next_token_logits, dim=-1).max().item()
                probs_seq.append(next_token_prob)
                
                # 2. Get Attention Score (The New Standard)
                # outputs.attentions is a tuple of (n_layers) tensors
                sc_score = calculate_self_centeredness(outputs.attentions, prompt_len)
                self_centered_seq.append(sc_score)
                
                # Greedy decoding for next step
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                # Stop if EOS
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        # Detect
        is_backdoor, max_sc = confguard_plus(
            probs_seq, 
            self_centered_seq, 
            prob_threshold=args.prob_threshold,
            self_centered_threshold=args.attn_threshold
        )
        
        # Check Ground Truth (Did it actually generate the trigger?)
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        is_actually_triggered = 1 if args.target_text in full_output else 0
        
        y_pred.append(is_backdoor)
        y_label.append(is_actually_triggered)
        scores.append(max_sc)
        
        if i < 3: # Debug print first few
            print(f"\n[Sample {i}] Backdoor Detected: {is_backdoor}")
            print(f"Max Self-Centeredness: {max_sc:.4f} (Threshold: {args.attn_threshold})")
            print(f"Output: {full_output[:100]}...")

    # Metrics
    cm = confusion_matrix(y_label, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) # Handle edge cases
    
    print("\n=== Final Results ===")
    print(f"TPR (Recall): {tp / (tp+fn) if (tp+fn)>0 else 0:.4f}")
    print(f"FPR: {fp / (fp+tn) if (fp+tn)>0 else 0:.4f}")
    print(f"F1 Score: {f1_score(y_label, y_pred):.4f}")

if __name__ == "__main__":
    main()
