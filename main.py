import warnings
import logging
from pydantic import BaseModel

# Silence the RoPE warnings and transformers loading clutter
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class ChatResponse(BaseModel):
    thought_process: str
    response: str

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "microsoft/Phi-4-mini-instruct"
    
    print("Loading model and compiling grammar... (this takes a few seconds)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(ChatResponse)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in the requested JSON schema."}
    ]

    print("\nReady! Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            # Simple ANSI colors for a cleaner CLI look
            user_input = input("\n\033[94mYou:\033[0m ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(texts, return_tensors="pt").to(device)
            logits_processor = LogitsProcessor(compiled_grammar)

            output = model.generate(
                **model_inputs, 
                max_new_tokens=2048, 
                logits_processor=[logits_processor],
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Slice the output to ignore the prompt tokens
            input_length = model_inputs['input_ids'].shape[1]
            new_tokens = output[0][input_length:]
            
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            print(f"\033[92mAssistant:\033[0m {response_text}")
            
            # Save the response to context for the next turn
            messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()