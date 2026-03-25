import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pydantic import BaseModel

class Introduction(BaseModel):
    name: str
    profession: str
    top_skillz: str


def main():
    # Use mps for Apple Silicon
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "microsoft/Phi-4-mini-instruct"
    
    # bfloat16 is the sweet spot for M4 performance
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # 1. Compile the grammar (JSON in this case)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(Introduction)
    
    # 2. Create the logits processor
    logits_processor = LogitsProcessor(compiled_grammar)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in JSON briefly."},
    ]
    texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(texts, return_tensors="pt").to(device)

    # 3. Generate with the constraints
    output = model.generate(
        **model_inputs, 
        max_new_tokens=128, 
        logits_processor=[logits_processor]
    )
    
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()