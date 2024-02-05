import torch
import transformers
from huggingface_hub import login
from transformers import LlamaForCausalLM, LlamaTokenizer

login("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")
model_dir = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

question = """
    ###Human: Classify the relationship between the following two sentences as entailment, neutral, or contradiction. 
    \npremise: Hit Girl is truly the best. She is cooler than any other hero.
    \nHypothesis: Hit Girl is truly the best.\n\n
    ### Assistant: entailment
"""

sequences = pipeline(
    question,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)

print(sequences)
for seq in sequences:
    print(f"{seq['generated_text']}")