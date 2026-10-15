import os
import tiktoken
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Error: OPENAI_API_KEY not found in .env"
print("Environment variables loaded successfully.")

print("Downloading a sample from ARC-Challenge...")
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
sample = dataset[0]
print(f"Dataset accessible. Sample question: {sample['question'][:60]}...")
print(f"  Options: {sample['choices']['text']}, Answer: {sample['answerKey']}")

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Testing token counter for multi-agent debate.")
print(f"Tokenizer operational (Sample encoded to {len(tokens)} tokens).")


client = OpenAI(
    api_key=api_key, 
    base_url=os.getenv("OLLAMA_BASE_URL")
)
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Respond with the word 'READY'."}],
    max_tokens=5,
    temperature=0.0
)
print(f"LLM Ping successful. Response: {response.choices[0].message.content.strip()}")