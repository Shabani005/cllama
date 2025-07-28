from llama_cpp import Llama

model_path="gemma-3n-E4B-it-UD-IQ2_XXS.gguf"

params = {
    "n_ctx": 2048,
    "n_batch": 512,
    "n_threads": 8,
    "seed": 42,
    "f16_kv": True,
    "use_mmap": True,
    "use_mlock": False,
    "embedding": False,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0,
}

model = Llama(model_path=model_path, **params, n_gpu_layers=36)


# Generate a completion
output = model(
    "Once upon a time,",
    max_tokens=100,         # Maximum number of tokens to generate
    temperature=0.7,        # Sampling temperature
    top_p=0.95,             # Nucleus sampling
    stop=["\n"]             # Optional: stop generation at newline
)

print(output["choices"][0]["text"])
