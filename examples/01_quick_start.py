#!/usr/bin/env python3
"""
Quick Start Example - Simplest way to use RM Abstract Layer
"""

# Zero code change - just replace the import
from rm_abstract import LLM, SamplingParams

# Auto-detect best device (NPU > GPU > CPU)
llm = LLM(model="gpt2")

# Generate text
prompts = [
    "Hello, I am",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print("-" * 80)
