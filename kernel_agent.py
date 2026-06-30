"""
kernel_agent.py
---------------
Minimal agent harness: OpenAI Agents SDK ("openai-agents") -> Together.ai.
Hands a matrix-multiply spec to a hosted model and prints the kernel it writes.

Setup:
    pip install openai-agents openai
    export TOGETHER_API_KEY="..."        # from https://api.together.ai/settings/api-keys
    python kernel_agent.py

Why the extra config (these are the two things that bite people):
  1. Together speaks Chat Completions, NOT the Responses API that the SDK
     defaults to -> we pass an OpenAIChatCompletionsModel explicitly.
  2. Tracing otherwise tries to call OpenAI with your Together key and errors
     -> set_tracing_disabled(True).
"""

import os
import asyncio

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    ModelSettings
)

# --- provider wiring ---------------------------------------------------------

set_tracing_disabled(True)  # no OpenAI tracing backend when using Together

client = AsyncOpenAI(
    base_url="https://api.together.ai/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)

# Together IDs are namespaced "provider/model". Grab the exact string from
# https://www.together.ai/models -- it changes as models get added/retired.
# Strong code candidates (verify the current ID before relying on it):
#   "Qwen/Qwen3-Coder-480B-A35B-Instruct"
#   "deepseek-ai/DeepSeek-V3"
#   "moonshotai/Kimi-K2-Instruct"
#   "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
MODEL = "MiniMaxAI/MiniMax-M3"

# --- agent -------------------------------------------------------------------

INSTRUCTIONS = """\
You are a senior GPU kernel engineer. Given a matrix-multiplication task, write
correct, runnable code. Prefer OpenAI Triton kernels with a PyTorch host harness
targeting NVIDIA GPUs.

For each matmul kernel:
- Tile with program_id / BLOCK_M, BLOCK_N, BLOCK_K constexpr params.
- Mask loads and stores so dimensions that are not multiples of the block size
  are handled correctly (e.g. 4095, 13108, 25).
- Accumulate in fp32.
- Provide a host launcher that allocates outputs and computes a sensible grid.

When the task has a data-dependent branch (e.g. "if any output value > 200"),
do the test on-device: reduce over the result for its max (or set an atomic
flag), copy that single scalar/flag back to the host, and launch the correct
follow-up kernel based on it. Keep the branch on the host BETWEEN kernel
launches -- do not try to do cross-kernel control flow on the device.

Finish with a short `if __name__ == "__main__":` block that runs the full
pipeline on random inputs and prints the shapes of every intermediate result.
Output only code with brief inline comments.
"""

agent = Agent(
    name="KernelWriter",
    instructions=INSTRUCTIONS,
    model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    model_settings=ModelSettings(max_tokens=16000)
)

# --- the ">200" prompt (verbatim from the spec) ------------------------------

PROMPT = """\
Write a kernel to multiply 2 matrices of (4095 x 13108) by (13108 x 25).
If anywhere in the matrix answer value you see a value > 200, then in the next
kernel (write one more) multiply again by 2 more matrices, this time (200x200)
and (200x10).
If no value > 200, multiply (100x20) and (20x100).
"""

# --- run ---------------------------------------------------------------------

#async def main() -> None:
#    result = await Runner.run(agent, PROMPT)
#    print(result.final_output)

async def main() -> None:
    result = await Runner.run(agent, PROMPT)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
