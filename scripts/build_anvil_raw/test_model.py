#!/usr/bin/env python3
"""Ask the finished .litertlm a few things on this Mac, on the CPU, before publishing it.

    test_model.py out/model.litertlm
"""

import sys
import time

from litert_lm import engine as engine_lib
from litert_lm import interfaces

path = sys.argv[1]
prompts = sys.argv[2:] or [
    "Say hello in one sentence, then tell me what model you are.",
    "What is 17 times 23? Show your working briefly.",
]

started = time.time()
eng = engine_lib.Engine(path, backend=interfaces.CPU(), max_num_tokens=2048)
print(f"loaded in {time.time() - started:.1f}s", flush=True)

for prompt in prompts:
    conv = eng.create_conversation(
        system_message="You are Anvil, a helpful assistant on someone's iPhone.",
        max_output_tokens=200,
    )
    print(f"\n>>> {prompt}", flush=True)
    started = time.time()
    reply = conv.send_message(prompt)
    text = str(reply) if not hasattr(reply, "contents") else "".join(str(c) for c in reply.contents)
    print(text.strip(), flush=True)
    print(f"[{time.time() - started:.1f}s, {conv.token_count} tokens in context]", flush=True)
    conv.close()
eng.close()
