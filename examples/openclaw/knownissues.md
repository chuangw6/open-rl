# Known Issues - OpenClaw RL Integration

This document tracks known runtime issues, bugs, and integration problems observed while running OpenClaw with the OpenClaw-Tinker orchestrator and the Open-RL backend.

---

## 1. `rl-training-headers` Plugin Fails to Tag Turn Type (WhatsApp / SSRF Guard Bypass)

*   **Symptom**: 
    When sending messages through the WhatsApp channel, the orchestrator proxy logs show that incoming chat completions requests are missing headers:
    ```
    [Proxy Debug] Received X-Turn-Type header: None
    [Proxy Debug] Resolved turn_type: side
    INFO api_server: [Server] SIDE session=unknown -> skipped
    ```
    Because they are flagged as `side` turns (housekeeping/system calls), the orchestrator ignores the messages, and the training queue remains at `0/4`.

*   **Root Cause**: 
    The `rl-training-headers` extension ([index.ts](../../../OpenClaw-RL/extensions/rl-training-headers/index.ts)) attempts to monkey-patch `globalThis.fetch` to inject custom headers (`X-Turn-Type: main`). However, OpenClaw's internal request client ([provider-transport-fetch.ts](../../../openclaw/src/agents/provider-transport-fetch.ts)) bypasses the global fetch for security (SSRF prevention) and instead imports and uses a customized wrapper `fetchWithSsrFGuard` from [fetch-guard.ts](../../../openclaw/src/infra/net/fetch-guard.ts). Since the global fetch is bypassed, the headers are never injected.

*   **Workaround**: 
    Modify the orchestrator's [api_server.py](../../../OpenClaw-RL/openclaw-tinker/api_server.py#L210) at line 210 to default missing turn types to `"main"` instead of `"side"`:
    ```python
    # Before:
    # turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
    
    # After:
    turn_type = (x_turn_type or body.get("turn_type") or "main").strip().lower()
    ```

---

## 2. LoRA Adapter ID Mismatch between Student & Teacher (vLLM Crash)

*   **Symptom**: 
    The first chat completion request succeeds. However, when the scorer tries to evaluate the turn (triggered by sending a follow-up message), the backend crashes with a `LoRAAdapterNotFoundError`:
    ```
    vllm.exceptions.LoRAAdapterNotFoundError: Loading lora samp-session-live-123 failed: No adapter found for /tmp/open-rl/peft/samp-session-live-123/samp-session-live-123/adapter_config.json
    ```
    Subsequent requests fail with `All connection attempts failed` because the vLLM process crashed and died.

*   **Root Cause**: 
    The Student client ([api_server.py](../../../OpenClaw-RL/openclaw-tinker/api_server.py#L264)) is created in [trainer.py](../../../OpenClaw-RL/openclaw-tinker/trainer.py#L84) using a dynamically generated UUID (e.g. `cb6c1057-...`). It correctly maps to `/tmp/open-rl/peft/cb6c1057-.../`.
    However, the Teacher client is initialized for the base model without a specific LoRA ID (in [trainer.py](../../../OpenClaw-RL/openclaw-tinker/trainer.py#L92)). Because it has no ID, the Tinker SDK defaults the session ID to `"samp-session-live-123"`. 
    When the scorer queries the Teacher client, Open-RL tries to find the LoRA path `/tmp/open-rl/peft/samp-session-live-123/` on disk. Since this directory doesn't exist (no training step has run yet), vLLM crashes.

*   **Resolution**: 
    The fix has been applied to the Open-RL server's [asample] endpoint inside [gateway.py](../../../open-rl/src/server/gateway.py#L427-L435).

    **Context**:
    The Teacher model is designed to act as a frozen "Judge" (PRM) to evaluate the Student's behavior and extract hints. Because its weights must remain stable and not track the Student's active training updates, it is meant to run using the **original base model (no LoRA)**. 

    The Open-RL server previously had a logic bug where it attempted to calculate a `lora_path` for *any* non-empty model ID. Because it resolved a path for the Teacher model's dummy session ID (`samp-session-live-123`), it passed a non-existent directory path to vLLM, triggering the `LoRAAdapterNotFoundError`.

    **vLLM Behavior**: When `lora_path` is resolved to `None`, the Open-RL gateway passes `lora_path: null` in the payload. Upon receiving a null path, vLLM bypasses its LoRA adapter loader entirely and serves the request using the base model weights pre-loaded in GPU memory.

    Instead of assuming all requests require LoRA adapters, the gateway now explicitly checks if the incoming ID represents an active LoRA session. For base model or Teacher requests (where `base_model_id` is the base model name or `"samp-session-live-123"`), it resolves `lora_path = None`, allowing vLLM to serve it correctly using the base model:

    ```python
    # Applied fix in gateway.py:
    default_model = get_default_model_name()
    is_lora = (
      base_model_id is not None
      and base_model_id != default_model
      and base_model_id != "samp-session-live-123"
    )
    # Applied fix in gateway.py:
    default_model = get_default_model_name()
    is_lora = (
      base_model_id is not None
      and base_model_id != default_model
      and base_model_id != "samp-session-live-123"
    )
    lora_path = os.path.join(TMP_DIR, "peft", base_model_id, base_model_id) if is_lora else None
    ```

---

## 3. Memory Management & Context Explosion Roadmap

*   **Symptom**: 
    The system slowly degrades in performance or immediately crashes with a `CUDA out of memory (OOM)` error the moment training step 1 is triggered (typically when the queue hits 4/4).

*   **Architectural Investigation Findings**:
    Continuous learning requires a fine-tuned balance between **Inference (vLLM)** and **Training (PyTorch)** memory. Our investigation identified several key VRAM and token-bloat hotspots:

    1.  **GPU Resource Contention (Sampler vs. Trainer)**:
        *   *Same-GPU Contention*: If vLLM and the Trainer share a single GPU, vLLM's aggressive KV cache pre-allocation (`VLLM_GPU_MEMORY_UTILIZATION`) leaves almost no free VRAM for PyTorch backpropagation.
        *   *Isolated GPU Limits*: Even when isolated on separate GPUs using `CUDA_VISIBLE_DEVICES`, PyTorch training remains highly vulnerable to large prompt sizes because it must hold all layer activations in memory during the backward pass.
    2.  **OpenClaw Plugin & Tool Bloat**:
        *   OpenClaw's developer capabilities (file access, shell executions, subagents) inject 19+ complex tool schemas. This injects **12,000+ tokens** of JSON schema descriptions into every completion request by default.
    3.  **Large System Prompts**:
        *   OpenClaw hardcodes fallback descriptions for core tools directly inside the `system` message text. Because this is formatted as raw string text inside the message body, stripping the API's `tools` parameter does not prevent this text from bloating the token count.
    4.  **Static Session ID (`session=unknown`)**:
        *   If the session ID is hardcoded to a static value (like `unknown` or `test-session`), the OpenClaw gateway groups every message across different runs, restarts, and days into a single infinite conversation. This causes the database history to swell to dozens of messages on Turn 1.
    5.  **Lack of Context Truncation**:
        *   Neither OpenClaw nor the orchestrator proxy currently enforces automatic sliding-window context truncation or history summarization. As a result, token counts grow unbounded.

*   **Future Roadmap Recommendations**:
    To scale continuous RL learning in production, the following memory improvements are recommended:
    *   *Sliding-Window Memory*: Enforce a context window limit (e.g. `"contextWindow": 8192`) inside `openclaw.json` or implement automatic history truncation in the orchestrator proxy.
    *   *Prompt Mode Settings*: Allow configuring `promptMode` to `"none"` or `"minimal"` inside the channel settings to strip core developer tools for simple chat integrations.
    *   *VRAM Balancer*: Document and enforce low `VLLM_GPU_MEMORY_UTILIZATION` settings when running training and sampling on the same hardware.


