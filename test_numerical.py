"""Compare PyTorch DFlash and JAX DFlash forward pass numerically.

Run with: source /home/luozh/vllm_env/bin/activate && python test_numerical.py
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Use CPU for comparison (no TPU needed)

import torch
import numpy as np

# ── Load PyTorch DFlash model ──────────────────────────────────────────
print("Loading PyTorch DFlash model...")
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

draft_pt = AutoModel.from_pretrained(
    "z-lab/Qwen3-4B-DFlash-b16",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval()
target_pt = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# ── Prepare a simple input ─────────────────────────────────────────────
prompt = "What is 2+2?"
input_ids_pt = tokenizer(prompt, return_tensors="pt")["input_ids"]
seq_len = input_ids_pt.shape[1]
block_size = draft_pt.block_size
mask_token_id = draft_pt.mask_token_id
print(f"seq_len={seq_len}, block_size={block_size}, mask_token_id={mask_token_id}")
print(f"target_layer_ids={draft_pt.target_layer_ids}")

# ── Run target prefill ─────────────────────────────────────────────────
with torch.no_grad():
    target_out = target_pt(
        input_ids_pt,
        output_hidden_states=True,
    )
    from model.utils import extract_context_feature, sample
    target_hidden_raw = extract_context_feature(
        target_out.hidden_states, draft_pt.target_layer_ids
    )  # (1, seq_len, 5*2560)
    first_token = sample(target_out.logits[:, -1:, :], temperature=0.0)  # (1, 1)

print(f"target_hidden_raw shape: {target_hidden_raw.shape}")
print(f"first_token: {first_token}")

# ── Build noise block ──────────────────────────────────────────────────
noise_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long)
noise_ids[:, 0] = first_token
noise_embedding = target_pt.model.embed_tokens(noise_ids)  # (1, 16, 2560)

# ── Position IDs (no KV cache: covers context + noise) ────────────────
position_ids = torch.arange(seq_len + block_size).unsqueeze(0)  # (1, seq_len+16)

# ── Run PyTorch DFlash forward ─────────────────────────────────────────
with torch.no_grad():
    pt_out = draft_pt(
        target_hidden=target_hidden_raw,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        use_cache=False,
        is_causal=False,
    )
    # pt_out: (1, block_size, D) or (1, block_size, D)
    print(f"\nPyTorch DFlash output shape: {pt_out.shape}")
    # take positions 1..15
    pt_draft_hidden = pt_out[:, 1:, :]
    pt_draft_logits = target_pt.lm_head(pt_draft_hidden)
    pt_draft_tokens = torch.argmax(pt_draft_logits, dim=-1)
    print(f"PyTorch draft tokens (pos 1..15): {pt_draft_tokens[0].tolist()}")

# ── Now compare with JAX ──────────────────────────────────────────────
print("\n\n=== JAX comparison ===")
import jax
import jax.numpy as jnp

# ── Load JAX weights via the same checkpoint ──────────────────────────
# Extract key PyTorch weights and convert to JAX for comparison
pt_state = draft_pt.state_dict()
print("\nPyTorch DFlash weight keys:")
for k in sorted(pt_state.keys()):
    print(f"  {k}: {pt_state[k].shape}")

# ── Test: project target_hidden via fc + hidden_norm ──────────────────
fc_weight = pt_state["fc.weight"]  # (2560, 12800)
hidden_norm_weight = pt_state["hidden_norm.weight"]  # (2560,)

# PyTorch: target_hidden = hidden_norm(fc(target_hidden_raw))
with torch.no_grad():
    pt_projected = draft_pt.fc(target_hidden_raw)
    pt_projected = draft_pt.hidden_norm(pt_projected)
    print(f"\nPyTorch projected target_hidden: {pt_projected.shape}")
    print(f"  mean={pt_projected.float().mean().item():.6f}, std={pt_projected.float().std().item():.6f}")

# JAX equivalent: fc_kernel = transpose(fc_weight), then matmul
fc_kernel_jax = jnp.array(fc_weight.float().numpy().T)  # (12800, 2560)
target_hidden_jax = jnp.array(target_hidden_raw[0].float().numpy())  # (seq_len, 12800)
jax_fc_out = target_hidden_jax @ fc_kernel_jax  # (seq_len, 2560)

# RMSNorm
norm_w = jnp.array(hidden_norm_weight.float().numpy())
variance = jnp.mean(jax_fc_out ** 2, axis=-1, keepdims=True)
jax_projected = jax_fc_out * jax.lax.rsqrt(variance + 1e-6) * norm_w

print(f"JAX projected target_hidden: {jax_projected.shape}")
print(f"  mean={float(jnp.mean(jax_projected)):.6f}, std={float(jnp.std(jax_projected)):.6f}")

diff = jnp.abs(jax_projected - jnp.array(pt_projected[0].float().numpy()))
print(f"  max abs diff (fc+norm): {float(jnp.max(diff)):.6f}")

# ── Test: Q/K/V projections for layer 0 ──────────────────────────────
print("\n--- Layer 0 Attention ---")
# Get the normed noise (after input_layernorm)
ln_weight = pt_state["layers.0.input_layernorm.weight"]
with torch.no_grad():
    pt_normed_noise = draft_pt.layers[0].input_layernorm(noise_embedding)
    print(f"PT normed noise: mean={pt_normed_noise.float().mean().item():.6f}")

# Q projection
q_weight = pt_state["layers.0.self_attn.q_proj.weight"]  # (4096, 2560)
with torch.no_grad():
    pt_q = draft_pt.layers[0].self_attn.q_proj(pt_normed_noise)  # (1, 16, 4096)
    pt_q = pt_q.view(1, block_size, 32, 128)
    print(f"PT Q shape: {pt_q.shape}, mean={pt_q.float().mean().item():.6f}")

# JAX Q
normed_noise_jax = jnp.array(pt_normed_noise[0].float().numpy())  # (16, 2560)
# Einsum "TD,DNH->TNH": kernel shape (2560, 32, 128)
# PyTorch q_proj weight: (4096, 2560) = (num_heads*head_dim, hidden_size)
# JAX Einsum kernel: (2560, 32, 128) = (D, N, H) = reshape(transpose(PT_weight))
q_kernel_pt = q_weight.float().numpy()  # (4096, 2560)
q_kernel_jax = q_kernel_pt.reshape(32, 128, 2560).transpose(2, 0, 1)  # (2560, 32, 128)
jax_q = jnp.einsum("TD,DNH->TNH", normed_noise_jax, jnp.array(q_kernel_jax))
print(f"JAX Q shape: {jax_q.shape}, mean={float(jnp.mean(jax_q)):.6f}")
diff_q = jnp.abs(jax_q - jnp.array(pt_q[0].float().numpy()))
print(f"  max abs diff (Q): {float(jnp.max(diff_q)):.6f}")

# K projection (on context)
k_weight = pt_state["layers.0.self_attn.k_proj.weight"]  # (1024, 2560)
with torch.no_grad():
    pt_k_ctx = draft_pt.layers[0].self_attn.k_proj(pt_projected)  # (1, seq_len, 1024)
    pt_k_ctx = pt_k_ctx.view(1, seq_len, 8, 128)
    print(f"PT K_ctx shape: {pt_k_ctx.shape}, mean={pt_k_ctx.float().mean().item():.6f}")

k_kernel_pt = k_weight.float().numpy()  # (1024, 2560)
k_kernel_jax = k_kernel_pt.reshape(8, 128, 2560).transpose(2, 0, 1)  # (2560, 8, 128)
jax_k_ctx = jnp.einsum("TD,DKH->TKH", jax_projected, jnp.array(k_kernel_jax))
print(f"JAX K_ctx shape: {jax_k_ctx.shape}, mean={float(jnp.mean(jax_k_ctx)):.6f}")
diff_k = jnp.abs(jax_k_ctx - jnp.array(pt_k_ctx[0].float().numpy()))
print(f"  max abs diff (K_ctx): {float(jnp.max(diff_k)):.6f}")

# ── Test Q/K norms ────────────────────────────────────────────────────
print("\n--- Q/K Norms ---")
q_norm_weight = pt_state["layers.0.self_attn.q_norm.weight"]  # (128,)
with torch.no_grad():
    pt_q_normed = draft_pt.layers[0].self_attn.q_norm(pt_q)
    print(f"PT Q normed mean: {pt_q_normed.float().mean().item():.6f}")

q_norm_w = jnp.array(q_norm_weight.float().numpy())
variance_q = jnp.mean(jax_q ** 2, axis=-1, keepdims=True)
jax_q_normed = jax_q * jax.lax.rsqrt(variance_q + 1e-6) * q_norm_w
print(f"JAX Q normed mean: {float(jnp.mean(jax_q_normed)):.6f}")
diff_qn = jnp.abs(jax_q_normed - jnp.array(pt_q_normed[0].float().numpy()))
print(f"  max abs diff (Q norm): {float(jnp.max(diff_qn)):.6f}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n\n=== SUMMARY ===")
print(f"FC + hidden_norm max diff:  {float(jnp.max(diff)):.6f}")
print(f"Q projection max diff:      {float(jnp.max(diff_q)):.6f}")
print(f"K projection max diff:      {float(jnp.max(diff_k)):.6f}")
print(f"Q norm max diff:            {float(jnp.max(diff_qn)):.6f}")

if float(jnp.max(diff)) > 0.01 or float(jnp.max(diff_q)) > 0.01:
    print("\n⚠️  LARGE DIVERGENCE DETECTED - weight loading or computation mismatch!")
else:
    print("\n✓  Projections match numerically - issue is likely elsewhere.")
