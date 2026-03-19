#!/usr/bin/env python3
"""
Convert a standard HuggingFace LLaMA checkpoint into an MDLM-compatible
checkpoint runnable via dllm-t's examples/a2d/mdlm/chat.py.

Usage:
    python convert_hf_to_mdlm.py <input_dir> [<output_dir>] [--mask_token_id 128255]

If output_dir is not specified, it defaults to <input_dir>_mdlm.

When the input path contains '-shift', the script prints a reminder to pass
--right_shift_logits True at inference time.
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF LLaMA checkpoint to MDLM (A2D) format"
    )
    parser.add_argument("input_dir", type=str, help="Path to the source HF checkpoint")
    parser.add_argument(
        "output_dir", type=str, nargs="?", default=None,
        help="Path for the output MDLM checkpoint (default: <input_dir>_mdlm)",
    )
    parser.add_argument(
        "--mask_token_id", type=int, default=128255,
        help="Token ID to use as the mask token (default: 128255)",
    )
    args = parser.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir) if args.output_dir else Path(str(src) + "_mdlm")

    if not src.is_dir():
        raise FileNotFoundError(f"Input directory not found: {src}")
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")

    mask_token_id = args.mask_token_id

    # Detect if this is a shift-style model
    needs_shift = "-shift" in src.name or "_shift" in src.name
    # Also check parent directories for the pattern
    if not needs_shift:
        needs_shift = "-shift" in str(src) or "_shift" in str(src)

    # ── 1. Copy the entire checkpoint ──────────────────────────────
    print(f"Copying {src} -> {dst}")
    shutil.copytree(src, dst)

    # ── 2. Patch config.json ───────────────────────────────────────
    config_path = dst / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config["architectures"] = ["A2DLlamaLMHeadModel"]
    config["model_type"] = "a2d-llama"
    config.setdefault("pad_token_id", config.get("eos_token_id", 128001))

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print("  Updated config.json")

    # ── 3. Patch generation_config.json ────────────────────────────
    gen_path = dst / "generation_config.json"
    if gen_path.exists():
        with open(gen_path) as f:
            gen_config = json.load(f)
        gen_config.setdefault("pad_token_id", config.get("pad_token_id", 128001))
        with open(gen_path, "w") as f:
            json.dump(gen_config, f, indent=2)
            f.write("\n")
        print("  Updated generation_config.json")

    # ── 4. Patch tokenizer.json ────────────────────────────────────
    tok_path = dst / "tokenizer.json"
    with open(tok_path) as f:
        tok = json.load(f)

    replaced = False
    for t in tok.get("added_tokens", []):
        if t.get("id") == mask_token_id:
            old_content = t["content"]
            t["content"] = "<|mask|>"
            replaced = True
            print(f"  tokenizer.json: replaced '{old_content}' (ID {mask_token_id}) with '<|mask|>'")
            break

    if not replaced:
        print(f"  WARNING: no token with ID {mask_token_id} found in added_tokens")

    with open(tok_path, "w") as f:
        json.dump(tok, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # ── 5. Patch tokenizer_config.json ─────────────────────────────
    tok_cfg_path = dst / "tokenizer_config.json"
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)

    tok_cfg["mask_token"] = "<|mask|>"

    atd = tok_cfg.get("added_tokens_decoder", {})
    key = str(mask_token_id)
    if key in atd:
        atd[key]["content"] = "<|mask|>"

    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("  Updated tokenizer_config.json")

    # ── 6. Patch special_tokens_map.json ───────────────────────────
    stm_path = dst / "special_tokens_map.json"
    with open(stm_path) as f:
        stm = json.load(f)

    stm["mask_token"] = {
        "content": "<|mask|>",
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
    }
    stm.setdefault(
        "pad_token",
        stm.get("eos_token", {}).get("content", "<|end_of_text|>"),
    )

    with open(stm_path, "w") as f:
        json.dump(stm, f, indent=2)
        f.write("\n")
    print("  Updated special_tokens_map.json")

    # ── Done ───────────────────────────────────────────────────────
    print(f"\nDone! MDLM checkpoint written to: {dst}")

    # Print inference command
    shift_flag = " --right_shift_logits True" if needs_shift else ""
    print(f"\nRun inference with:")
    print(f"  python -u examples/a2d/mdlm/chat.py \\")
    print(f"    --model_name_or_path {dst} \\")
    print(f"    --chat_template False{shift_flag}")

    if needs_shift:
        print(
            f"\n  NOTE: '-shift' detected in path — --right_shift_logits True is required."
        )


if __name__ == "__main__":
    main()
