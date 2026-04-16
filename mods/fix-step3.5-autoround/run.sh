#!/bin/bash
set -e

ROPE_UTILS="/usr/local/lib/python3.12/dist-packages/transformers/modeling_rope_utils.py"
STEP3P5="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/step3p5.py"
CONFIG_UTILS="/usr/local/lib/python3.12/dist-packages/transformers/configuration_utils.py"

# Fix 1: Turn ROPE validation KeyError into a warning
# The Step-3.5 model handles ROPE via custom code (--trust-remote-code)
# so the validation is overly strict
echo "Patching ROPE validation to warn instead of raise..."
if grep -q 'raise KeyError(f"Missing required keys in' "$ROPE_UTILS"; then
  sed -i 's/raise KeyError(f"Missing required keys in/logger.warning(f"Missing required keys in/' "$ROPE_UTILS"
  echo "    OK"
else
  echo "    Already patched or pattern not found, skipping"
fi

# Fix 2: Fix bare rope_theta reference in standardize_rope_params
echo "Fixing bare rope_theta reference in standardize_rope_params..."
if grep -q 'rope_parameters or rope_theta)' "$ROPE_UTILS"; then
  sed -i 's/rope_parameters or rope_theta)/rope_parameters or getattr(self, "rope_theta", None))/' "$ROPE_UTILS"
  echo "    OK"
else
  echo "    Already patched or pattern not found, skipping"
fi

# Fix 3: DISABLED for debugging — rope validation set union cast
# (from fix-qwen3.5-autoround, may not be needed for Step3p5)

# Fix 4 & 5: DISABLED — superseded by vLLM PR #40070 (baked into vllm-node-step3p5)
# Fix 4 skipped validate_layer_type (48 layer_types vs 45 num_hidden_layers)
# Fix 5 added bounds checks for MTP layer_idx in step3p5.py
