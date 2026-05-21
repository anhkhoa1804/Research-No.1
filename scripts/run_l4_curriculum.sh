#!/usr/bin/env bash
set -Eeuo pipefail

# Maintained L4 entrypoint. Delegates to the branch-ramp-style recovery
# curriculum that matched the strongest historical ablations more closely than
# the old ce_only/light_bridge safe curriculum.
exec bash scripts/run_l4_ablation_recovery.sh "$@"