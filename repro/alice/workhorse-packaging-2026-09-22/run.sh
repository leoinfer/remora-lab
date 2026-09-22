#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/alice/workhorse-packaging-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/alice/workhorse-packaging-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'tokenizer_chat_template_present=false\n'
printf 'served_context=262144\n'
printf 'kv_in_host_ram=true\n'
printf 'api=OpenAI_compatible_local_server\n'
printf 'checkpoint_class=BASE_NOT_INSTRUCT\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime tree are excluded from this repository\n'
