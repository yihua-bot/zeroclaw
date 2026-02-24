#!/bin/sh
set -eu

CONFIG_PATH="${ZEROCLAW_CONFIG_PATH:-/zeroclaw-data/.zeroclaw/config.toml}"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
WORKSPACE_DIR="${ZEROCLAW_WORKSPACE:-/zeroclaw-data/workspace}"
GATEWAY_API_KEY="${ZEROCLAW_GATEWAY_API_KEY:-}"
GATEWAY_REQUIRE_PAIRING="${ZEROCLAW_GATEWAY_REQUIRE_PAIRING:-}"

if [ -z "$GATEWAY_REQUIRE_PAIRING" ]; then
  if [ -n "$GATEWAY_API_KEY" ]; then
    GATEWAY_REQUIRE_PAIRING="false"
  else
    GATEWAY_REQUIRE_PAIRING="true"
  fi
fi

if [ ! -s "$CONFIG_PATH" ]; then
  mkdir -p "$CONFIG_DIR" "$WORKSPACE_DIR"
  cat > "$CONFIG_PATH" <<CONFIG_EOF
workspace_dir = "/zeroclaw-data/workspace"
config_path = "/zeroclaw-data/.zeroclaw/config.toml"
api_key = ""
default_provider = "${ZEROCLAW_PROVIDER:-openrouter}"
default_model = "${ZEROCLAW_MODEL:-anthropic/claude-sonnet-4-20250514}"
default_temperature = 0.7

[gateway]
port = ${ZEROCLAW_GATEWAY_PORT:-42617}
host = "${ZEROCLAW_GATEWAY_HOST:-[::]}"
allow_public_bind = ${ZEROCLAW_ALLOW_PUBLIC_BIND:-true}
require_pairing = ${GATEWAY_REQUIRE_PAIRING}
api_key = "${GATEWAY_API_KEY}"
CONFIG_EOF
  chmod 600 "$CONFIG_PATH" || true
fi

exec zeroclaw "$@"
