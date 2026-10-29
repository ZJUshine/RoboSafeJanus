#!/usr/bin/env bash
# Real-time TX/RX throughput for one interface, no external deps (/proc/net/dev).
#
#   bash scripts/internal/netspeed.sh                 # default route iface, 1s, auto unit
#   bash scripts/internal/netspeed.sh --mb            # force MiB/s (下载/上传都用 MB/s 显示)
#   bash scripts/internal/netspeed.sh --mbit          # force Mbit/s
#   bash scripts/internal/netspeed.sh enp3s0f0np0 0.5
#   bash scripts/internal/netspeed.sh --list          # all ifaces + default route
#
# RX = download, TX = upload. Totals are since the script started.

set -uo pipefail

UNIT="auto"   # auto | mb | mbit
POS=()

read_iface() {
  ip route show default 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "dev") {print $(i + 1); exit}}'
}

for arg in "$@"; do
  case "$arg" in
    --mb) UNIT="mb" ;;
    --mbit) UNIT="mbit" ;;
    --list)
      def="$(read_iface)"
      while read -r name rx _ _ _ _ _ _ _ tx _; do
        [[ "$rx" =~ ^[0-9]+$ ]] || continue
        name="${name%:}"
        mark=""
        [[ "$name" == "$def" ]] && mark="  <- default route"
        printf '%-18s rx=%-16s tx=%s%s\n' "$name" "$rx" "$tx" "$mark"
      done < /proc/net/dev
      exit 0
      ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) POS+=("$arg") ;;
  esac
done

IFACE="${POS[0]:-$(read_iface)}"
INTERVAL="${POS[1]:-1}"
[[ -z "$IFACE" ]] && { echo "no interface found; pass one explicitly (--list)" >&2; exit 1; }
[[ -r /proc/net/dev ]] || { echo "/proc/net/dev not readable" >&2; exit 1; }

counters() {
  awk -v ifc="$IFACE" '$1 == ifc ":" {print $2, $10}' /proc/net/dev
}

read -r rx0 tx0 < <(counters)
if [[ -z "${rx0:-}" ]]; then
  echo "interface '$IFACE' not found (try --list)" >&2
  exit 1
fi

# bytes -> display string, honouring UNIT
fmt() {  # $1 = bytes or bytes/s, $2 = "/s" or ""
  awk -v b="$1" -v s="$2" -v mode="$UNIT" 'BEGIN {
    if (mode == "mb")        { printf "%9.2f MiB%s", b / 1048576, s; exit }
    if (mode == "mbit")      { printf "%9.2f Mbit%s", b * 8 / 1000000, s; exit }
    split("B KiB MiB GiB TiB", u, " "); i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf "%7.2f %s%s", b, u[i], s
  }'
}

printf 'iface=%s  interval=%ss  unit=%s   (Ctrl-C to stop)\n' "$IFACE" "$INTERVAL" "$UNIT"
printf '%8s  %16s  %16s  %14s  %14s\n' "elapsed" "down (RX)" "up (TX)" "down tot" "up tot"

start=$(date +%s)
while sleep "$INTERVAL"; do
  read -r rx1 tx1 < <(counters)
  [[ -z "${rx1:-}" ]] && continue
  dm=$((rx1 - rx0)); um=$((tx1 - tx0))
  rx0=$rx1; tx0=$tx1
  rate_d=$(awk -v b="$dm" -v t="$INTERVAL" 'BEGIN{printf "%.0f", b/t}')
  rate_u=$(awk -v b="$um" -v t="$INTERVAL" 'BEGIN{printf "%.0f", b/t}')
  printf '%7ss  %16s  %16s  %14s  %14s\n' \
    "$(( $(date +%s) - start ))" "$(fmt "$rate_d" "/s")" "$(fmt "$rate_u" "/s")" \
    "$(fmt "$rx1" "")" "$(fmt "$tx1" "")"
done
