#!/usr/bin/env bash
# Install the game-clock fetcher on the Hetzner box.
#
# Run from this directory:  ./install.sh [user@host]
#
# Idempotent: safe to rerun to push a script change or re-seed. It never
# touches data/games.jsonl, so a rerun resumes rather than restarting.
set -euo pipefail

HOST="${1:-root@138.199.195.186}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR=/opt/gamefetch
EXPLORER_DB="${EXPLORER_DB:-/opt/bughouse/data/games.db}"

# Seeding parameters. process_parquet_file filters at 2200 on all four players,
# so seeding below that just buys wasted fetches.
MIN_RATING="${MIN_RATING:-2200}"
TIME_CONTROLS="${TIME_CONTROLS:-180,120}"
ORDER="${ORDER:-recent}"

ssh_() { ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "$HOST" "$@"; }
scp_() { scp -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" "$@"; }

echo "==> Checking the explorer DB is present on $HOST"
ssh_ "test -r $EXPLORER_DB" || {
    echo "ERROR: $EXPLORER_DB not readable. The explorer was retired but its" >&2
    echo "       database was meant to be preserved; check /opt/bughouse."     >&2
    exit 1
}

echo "==> Creating the service account and layout"
ssh_ "set -e
    id -u gamefetch >/dev/null 2>&1 || useradd --system --home-dir $REMOTE_DIR \
        --shell /usr/sbin/nologin gamefetch
    mkdir -p $REMOTE_DIR/data
    chown -R gamefetch:gamefetch $REMOTE_DIR"

echo "==> Copying scripts"
scp_ fetch_games.py jsonl_to_parquet.py "$HOST:$REMOTE_DIR/"
ssh_ "chown gamefetch:gamefetch $REMOTE_DIR/fetch_games.py $REMOTE_DIR/jsonl_to_parquet.py"

echo "==> Building the venv (cloudscraper only; polars is not needed here)"
ssh_ "set -e
    test -d $REMOTE_DIR/venv || python3 -m venv $REMOTE_DIR/venv
    $REMOTE_DIR/venv/bin/pip install --quiet --upgrade pip
    $REMOTE_DIR/venv/bin/pip install --quiet cloudscraper
    chown -R gamefetch:gamefetch $REMOTE_DIR/venv"

echo "==> Seeding candidate ids from the explorer DB"
# Done here, as root and out of band, so the sandboxed service never needs to
# open the DB -- sqlite wants to create -wal/-shm siblings even for reads, and
# ProtectSystem=strict forbids it.
ssh_ "$REMOTE_DIR/venv/bin/python $REMOTE_DIR/fetch_games.py seed \
        --db '$EXPLORER_DB' \
        --out $REMOTE_DIR/data/candidates.txt \
        --min-rating $MIN_RATING \
        --time-controls '$TIME_CONTROLS' \
        --order $ORDER
    chown gamefetch:gamefetch $REMOTE_DIR/data/candidates.txt"

echo "==> Installing the unit"
scp_ gamefetch.service "$HOST:/etc/systemd/system/gamefetch.service"
ssh_ "systemctl daemon-reload"

cat <<EOF

Installed. Nothing is running yet.

  start        ssh $HOST systemctl start gamefetch
  follow       ssh $HOST journalctl -u gamefetch -f
  progress     ssh $HOST "wc -l < $REMOTE_DIR/data/games.jsonl"   # 2 lines per game
  stop         ssh $HOST systemctl stop gamefetch

At one request per second and two requests per game, 10000 games is about
5.6 hours. The service exits 0 when it reaches its target; to extend, edit
--target-games in /etc/systemd/system/gamefetch.service, daemon-reload, and
start it again. Progress is never discarded.

Enable it across reboots only if you want that:
  ssh $HOST systemctl enable gamefetch

When it finishes, pull the data and convert it where polars lives:
  scp -i $SSH_KEY $HOST:$REMOTE_DIR/data/games.jsonl .
  python jsonl_to_parquet.py --jsonl games.jsonl --out data/games.parquet
EOF
