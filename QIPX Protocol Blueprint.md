# QIPX Protocol Blueprint

## Quantum Inter-Projection Exchange for BrailleStream

**Working name:** `QIPX`
**Backronym:** **Quantum Inter-Projection eXchange**
**Role:** A lightweight LAN mesh protocol where every BrailleStream instance can discover other instances, exchange retinal byte streams, sync fold states, merge projections, and evolve shared “reality” layers.

The important simplification:

> QIPX is not real quantum networking. It is a character/byte-stream protocol that treats each visual projection as a possible “state” of the same retinal tape.

BrailleStream already has the correct foundation: its core stream is a raw `bytearray`, each byte is a 2×4 micro-pixel cell, and the renderer folds that 1D stream into 2D screen geometry.  The native field is exactly `640×480`, with `CELL_W=2`, `CELL_H=4`, `GRID_W=320`, `GRID_H=120`, and `MAX_CELLS=38400`. 

---

# 1. Core Concept

Each BrailleStream instance becomes a **node**.

Each node has:

```text
1. A retinal stream        = bytearray of 8-bit masks
2. A projection state      = fold_w, fold_h, palette, mode
3. A resonance opinion     = current score + best known folds
4. A peer memory           = other nodes' streams/states
5. A merge engine          = combines local + remote projections
6. A Pazuzu cognition loop = lightweight criticality/stability controller
```

BrailleStream already has render modes for native pixel view, arbitrary fold scan, scaled preview, density map, resonance analyzer, and paradox overlay.  QIPX simply lets those modes become **network-visible states**.

---

# 2. User Behavior

On startup:

```text
BrailleStream starts normally.
QIPX module initializes in standby mode.
Node creates identity.
Node opens UDP discovery socket.
Node opens TCP/UDP data socket.
QIPX remains passive unless enabled.
```

Keyboard:

```text
Q = toggle QIPX ON/OFF
Shift+Q = hard reset QIPX peer table
Ctrl+Q = toggle auto-merge mode
Alt+Q = show QIPX diagnostics overlay
```

When `Q` is enabled:

```text
1. Node announces itself.
2. Node discovers peers.
3. Node exchanges state packets.
4. Node compares stream hashes.
5. Node requests missing streams.
6. Node receives remote fold/projection states.
7. Node can ghost, merge, or consensus-lock.
```

---

# 3. Protocol Layers

```text
QIPX/0.1
├─ Layer 0: Local Node Identity
├─ Layer 1: LAN Auto Discovery
├─ Layer 2: State Exchange
├─ Layer 3: Stream Transfer
├─ Layer 4: Projection Sync
├─ Layer 5: Merge / Reality Creation
├─ Layer 6: Pazuzu Cognition
└─ Layer 7: Safety / Rollback / Trust Ledger
```

---

# 4. QIPX Packets

Use simple newline-delimited UTF-8 headers with optional binary or base64 payloads.

## 4.1 HELLO

```text
QIPX/0.1 HELLO
node_id=bs-a83f21
name=mikey-node-1
version=0.1
caps=state,stream,ghost,merge,pazuzu
screen=640x480
cell=2x4
max_cells=38400
port=47777
```

## 4.2 STATE

Sent several times per second while enabled.

```text
QIPX/0.1 STATE
node_id=bs-a83f21
seq=1082
stream_hash=sha256:9f03...
stream_len=38400
fold_w=310
fold_h=124
palette_id=1
palette_name=Density Fire
render_mode=1
mode_name=Fold Scan
entropy=7.53
avg_density=4.09
unique_masks=9
score=11.305
scan_active=0
ghost_enabled=1
ghost_w=155
```

These values map directly to existing renderer and engine state. The renderer already tracks `stream`, `fold_w`, `palette_id`, `render_mode`, ghost state, scan state, lock flash, and redraw state.  The HUD already displays fold width, height, score, palette, mode, length, entropy, average density, and unique mask count. 

## 4.3 LOCK

Broadcast when a node locks onto a strong fold.

```text
QIPX/0.1 LOCK
node_id=bs-a83f21
stream_hash=sha256:9f03...
fold_w=196
fold_h=196
score=11.252
confidence=0.81
reason=resonance_minimum
```

BrailleStream already has resonance scoring and best-width search. The score uses aspect distance, divisor penalty, and screen-fit penalty; lower is better. 

## 4.4 STREAM REQUEST

```text
QIPX/0.1 GET_STREAM
node_id=bs-b90c11
target=bs-a83f21
stream_hash=sha256:9f03...
encoding=raw
```

## 4.5 STREAM DATA

```text
QIPX/0.1 STREAM
node_id=bs-a83f21
stream_hash=sha256:9f03...
stream_len=38400
encoding=b64
chunk=0
chunks=1

<base64 data>
```

Native frames are tiny: `38400` raw bytes. Base64 is roughly 51 KB. That is small enough for LAN sync.

BrailleStream already exports and imports raw binary streams through `.BIN`, with padding/clamping to `MAX_CELLS`.  QIPX uses the same data model, just over the network.

## 4.6 GHOST

```text
QIPX/0.1 GHOST
node_id=bs-a83f21
target=all
stream_hash=sha256:9f03...
ghost_w=155
alpha=60
polarity=density_split
```

The renderer already supports a second fold-width ghost overlay, using translucent blue/red overlay depending on density.  QIPX turns that into **remote ghosting**.

## 4.7 MERGE

```text
QIPX/0.1 MERGE
node_id=bs-a83f21
merge_id=mx-00041
parents=sha256:a...,sha256:b...
method=weighted_density
alpha=0.50
result_hash=sha256:c...
```

## 4.8 PAZUZU

```text
QIPX/0.1 PAZUZU
node_id=bs-a83f21
lambda_dom=0.037
critical_band=0.001,0.100
coherence=0.72
novelty=0.18
entropy_potential=0.64
elegance=0.41
criticality_index=0.77
parity=+1
action=stabilize
```

PazuzuCore’s uploaded framework defines a critical band control model, where the dominant real eigenvalue is kept inside `lambda_min < |Re lambda| < lambda_max`, with defaults around `1e-3` to `1e-1`.  It also defines five metrics: novelty, entropic potential, elegance, coherence, and criticality index.  For BrailleStream, these become **cognition-like control metrics**, not literal consciousness.

---

# 5. Auto Discovery

Use UDP multicast or broadcast.

Default:

```text
UDP discovery port: 47777
UDP multicast group: 239.77.77.77
TCP stream port: 47778
Protocol name: QIPX/0.1
```

Discovery loop:

```text
Every 1000 ms:
    broadcast HELLO if QIPX enabled

Every received HELLO:
    validate version
    store peer
    reply HELLO_ACK
    request STATE
```

Peer table:

```python
@dataclass
class QipxPeer:
    node_id: str
    name: str
    addr: tuple[str, int]
    last_seen: float
    stream_hash: str
    stream_len: int
    fold_w: int
    fold_h: int
    palette_id: int
    render_mode: int
    score: float
    entropy: float
    avg_density: float
    unique_masks: int
    pazuzu: dict
```

---

# 6. Sync Model

QIPX has three sync levels.

## Level 1 — State Sync

Nodes share fold width, palette, render mode, entropy, and resonance score.

```text
Cheap
Fast
No stream transfer needed
Good for swarm awareness
```

## Level 2 — Stream Sync

Nodes exchange the actual 38,400-byte retinal stream when hashes differ.

```text
Medium cost
Needed for shared reality
Allows remote ghosting
Allows merge
```

## Level 3 — Reality Sync

Nodes combine streams and projections into shared derived realities.

```text
Highest cost
Creates new stream
Can be accepted, rejected, or sandboxed
```

---

# 7. Merge / “Create Reality” Engine

This is where QIPX becomes fun.

Each node can combine local and remote streams into a new generated stream.

## 7.1 Merge Methods

### A. XOR Reality

Good for interference/paradox effects.

```python
out[i] = local[i] ^ remote[i]
```

### B. OR Reality

Accumulates all visible structure.

```python
out[i] = local[i] | remote[i]
```

### C. AND Reality

Only keeps mutually agreed structure.

```python
out[i] = local[i] & remote[i]
```

### D. Weighted Density Merge

Preserves density rather than exact bits.

```python
d = round(alpha * density(local[i]) + (1-alpha) * density(remote[i]))
out[i] = canonical_mask_for_density[d]
```

### E. Resonance-Winner Merge

At each block, pick the stream from the node with better local resonance score.

```python
out_block = block_from_best_scoring_peer
```

### F. Pazuzu-Stabilized Merge

Only accepts a merge if it improves critical metrics:

```text
accept if:
    coherence increases
    entropy does not explode
    resonance score improves or remains stable
    novelty improves without destroying structure
```

PazuzuCore’s framework already emphasizes critical-band compliance, Pareto hypervolume improvement, diagnostics, falsification, and rollback.  QIPX should borrow that attitude: merge aggressively, but rollback when the result destabilizes.

---

# 8. Pazuzu Cognition Layer

This should be framed carefully:

> Pazuzu Cognition gives each node a lightweight adaptive control loop. It is not proof of literal consciousness. It is a stateful perception/decision system that behaves like a primitive cognitive agent.

## 8.1 Node “Mind State”

```python
@dataclass
class PazuzuMindState:
    lambda_dom: float
    coherence: float
    novelty: float
    entropic_potential: float
    elegance: float
    criticality_index: float
    parity: int
    mood: str
    action: str
```

## 8.2 BrailleStream Metrics Mapping

| Pazuzu Metric      | BrailleStream Translation                                    |
| ------------------ | ------------------------------------------------------------ |
| Novelty            | How different this projection/merge is from previous states  |
| Entropic Potential | How much usable entropy exists without becoming noise        |
| Elegance           | How compact/simple the projection is                         |
| Coherence          | How aligned bands/structures are across folds/nodes          |
| Criticality Index  | How close the node is to useful instability without collapse |

## 8.3 Critical Band for Visual Reality

Define:

```text
lambda_dom = weighted instability estimate
```

Possible proxy:

```text
lambda_dom =
    0.30 * normalized_entropy_change
  + 0.25 * normalized_score_change
  + 0.20 * peer_disagreement
  + 0.15 * merge_instability
  + 0.10 * scan_velocity
```

Target:

```text
0.001 < abs(lambda_dom) < 0.100
```

Below band: node is too static.

Above band: node is chaotic/noisy.

Inside band: node is creatively alive.

## 8.4 Pazuzu Actions

```text
STABILIZE     reduce scan speed, reject wild merges
EXPLORE       scan new fold widths
LOCK          accept best fold
GHOST         overlay peer projection
MERGE         create derived stream
ROLLBACK      return to last stable stream
BROADCAST     share good discovery
QUIET         observe only
```

---

# 9. Q Button Integration

Current controls already use keyboard-driven Pygame events in `bs_main.py`, including width controls, pattern cycling, palette, render mode, ghost, scan, lock, import/export, binary load, screenshot, and quit.  Add `Q` beside those event handlers.

Pseudo-patch:

```python
# imports
from bs_qipx import QipxNode

def main():
    r = BrailleStreamRenderer()
    qipx = QipxNode(renderer=r, enabled=False)
    qipx.start()

    ...

    while r.running:
        for event in pygame.event.get():
            ...

            elif key == pygame.K_q:
                qipx.toggle()
                r.needs_redraw = True

        # after local controls
        qipx.tick()

        if qipx.enabled:
            qipx.publish_state()
            qipx.apply_pending_updates()

        if r.needs_redraw or r.scan_active or r.lock_flash > 0:
            r.draw()

    qipx.stop()
    pygame.quit()
```

---

# 10. New Files

Add these modules:

```text
bs_qipx.py          main QIPX node, sockets, packet routing
bs_qipx_packets.py  packet encode/decode
bs_qipx_peer.py     peer table
bs_qipx_merge.py    stream merge functions
bs_pazuzu.py        Pazuzu cognition loop for BrailleStream
bs_identity.py      node ID, local name, keys later
```

No new dependencies required for v0.1.

Use only Python standard library:

```text
socket
threading
queue
time
hashlib
base64
json
dataclasses
uuid
```

Existing requirements are only `pygame`, `Pillow`, and `numpy`.  Keep it that way for now.

---

# 11. `bs_qipx.py` Skeleton

```python
from __future__ import annotations
from dataclasses import dataclass, field
import socket, threading, time, hashlib, base64, json, uuid, queue
from typing import Dict, Optional

DISCOVERY_PORT = 47777
DATA_PORT = 47778
BROADCAST_ADDR = "255.255.255.255"
QIPX_VERSION = "0.1"


@dataclass
class PeerState:
    node_id: str
    addr: tuple
    last_seen: float
    state: dict = field(default_factory=dict)
    stream: Optional[bytearray] = None


class QipxNode:
    def __init__(self, renderer, name: str = "braillestream-node", enabled: bool = False):
        self.renderer = renderer
        self.name = name
        self.node_id = "bs-" + uuid.uuid4().hex[:8]
        self.enabled = enabled
        self.running = False
        self.peers: Dict[str, PeerState] = {}
        self.inbox = queue.Queue()
        self.last_hello = 0.0
        self.last_state = 0.0

    def start(self):
        self.running = True
        self.discovery_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.discovery_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.discovery_sock.bind(("", DISCOVERY_PORT))

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

    def stop(self):
        self.running = False
        try:
            self.discovery_sock.close()
        except Exception:
            pass

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.send_hello(force=True)

    def tick(self):
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_hello > 1.0:
            self.send_hello()
            self.last_hello = now

        if now - self.last_state > 0.20:
            self.publish_state()
            self.last_state = now

        self._drain_inbox()

    def stream_hash(self) -> str:
        return hashlib.sha256(bytes(self.renderer.stream)).hexdigest()

    def current_state(self) -> dict:
        stats = self.renderer.stream_stats()
        w = self.renderer.fold_w
        h = max(1, -(-len(self.renderer.stream) // w))
        return {
            "type": "STATE",
            "version": QIPX_VERSION,
            "node_id": self.node_id,
            "name": self.name,
            "stream_hash": self.stream_hash(),
            "stream_len": len(self.renderer.stream),
            "fold_w": w,
            "fold_h": h,
            "palette_id": self.renderer.palette_id,
            "render_mode": self.renderer.render_mode,
            "score": self.renderer.current_score(),
            "entropy": stats.get("entropy", 0.0),
            "avg_density": stats.get("avg_density", 0.0),
            "unique_masks": stats.get("unique_masks", 0),
            "ghost_w": self.renderer.ghost_w,
            "show_ghost": self.renderer.show_ghost,
            "time": time.time(),
        }

    def send_hello(self, force=False):
        pkt = {
            "type": "HELLO",
            "version": QIPX_VERSION,
            "node_id": self.node_id,
            "name": self.name,
            "port": DATA_PORT,
            "caps": ["state", "stream", "ghost", "merge", "pazuzu"],
            "time": time.time(),
        }
        self._send_json(pkt, (BROADCAST_ADDR, DISCOVERY_PORT))

    def publish_state(self):
        self._send_json(self.current_state(), (BROADCAST_ADDR, DISCOVERY_PORT))

    def _send_json(self, obj: dict, addr):
        data = ("QIPX " + json.dumps(obj, separators=(",", ":"))).encode("utf-8")
        self.discovery_sock.sendto(data, addr)

    def _rx_loop(self):
        while self.running:
            try:
                data, addr = self.discovery_sock.recvfrom(65535)
                if not data.startswith(b"QIPX "):
                    continue
                obj = json.loads(data[5:].decode("utf-8"))
                if obj.get("node_id") == self.node_id:
                    continue
                self.inbox.put((obj, addr))
            except OSError:
                break
            except Exception:
                continue

    def _drain_inbox(self):
        while not self.inbox.empty():
            obj, addr = self.inbox.get_nowait()
            t = obj.get("type")

            node_id = obj.get("node_id")
            if not node_id:
                continue

            peer = self.peers.get(node_id)
            if peer is None:
                peer = PeerState(node_id=node_id, addr=addr, last_seen=time.time())
                self.peers[node_id] = peer

            peer.last_seen = time.time()
            peer.addr = addr
            peer.state.update(obj)

            if t == "LOCK":
                self._handle_remote_lock(peer, obj)

    def _handle_remote_lock(self, peer, obj):
        # v0.1 behavior: remote lock becomes ghost width suggestion
        remote_w = int(obj.get("fold_w", 0))
        if remote_w > 0:
            self.renderer.ghost_w = remote_w
            self.renderer.show_ghost = True
            self.renderer.needs_redraw = True
```

---

# 12. Pazuzu Cognition Skeleton

```python
from dataclasses import dataclass
import math
import time


@dataclass
class PazuzuState:
    lambda_dom: float = 0.01
    coherence: float = 0.5
    novelty: float = 0.0
    entropic_potential: float = 0.5
    elegance: float = 0.5
    criticality_index: float = 0.5
    parity: int = 1
    mood: str = "OBSERVE"
    action: str = "QUIET"


class BraillePazuzu:
    def __init__(self, lambda_min=1e-3, lambda_max=1e-1):
        self.state = PazuzuState()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.prev_entropy = None
        self.prev_score = None
        self.prev_hash = None

    def step(self, renderer, peers):
        stats = renderer.stream_stats()
        entropy = stats["entropy"]
        score = renderer.current_score()

        if self.prev_entropy is None:
            self.prev_entropy = entropy
            self.prev_score = score

        d_entropy = abs(entropy - self.prev_entropy)
        d_score = abs(score - self.prev_score)

        peer_count = len(peers)
        peer_disagreement = self._peer_disagreement(renderer, peers)

        lambda_dom = (
            0.30 * min(1.0, d_entropy)
            + 0.25 * min(1.0, d_score / 10.0)
            + 0.20 * peer_disagreement
            + 0.15 * min(1.0, peer_count / 8.0)
            + 0.10 * (1.0 if renderer.scan_active else 0.0)
        )

        self.state.lambda_dom = lambda_dom
        self.state.coherence = 1.0 - peer_disagreement
        self.state.entropic_potential = min(1.0, entropy / 8.0)
        self.state.elegance = 1.0 / (1.0 + score)
        self.state.criticality_index = self._criticality(lambda_dom)

        self.state.action = self._choose_action()
        self.state.mood = self._mood()

        self.prev_entropy = entropy
        self.prev_score = score
        return self.state

    def _peer_disagreement(self, renderer, peers):
        if not peers:
            return 0.0
        local_w = renderer.fold_w
        diffs = []
        for p in peers.values():
            rw = p.state.get("fold_w")
            if rw:
                diffs.append(abs(local_w - int(rw)) / 320.0)
        return min(1.0, sum(diffs) / max(1, len(diffs)))

    def _criticality(self, lam):
        if self.lambda_min <= abs(lam) <= self.lambda_max:
            return 1.0
        if abs(lam) < self.lambda_min:
            return abs(lam) / self.lambda_min
        return max(0.0, 1.0 - (abs(lam) - self.lambda_max))

    def _choose_action(self):
        lam = abs(self.state.lambda_dom)
        if lam < self.lambda_min:
            return "EXPLORE"
        if lam > self.lambda_max:
            return "STABILIZE"
        if self.state.coherence > 0.85:
            return "LOCK"
        if self.state.entropic_potential > 0.75:
            return "MERGE"
        return "OBSERVE"

    def _mood(self):
        if self.state.action == "EXPLORE":
            return "HUNGRY"
        if self.state.action == "STABILIZE":
            return "BOUNDARY"
        if self.state.action == "LOCK":
            return "FOCUSED"
        if self.state.action == "MERGE":
            return "DREAMING"
        return "WATCHING"
```

---

# 13. Renderer HUD Additions

Add to HUD:

```text
QIPX: ON peers=3 reality=local/merged pazuzu=FOCUSED action=LOCK
```

Possible state colors:

```text
OFF       gray
ON        cyan
LOCK      yellow
MERGE     magenta
STABLE    green
CHAOS     red
```

Add a small peer list in diagnostics mode:

```text
QIPX PEERS
bs-a83f21  W=196 H=196 score=11.25 mood=FOCUSED
bs-b90c11  W=310 H=124 score=11.30 mood=DREAMING
bs-c13ee9  W=90  H=427 score=??    mood=WATCHING
```

---

# 14. Reality Creation Rules

A node should not overwrite its own current stream instantly.

Use three reality buffers:

```text
local_stream      = what this node owns
remote_streams    = peer streams
reality_stream    = merged/generated stream
```

Modes:

```text
LOCAL      render only local stream
GHOST      render local + remote projection overlay
MERGED     render accepted merged stream
SANDBOX    preview generated reality without committing
CONSENSUS  use swarm-voted lock/merge
```

Acceptance gate:

```text
Accept merged reality if:
    entropy <= local_entropy + entropy_margin
    score <= local_score + score_margin
    unique_masks does not collapse to 1 unless intended
    Pazuzu action != STABILIZE
```

Rollback:

```text
Before every merge:
    save stream snapshot
    save fold_w, palette, mode
    save hash
```

---

# 15. Security / Trust Model v0.1

LAN toy mode first:

```text
No authentication
Accept local subnet only
Ignore packets over max size
Ignore stream_len > MAX_CELLS
Ignore incompatible cell geometry
Rate-limit peer updates
```

v0.2:

```text
Ed25519 node keys
signed STATE/LOCK/MERGE packets
trust score per node
ban malformed nodes
Merkle ledger for reality merges
```

PazuzuCore already uses the idea of a typed Merkle ledger for axioms/configs/state accountability.  QIPX can later use a simplified Merkle chain for merge history.

---

# 16. Development Phases

## Phase 0 — Minimal Q Toggle

```text
[ ] Add bs_qipx.py
[ ] Start node on app startup
[ ] Q toggles enabled/disabled
[ ] Broadcast HELLO
[ ] Receive HELLO
[ ] Show peer count in HUD
```

## Phase 1 — State Swarm

```text
[ ] Broadcast STATE 5 times/sec
[ ] Peer table stores fold/palette/mode/score
[ ] HUD shows peer count
[ ] Remote LOCK sets ghost_w suggestion
```

## Phase 2 — Stream Exchange

```text
[ ] Stream hash comparison
[ ] Request missing stream
[ ] Send base64 stream chunks
[ ] Store remote stream
[ ] Render remote stream as ghost
```

## Phase 3 — Merge Reality

```text
[ ] XOR/OR/AND merge
[ ] Weighted density merge
[ ] Sandbox preview
[ ] Accept/reject merge
[ ] Rollback key
```

## Phase 4 — Pazuzu Cognition

```text
[ ] Add bs_pazuzu.py
[ ] Compute node cognitive state
[ ] Broadcast PAZUZU packet
[ ] Use action suggestions: EXPLORE, LOCK, MERGE, STABILIZE
[ ] Show node mood in HUD
```

## Phase 5 — Consensus Reality

```text
[ ] Vote on best fold width
[ ] Vote on best stream hash
[ ] Consensus lock
[ ] Multi-node merge
[ ] Reality ledger
```

---

# 17. The Clean Architecture Statement

> QIPX turns BrailleStream from a single retinal byte-field renderer into a distributed perceptual mesh. Each instance becomes a visual cortex node: it scans, folds, scores, ghosts, merges, and shares its best projection. Pazuzu Cognition provides the adaptive control shell, keeping each node inside a useful critical band between dead stability and chaotic noise.

Or in the more fun phrasing:

> Every node dreams the same byte-stream differently. QIPX lets them compare dreams, ghost each other’s realities, and merge the best hallucinations into a shared retinal world.
