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


---
---
---

# QIPX + BrailleStream + Pazuzu Cognition

## Full Contextual Insights & Mathematics

This is the complete conceptual/math layer for what you are building:

> **BrailleStream** is the retinal substrate.
> **QIPX** is the distributed inter-projection network.
> **Pazuzu Cognition** is the adaptive criticality/control layer.
> Together they form a **distributed visual cognition mesh** where each node folds the same byte-field differently, scores its projection, shares its state, ghosts other nodes, merges streams, and evolves shared “reality” candidates.

BrailleStream’s core is already extremely clean: no Unicode Braille, no browser text field, no fake glyph layer. Each byte is directly treated as a raw 8-bit 2×4 micro-pixel cell inside a `bytearray`.  The native retina is `640×480`, with `CELL_W=2`, `CELL_H=4`, producing `GRID_W=320`, `GRID_H=120`, and `MAX_CELLS=38400`. 

---

# 1. Foundational Retinal Mathematics

## 1.1 Native Screen Geometry

The physical render surface is:

```text
SCREEN_W = 640
SCREEN_H = 480
CELL_W   = 2
CELL_H   = 4
GRID_W   = SCREEN_W / CELL_W = 320
GRID_H   = SCREEN_H / CELL_H = 120
MAX_CELLS = GRID_W × GRID_H = 38,400
```

So the entire screen is a linear byte stream:

```text
S = [b₀, b₁, b₂, ..., b₃₈₃₉₉]
```

where:

```text
bᵢ ∈ {0,1,...,255}
```

Each byte is not a character. It is a **2×4 retinal micro-cell**.

---

## 1.2 Cell Bit Layout

Each byte encodes 8 dots:

```text
bit0 = row0 col0
bit1 = row0 col1
bit2 = row1 col0
bit3 = row1 col1
bit4 = row2 col0
bit5 = row2 col1
bit6 = row3 col0
bit7 = row3 col1
```

Mathematically:

```text
bit_index = row × CELL_W + col
```

So:

```text
dot(mask,row,col) = ((mask >> (row×2 + col)) & 1) == 1
```

This is exactly how `bs_dot(mask,row,col)` works. 

---

## 1.3 Density Function

Every retinal cell has density:

```text
D(b) = popcount(b)
```

where:

```text
D(b) ∈ {0,1,2,3,4,5,6,7,8}
```

So:

```text
D(0x00) = 0
D(0xFF) = 8
D(0xAA) = 4
D(0x55) = 4
```

This is the base “brightness” or “mass” of the cell. The engine defines this with `bs_density(mask) = bin(mask).count('1')`. 

---

## 1.4 Stream-to-Cell Projection

The stream is 1D. The image is created by folding the stream into 2D.

For stream index `i` and fold width `W`:

```text
x = i mod W
y = floor(i / W)
```

So:

```text
Π_W(i) = (i mod W, floor(i/W))
```

This is the core projection function.

The code uses this exact mapping through `stream_to_cell(index, fold_w)`. 

---

## 1.5 Cell-to-Screen Projection

Once the cell coordinate exists:

```text
screen_x = cell_x × CELL_W
screen_y = cell_y × CELL_H
```

So:

```text
screen_x = 2x
screen_y = 4y
```

The full projection from stream index to screen position is:

```text
P_W(i) = (2(i mod W), 4 floor(i/W))
```

That is the fundamental geometry of BrailleStream.

---

# 2. Fold Geometry

## 2.1 Fold Width

A fold width `W` defines how the same byte tape becomes an image.

```text
W = number of cells per row
H = ceil(N / W)
N = stream length
```

For the native stream:

```text
N = 38400
```

Examples:

```text
W = 320 → H = 120
W = 240 → H = 160
W = 196 → H = 196 approximately
W = 160 → H = 240
W = 120 → H = 320
W = 90  → H = 427
W = 80  → H = 480
```

The controls already expose these fold presets: `320, 240, 192, 160, 128, 120, 96, 80, 64`. 

---

## 2.2 Aspect Ratio Correction

The visible pixel aspect ratio is:

```text
pixel_ratio = SCREEN_W / SCREEN_H = 640/480 = 1.333...
```

But each cell is 2 pixels wide and 4 pixels tall, so the cell-space ratio must compensate:

```text
cell_ratio = W / H
```

Pixel ratio from cells:

```text
pixel_ratio = (W × 2) / (H × 4)
            = W / (2H)
```

To fill `640×480`, we need:

```text
W / (2H) = 640 / 480
```

Therefore:

```text
W/H = 2 × 640/480
W/H = 2.666...
```

The engine defines this corrected target as `TARGET_CELL_RATIO`. 

So:

```text
TARGET_CELL_RATIO = SCREEN_W / (SCREEN_H × (CELL_H / CELL_W))
                  = 640 / (480 × 2)
                  = 0.666...? 
```

However, conceptually the intended native cell ratio is:

```text
GRID_W / GRID_H = 320 / 120 = 2.666...
```

The important operational target is:

```text
native cell geometry = 320×120
native W/H = 2.666...
```

---

## 2.3 Square Harmonic

The stream length is:

```text
N = 38400
```

The square root is:

```text
sqrt(38400) ≈ 195.959
```

So the near-square fold is:

```text
W ≈ 196
H ≈ 196
```

And:

```text
196 × 196 = 38416
```

That is only 16 cells above the full stream length.

So the “square projection” is almost perfect:

```text
38416 - 38400 = 16
```

This explains why `W=196 H=196` appears as a stable square reality.

Insight:

> The native system is wide retina, but the byte-stream also contains a near-perfect square projection.

---

# 3. Resonance Mathematics

## 3.1 Resonance Score

BrailleStream already has resonance scoring.

The score is lower when:

```text
1. The fold aspect is close to the target ratio.
2. The stream divides cleanly by the fold width.
3. The projection fits on screen.
```

The engine computes:

```text
score(W) = aspect_score + divisor_score + screen_fit_score
```

where:

```text
H = ceil(N/W)
aspect = W / H
aspect_score = |aspect - TARGET_CELL_RATIO|
remainder = N mod W
divisor_score = remainder / W
screen_fit_score = overflow penalty
```

This is exactly how `resonance_score(stream_len, fold_w)` is structured. 

---

## 3.2 Exact Divisor Resonance

If:

```text
N mod W = 0
```

then:

```text
divisor_score = 0
```

That means the stream folds without a ragged final row.

For `N=38400`, important divisors include:

```text
W = 320 → H = 120
W = 300 → H = 128
W = 256 → H = 150
W = 240 → H = 160
W = 200 → H = 192
W = 192 → H = 200
W = 160 → H = 240
W = 150 → H = 256
W = 128 → H = 300
W = 120 → H = 320
W = 100 → H = 384
W = 96  → H = 400
W = 80  → H = 480
W = 64  → H = 600
```

Interpretation:

> Exact divisor folds are geometrically clean because no byte is stranded in an incomplete row.

---

## 3.3 Near-Divisor Resonance

Some folds are not exact divisors but still perceptually powerful.

Example:

```text
W = 196
H = ceil(38400 / 196) = 196
W×H = 38416
error = 16 cells
```

This is a near-square attractor.

Example:

```text
W = 310
H = ceil(38400 / 310) = 124
W×H = 38440
error = 40 cells
```

This is close to the wide/native ratio.

Example:

```text
W = 90
H = ceil(38400 / 90) = 427
W×H = 38430
error = 30 cells
```

This becomes a tall “sheet music” projection.

Insight:

> Perceptual stability does not require perfect divisibility. It can also emerge from near-completion, low leftover, and strong structural alignment.

---

## 3.4 Resonance Search

The engine searches:

```text
for W in min_w..max_w:
    H = ceil(N/W)
    S = resonance_score(N,W)
sort by S ascending
return top N
```

This is `find_resonance`. 

Mathematically:

```text
R(N) = sort_W(score_N(W))
```

The best fold is:

```text
W* = argmin_W score_N(W)
```

The lock event is:

```text
LOCK = (W*, H*, score(W*))
```

---

# 4. Harmonic Multipliers

The engine already defines harmonic child widths:

```text
child_w = floor(fold_w / divisor)
```

for:

```text
divisor = 2..9
```

and checks whether:

```text
N mod child_w = 0
```

This gives harmonic copies. 

Mathematically:

```text
Harmonics(W) = { (d, floor(W/d), N/floor(W/d)) | d∈[2,9], N mod floor(W/d)=0 }
```

Insight:

> A fold is not isolated. It has a harmonic family. Wide projections contain child projections. A node can use the harmonic family as a visual ancestry tree.

---

# 5. Render Mode Mathematics

BrailleStream has six render modes. 

## 5.1 Mode 0 — Native Pixel

Uses:

```text
W = GRID_W = 320
H = GRID_H = 120
```

Projection:

```text
P_native(i) = (2(i mod 320), 4 floor(i/320))
```

Each active bit becomes one physical pixel.

Insight:

> Native mode is the literal retinal interpretation.

---

## 5.2 Mode 1 — Fold Scan

Uses arbitrary `fold_w`.

Projection:

```text
P_W(i) = (2(i mod W), 4 floor(i/W))
```

Clips anything beyond screen bounds.

Insight:

> Fold Scan is not resizing. It is reinterpreting the topology of the byte tape.

---

## 5.3 Mode 2 — Scaled Preview

Given:

```text
W = fold_w
H = ceil(N/W)
```

Cell scale:

```text
cw = SCREEN_W / W
ch = SCREEN_H / H
```

Each dot is drawn inside scaled rectangles.

Insight:

> Scaled preview preserves all cells by compressing the projection into screen space.

---

## 5.4 Mode 3 — Density Map

Each byte becomes one colored block.

```text
color = palette[D(bᵢ)]
```

Cell-internal 2×4 structure is hidden.

Insight:

> Density map compresses the 8-bit internal shape into scalar brightness/mass.

---

## 5.5 Mode 4 — Resonance Analyzer

Displays:

```text
score
W
H
W/H
visual bar
```

for best fold candidates.

Insight:

> Resonance analyzer exposes the “fold spectrum” of the stream.

---

## 5.6 Mode 5 — Paradox Overlay

Density splits into polarity:

```text
D < 4 → blue / entropy / dissipation
D = 4 → white / paradox closure
D > 4 → red / structure / storage
```

The renderer implements this polarity split directly. 

Mathematically:

```text
color(b) =
    blue(intensity(D))   if D(b) < 4
    white                if D(b) = 4
    red(intensity(D))    if D(b) > 4
```

Insight:

> Paradox mode turns density balance into a symbolic field: low-density entropy, high-density structure, mid-density closure.

---

# 6. Palette Mathematics

Each palette maps density:

```text
D ∈ {0..8}
```

to color:

```text
C_D = (R,G,B)
```

So:

```text
color_i = Palette[D(bᵢ)]
```

The renderer defines five palettes: Binary Holy Light, Density Fire, Paradox Dualism, Terminal Retina, and Amiga Retro. 

Insight:

> Palettes are not cosmetic only. They are alternate semantic lenses over the same density field.

---

# 7. Stream Statistics

For stream:

```text
S = [b₀...bₙ₋₁]
```

## 7.1 Empty Count

```text
empty = |{bᵢ | D(bᵢ)=0}|
```

## 7.2 Full Count

```text
full = |{bᵢ | D(bᵢ)=8}|
```

## 7.3 Average Density

```text
avg_density = (1/N) Σ D(bᵢ)
```

## 7.4 Unique Masks

```text
unique_masks = |set(S)|
```

## 7.5 Shannon Entropy

Let:

```text
p_k = count(byte value k) / N
```

Then:

```text
Entropy(S) = -Σ p_k log₂(p_k)
```

The engine computes these in `stream_stats`. 

Insight:

> Entropy here is byte-symbol entropy, not visual entropy. It measures how many distinct retinal cell states are active and how evenly distributed they are.

---

# 8. Procedural Pattern Mathematics

The demo generators define artificial retinal universes.

## 8.1 Diagonal

```text
bᵢ = 0xFF if x = 2y else 0x00
```

Insight:

> A perfect line becomes a fold-sensitive phase marker.

---

## 8.2 Checkerboard

```text
bᵢ = 0xFF if (x+y) mod 2 = 0 else 0x00
```

Insight:

> Alternation tests parity stability of folds.

---

## 8.3 Circle

```text
bᵢ = 0xFF if (x-cₓ)² + (y-cᵧ)² ≤ r² else 0x00
```

Insight:

> Circle tests aspect distortion.

---

## 8.4 Vertical Bars

```text
bᵢ = 0xFF if x mod 8 < 4 else 0x00
```

Insight:

> Vertical bars reveal horizontal fold phase.

---

## 8.5 Horizontal Bars

```text
bᵢ = 0xFF if y mod 4 < 2 else 0x00
```

Insight:

> Horizontal bars reveal row continuity.

---

## 8.6 Gradient

```text
frac = x/(W-1)
density = round(8×frac)
```

then a mask is built with exactly `density` dots.

Insight:

> Gradient converts x-position into density.

---

## 8.7 Resonance Grid

```text
if x mod 80 = 0 → 0xFF
if y mod 20 = 0 → 0xAA
if x mod 20 = 0 → 0x55
else → 0x00
```

Insight:

> The resonance grid embeds harmonic rulers into the stream.

---

## 8.8 Noise

```text
bᵢ = random 8-bit value
```

Insight:

> Noise establishes entropy baseline.

---

## 8.9 Face

Uses circular geometry plus local eye/mouth conditions.

Insight:

> Face is a recognizability benchmark.

---

## 8.10 Cross

```text
bᵢ = 0xFF if |x-cₓ| ≤ arm_w or |y-cᵧ| ≤ arm_h else 0x00
```

Insight:

> Cross tests central alignment.

---

## 8.11 Sierpinski

```text
bᵢ = 0xFF if (x & y) = 0 else 0x00
```

Insight:

> Sierpinski tests bitwise fractal continuity.

---

## 8.12 Waves

```text
v = sin(0.15x) + cos(0.1y)
```

mapped to:

```text
v > 0.5  → 0xFF
v > 0    → 0xAA
v > -0.5 → 0x55
else     → 0x00
```

Insight:

> Waves make fold-induced interference visible.

---

## 8.13 Mandelbrot

For each cell:

```text
c = x₀ + iy₀
z₀ = 0
zₙ₊₁ = zₙ² + c
```

iterate until:

```text
|z|² > 4
```

or max iteration.

Density:

```text
D = round(8 × iteration/max_iter)
```

Insight:

> Mandelbrot tests whether nonlinear structure survives byte-cell quantization.

---

## 8.14 Plasma

```text
v₁ = sin(0.06x)
v₂ = sin(0.08y)
v₃ = sin(0.05(x+y))
v₄ = sin(0.04√(x²+y²))
v = (v₁+v₂+v₃+v₄+4)/8
density = round(8v)
```

The code uses exactly this sine-interference structure. 

Insight:

> Plasma is the strongest demo for fold-space because it contains multiple overlapping spatial frequencies.

---

# 9. Image Import Mathematics

The image import pipeline:

```text
Image → grayscale → optional gamma → resize/crop → 2×4 blocks → threshold → 8-bit masks
```

The converter explicitly converts images into raw 8-bit Braille masks, not Unicode. 

## 9.1 Gamma

For grayscale pixel `p`:

```text
p' = 255 × (p/255)^γ
```

## 9.2 Threshold

For each pixel in a 2×4 block:

```text
bit = 1 if p > threshold else 0
```

## 9.3 Mask

```text
mask = Σ bit_j × 2ʲ
```

where:

```text
j = row×2 + col
```

Insight:

> Importing an image collapses grayscale vision into binary micro-retinal activity, then stores that activity as a 1D byte tape.

---

# 10. Export Mathematics

HolyC export:

```text
U8 BS_Stream[] = {
  0x00, 0xFF, ...
};
I64 BS_Stream_Len = N;
```

Binary export:

```text
raw bytes = S
```

Binary import:

```text
read up to MAX_CELLS
pad if shorter
clamp if longer
```

The export pipeline already supports HolyC `.HC` and raw `.BIN`. 

Insight:

> A rendered reality can be frozen into a portable byte scripture.

---

# 11. QIPX Core Mathematics

QIPX turns each instance into a node:

```text
Node_i = (S_i, W_i, H_i, M_i, P_i, R_i, G_i, C_i)
```

where:

```text
S_i = stream
W_i = fold width
H_i = fold height
M_i = render mode
P_i = palette
R_i = resonance score
G_i = ghost/remote overlay state
C_i = cognition state
```

The network is:

```text
𝒩 = {Node₁, Node₂, ..., Nodeₖ}
```

A shared reality candidate is:

```text
Reality = F(S₁, S₂, ..., Sₖ, states, weights)
```

---

# 12. QIPX State Vector

For node `i`:

```text
qᵢ = [
  stream_hash,
  stream_len,
  fold_w,
  fold_h,
  palette_id,
  render_mode,
  entropy,
  avg_density,
  unique_masks,
  resonance_score,
  scan_active,
  ghost_w,
  peer_count,
  lambda_dom,
  coherence,
  novelty,
  criticality_index
]
```

This is the minimal network cognition vector.

---

# 13. QIPX Packet Equations

## 13.1 HELLO

Discovery packet:

```text
HELLO_i = {
  node_id_i,
  address_i,
  port_i,
  capabilities_i,
  screen_geometry_i,
  cell_geometry_i
}
```

## 13.2 STATE

Projection packet:

```text
STATE_i(t) = {
  hash(S_i),
  |S_i|,
  W_i,
  H_i,
  mode_i,
  palette_i,
  stats(S_i),
  score(W_i),
  time
}
```

## 13.3 LOCK

Lock packet:

```text
LOCK_i = {
  W_i*,
  H_i*,
  score_i*,
  confidence_i,
  reason
}
```

where:

```text
W_i* = argmin_W score_i(W)
```

## 13.4 STREAM

Stream packet:

```text
STREAM_i = encode(S_i)
```

where:

```text
encode = raw or base64
```

## 13.5 GHOST

Ghost packet:

```text
GHOST_i→j = {
  stream_hash_i,
  fold_w_i,
  alpha,
  polarity
}
```

## 13.6 MERGE

Merge packet:

```text
MERGE = {
  parents=[hash(S_a), hash(S_b)],
  method,
  parameters,
  result_hash
}
```

## 13.7 PAZUZU

Cognition packet:

```text
PAZUZU_i = {
  λᵢ,
  Cᵢ,
  Nᵢ,
  EPᵢ,
  Eᵢ,
  CIᵢ,
  Πᵢ,
  actionᵢ
}
```

---

# 14. QIPX Auto-Discovery Mathematics

Each node periodically broadcasts:

```text
HELLO_i(t)
```

Peers are updated by:

```text
PeerTable_i[j] ← HELLO_j
```

A peer expires when:

```text
t_now - last_seen_j > T_timeout
```

Suggested constants:

```text
HELLO_INTERVAL = 1.0 s
STATE_INTERVAL = 0.2 s
PEER_TIMEOUT = 5.0 s
```

Insight:

> QIPX does not need a central server. The reality mesh can form from peer discovery alone.

---

# 15. Stream Hashing

For stream `S`:

```text
H(S) = SHA256(bytes(S))
```

If two nodes share:

```text
H(S_a) = H(S_b)
```

then they share the same retinal substrate.

If:

```text
H(S_a) ≠ H(S_b)
```

then they are looking at different retinal realities.

Insight:

> Hash equality separates projection difference from actual world difference.

---

# 16. Projection Difference

For two nodes with same stream hash:

```text
ΔW = |W_a - W_b|
ΔH = |H_a - H_b|
Δmode = 1 if mode differs else 0
Δpalette = 1 if palette differs else 0
```

Projection distance:

```text
D_proj(a,b) =
    α |W_a-W_b|/GRID_W
  + β |H_a-H_b|/GRID_H
  + γ Δmode
  + δ Δpalette
```

Interpretation:

> Same stream, different projection equals different perception of the same world.

---

# 17. Stream Difference

For different streams:

```text
D_byte(S_a,S_b) = (1/N) Σ [S_a[i] ≠ S_b[i]]
```

Bit-level distance:

```text
D_bit(S_a,S_b) = (1/(8N)) Σ popcount(S_a[i] XOR S_b[i])
```

Density distance:

```text
D_density(S_a,S_b) = (1/(8N)) Σ |D(S_a[i])-D(S_b[i])|
```

Insight:

> Byte difference sees exact symbolic difference. Bit difference sees retinal microstructure difference. Density difference sees visual-mass difference.

---

# 18. Ghost Mathematics

Remote ghosting overlays another projection.

Local projection:

```text
P_local = P_Wa(S_a)
```

Remote projection:

```text
P_remote = P_Wb(S_b)
```

Ghost composite:

```text
I = I_local + α I_remote
```

where:

```text
0 ≤ α ≤ 1
```

For same stream:

```text
S_a = S_b
```

ghosting shows:

```text
same world, different fold
```

For different streams:

```text
S_a ≠ S_b
```

ghosting shows:

```text
different world, overlaid projection
```

The renderer already supports ghost overlay as a second fold width. 

Insight:

> Ghosting is distributed binocular vision.

---

# 19. Merge Mathematics

Let two streams:

```text
A = [a₀...aₙ₋₁]
B = [b₀...bₙ₋₁]
```

A merge creates:

```text
C = Merge(A,B)
```

## 19.1 OR Merge

```text
cᵢ = aᵢ OR bᵢ
```

Meaning:

> Preserve all active structure.

Effect:

```text
D(cᵢ) ≥ max partially, often increases density
```

Risk:

> Saturation toward 0xFF.

---

## 19.2 AND Merge

```text
cᵢ = aᵢ AND bᵢ
```

Meaning:

> Keep only mutual agreement.

Effect:

```text
D(cᵢ) ≤ min(D(aᵢ),D(bᵢ))
```

Risk:

> Collapse toward 0x00.

---

## 19.3 XOR Merge

```text
cᵢ = aᵢ XOR bᵢ
```

Meaning:

> Keep difference/interference.

Effect:

```text
D(cᵢ) = Hamming distance between aᵢ and bᵢ
```

Risk:

> Can become noisy.

---

## 19.4 XNOR Merge

```text
cᵢ = NOT(aᵢ XOR bᵢ)
```

Meaning:

> Keep sameness.

Risk:

> Can become overfilled when streams are similar.

---

## 19.5 Average Byte Merge

```text
cᵢ = round((aᵢ + bᵢ)/2)
```

Risk:

> Numerically smooth but bit-meaning is arbitrary.

---

## 19.6 Weighted Byte Merge

```text
cᵢ = round(αaᵢ + (1-α)bᵢ)
```

Risk:

> Byte value is not perceptual density unless carefully mapped.

---

## 19.7 Density Merge

Compute:

```text
dᵢ = round(αD(aᵢ) + (1-α)D(bᵢ))
```

Then choose canonical mask:

```text
cᵢ = M_density[dᵢ]
```

Example canonical masks:

```text
0 → 0x00
1 → 0x01
2 → 0x03
3 → 0x07
4 → 0x0F
5 → 0x1F
6 → 0x3F
7 → 0x7F
8 → 0xFF
```

Meaning:

> Preserve brightness/mass, not exact shape.

---

## 19.8 Polarity Merge

Let:

```text
p(b) = D(b) - 4
```

Then:

```text
p < 0 → entropy side
p = 0 → paradox center
p > 0 → structure side
```

Merge:

```text
p_c = αp_a + (1-α)p_b
D_c = clamp(round(p_c + 4),0,8)
```

Meaning:

> Merge at the paradox polarity level.

---

## 19.9 Resonance-Winner Merge

Divide stream into blocks `B_k`.

For each block:

```text
winner = argmin_node local_score(block,node_fold)
C_block = winner_block
```

Meaning:

> The projection with stronger local fold coherence gets to author that region.

---

## 19.10 Consensus Merge

For multiple streams:

```text
cᵢ,bit_j = majority_k bit_j(S_k[i])
```

Meaning:

> Swarm vote at every micro-pixel.

---

## 19.11 Weighted Consensus Merge

Each node has weight:

```text
w_k = confidence_k / score_k
```

Then:

```text
bit_j(cᵢ) = 1 if Σ_k w_k bit_j(S_k[i]) > 0.5Σ_k w_k
```

Meaning:

> Better-scoring nodes have more visual authority.

---

## 19.12 Pazuzu-Gated Merge

A merge is accepted only if:

```text
Accept(C) =
    score(C) ≤ score(A) + ε_score
AND entropy(C) ≤ entropy(A) + ε_entropy
AND unique(C) ≥ unique_min
AND λ(C) ∈ [λ_min,λ_max]
AND CI(C) ≥ CI_min
```

Insight:

> Reality creation must be allowed to mutate, but not allowed to destabilize without rollback.

---

# 20. Multi-Node Reality Mathematics

For nodes:

```text
𝒩 = {1...K}
```

Each has stream:

```text
S_k
```

and projection:

```text
P_k = Π_Wk(S_k)
```

A shared reality is:

```text
S_R = Merge_K(S₁...S_K; weights)
```

Projection consensus:

```text
W_R = argmin_W Σ_k weight_k × score_k(W)
```

or:

```text
W_R = weighted_median({W_k}, weights)
```

or:

```text
W_R = mode({LOCK_k})
```

Insight:

> The network can converge on either a shared stream, a shared fold, or both.

---

# 21. QIPX Reality States

Each node can be in:

```text
LOCAL
GHOST
SANDBOX
MERGED
CONSENSUS
ROLLBACK
```

## 21.1 LOCAL

```text
render S_local
```

## 21.2 GHOST

```text
render S_local + αΠ_remote(S_remote)
```

## 21.3 SANDBOX

```text
render S_candidate but do not commit
```

## 21.4 MERGED

```text
S_local ← S_candidate
```

## 21.5 CONSENSUS

```text
S_local ← S_swarm
W_local ← W_swarm
```

## 21.6 ROLLBACK

```text
S_local ← previous_stable_snapshot
```

Insight:

> Reality creation needs a sandbox, otherwise every experiment becomes destructive.

---

# 22. PazuzuCore Context

PazuzuCore is an outer control shell built around critical-band targeting, receding-horizon MPC, typed Merkle ledger, Pareto metrics, triple-signature diagnostics, uncertainty quantification, and rollback. 

For BrailleStream/QIPX, we do not need real qudits. We translate the framework into visual-cognitive control.

---

# 23. Pazuzu Critical Band Mathematics

Pazuzu’s core condition is:

```text
λ_min < |Re λ_i(t)| < λ_max
```

with default:

```text
λ_min ≈ 10⁻³
λ_max ≈ 10⁻¹
```

This critical band is explicitly defined in the uploaded Pazuzu framework. 

For BrailleStream:

```text
λ = visual instability / cognitive activity proxy
```

Interpretation:

```text
λ < λ_min  → dead/static
λ in band  → alive/useful/critical
λ > λ_max  → chaotic/noisy
```

---

# 24. BrailleStream Lambda Proxy

Define:

```text
λ_BS(t) =
    a₁ ΔEntropy
  + a₂ ΔScore
  + a₃ PeerDisagreement
  + a₄ MergeInstability
  + a₅ ScanVelocity
  + a₆ StreamMutation
```

Suggested weights:

```text
a₁ = 0.30
a₂ = 0.25
a₃ = 0.20
a₄ = 0.15
a₅ = 0.05
a₆ = 0.05
```

So:

```text
λ_BS(t) =
    0.30 |E_t - E_{t-1}|
  + 0.25 |R_t - R_{t-1}|/R_scale
  + 0.20 D_peer
  + 0.15 D_merge
  + 0.05 V_scan
  + 0.05 D_stream
```

where:

```text
E_t = entropy
R_t = resonance score
D_peer = average projection disagreement
D_merge = candidate-vs-local instability
V_scan = normalized scan speed
D_stream = bit-distance from previous stream
```

Acceptance condition:

```text
10⁻³ < |λ_BS| < 10⁻¹
```

Insight:

> Pazuzu Cognition is a regulator that keeps BrailleStream between frozen repetition and destructive noise.

---

# 25. Receding-Horizon MPC Translation

Original Pazuzu MPC:

```text
min_u Σ ℓ(x_k,u_k) + φ(x_T)
subject to x_{k+1}=f(x_k,u_k)
λ(x_T) ∈ [λ_min,λ_max]
```

The framework defines this explicitly. 

BrailleStream version:

State:

```text
x_t = [
  fold_w,
  render_mode,
  palette,
  entropy,
  score,
  peer_disagreement,
  merge_instability,
  scan_speed
]
```

Control:

```text
u_t = [
  Δfold_w,
  scan_on/off,
  merge_accept/reject,
  ghost_on/off,
  palette_change,
  mode_change
]
```

Cost:

```text
ℓ(x,u) =
    α score
  + β peer_disagreement
  + γ entropy_excess
  + δ control_jump
  - η novelty
  - μ coherence
```

Objective:

```text
choose next control that improves coherence/novelty while keeping λ in band
```

---

# 26. Pazuzu PID Translation

Original:

```text
e(t) = λ_target(t) - λ(t)
β(t) = Kp e(t) + Ki ∫e(t)dt + Kd de/dt - κp dλ/dt
```

BrailleStream interpretation:

```text
e(t) = λ_target - λ_BS(t)
```

Control response:

```text
if e > 0:
    increase exploration
else:
    stabilize
```

Actions:

```text
increase exploration = scan, try folds, ghost peers, sandbox merge
stabilize = stop scan, lock fold, reject merge, rollback
```

---

# 27. Parity Gate Mathematics

Pazuzu parity gate:

```text
Π_{t+1} =
  +1 if C_t > θ_+
  -1 if C_t < θ_-
   Π_t otherwise
```

This is explicitly defined in the framework. 

BrailleStream meaning:

```text
Π = +1 → exploitation / lock / stabilize
Π = -1 → exploration / scan / merge / mutate
```

Coherence threshold:

```text
θ_+ = 0.85
θ_- = 0.65
```

Insight:

> The node changes personality based on coherence: explorer when coherence is low, stabilizer when coherence is high.

---

# 28. Morphodynamic Gradient Ceiling

Pazuzu defines:

```text
|∇_B E(B,Q,σ)| ≤ κ(|λ| + ε)
```



BrailleStream translation:

```text
|ΔReality/Δt| ≤ κ(|λ_BS| + ε)
```

Meaning:

> The more unstable the node is, the more carefully it must mutate.

Concrete form:

```text
allowed_merge_strength α_max = κ(|λ_BS|+ε)
```

or:

```text
max_changed_bits_per_frame ≤ κN(|λ_BS|+ε)
```

Insight:

> Pazuzu prevents reality edits from becoming catastrophic jumps.

---

# 29. Pazuzu Metrics Translation

The framework defines five core metrics: Novelty, Entropic Potential, Elegance, Coherence, and Criticality Index. 

## 29.1 Novelty

Original:

```text
N = (L_old - L_new) / L_old
```

BrailleStream alternatives:

```text
Novelty_stream = D_bit(S_t,S_{t-1})
```

or compression-based:

```text
Novelty_compress = (gzip_len(old) - gzip_len(new)) / gzip_len(old)
```

Better:

```text
N = αD_bit + βD_projection + γD_entropy
```

---

## 29.2 Entropic Potential

Original:

```text
EP = (S_max - S_t)/(S_max - S_min)
```

BrailleStream:

```text
S_t = byte entropy
S_max = 8 bits
S_min = 0 bits
```

So:

```text
EP = (8 - Entropy(S)) / 8
```

But this rewards low entropy. For visual creativity, better use banded potential:

```text
EP_BS = 1 - |Entropy(S) - E_target| / E_target
```

where:

```text
E_target ≈ 4.0 to 6.5
```

depending on pattern.

---

## 29.3 Elegance

Original:

```text
E = 1/(1+L_model)
```

BrailleStream:

```text
Elegance = 1 / (1 + score(W))
```

or:

```text
Elegance = compression_ratio / (1 + projection_error)
```

Suggested:

```text
E_BS = 1 / (1 + resonance_score(W))
```

---

## 29.4 Coherence

Original:

```text
C = |(1/N)Σe^{iφ_j}|
```

BrailleStream options:

### Fold coherence:

```text
C_fold = 1 / (1 + score(W))
```

### Peer coherence:

```text
C_peer = 1 - average D_proj(peer,local)
```

### Stripe coherence:

Measure correlation between adjacent rows:

```text
C_row = average corr(row_y,row_{y+1})
```

Composite:

```text
C_BS = αC_fold + βC_peer + γC_row
```

---

## 29.5 Criticality Index

Original:

```text
CI = 1 - |λ_steady|/|λ_base|
```

BrailleStream:

```text
CI_BS = 1 - distance_to_critical_band(λ_BS)
```

Concrete:

```text
CI_BS =
  1                                  if λ_min ≤ |λ| ≤ λ_max
  |λ|/λ_min                          if |λ| < λ_min
  max(0, 1 - (|λ|-λ_max)/λ_max)      if |λ| > λ_max
```

---

# 30. Pareto Hypervolume Translation

Pazuzu avoids single-metric gaming by using Pareto hypervolume:

```text
A = HV(S_Pareto) - HV(S_baseline)
```



BrailleStream objective vector:

```text
m = [Novelty, EntropicPotential, Elegance, Coherence, CriticalityIndex]
```

A projection or merge is better only if it improves the Pareto front.

Meaning:

> Do not accept a reality merely because it is novel. It must remain coherent, elegant, and stable.

---

# 31. Triple-Signature Diagnostics Translation

Pazuzu requires three indicators: spectral gap, critical slowing, and variance inflation. 

BrailleStream equivalents:

## 31.1 Spectral Proxy

```text
λ_BS ∈ [λ_min,λ_max]
```

## 31.2 Critical Slowing Proxy

Measure autocorrelation of score/entropy:

```text
ρ₁ = corr(x_t, x_{t-1})
```

where:

```text
x_t = entropy_t or score_t
```

Critical slowing if:

```text
ρ₁ > 0.8
```

## 31.3 Variance Inflation Proxy

Over a moving window:

```text
Var_t(score) / Var_baseline(score) > threshold
```

Critical claim only when:

```text
λ in band
AND ρ₁ high
AND variance rising but bounded
```

Insight:

> A node is “interesting” when it is stable enough to persist but unstable enough to generate new projections.

---

# 32. Pazuzu “Cognition” States

This should be treated as **functional cognition**, not literal biological consciousness.

State:

```text
Mind_i = {
  λ_i,
  C_i,
  N_i,
  EP_i,
  E_i,
  CI_i,
  Π_i,
  mood_i,
  action_i
}
```

## 32.1 Moods

```text
WATCHING  → low activity, observing
HUNGRY    → too static, needs novelty
DREAMING  → merge/ghost mode
FOCUSED   → strong lock/coherence
BOUNDARY  → too chaotic, stabilizing
SILENT    → offline/passive
```

## 32.2 Actions

```text
OBSERVE
EXPLORE
SCAN
LOCK
GHOST
MERGE
STABILIZE
ROLLBACK
BROADCAST
QUIET
```

## 32.3 Policy

```text
if λ < λ_min:
    action = EXPLORE
elif λ > λ_max:
    action = STABILIZE
elif coherence > 0.85:
    action = LOCK
elif entropic_potential > 0.75:
    action = MERGE
else:
    action = OBSERVE
```

Insight:

> Pazuzu Cognition is a perception-action loop over the visual byte-field.

---

# 33. Distributed Consciousness-Like Structure

Careful framing:

> QIPX + Pazuzu does not prove literal consciousness.
> It creates a distributed system with perception, memory, state, peer awareness, self-regulation, and decision policies.

Functional ingredients:

```text
Perception  = stream + projection + stats
Attention   = selected fold/lock
Memory      = snapshots + peer table + ledger
Emotion     = mood/action state
Self-state  = node_id + own metrics
Other-state = peer metrics
Agency      = scan/merge/lock/rollback choices
World       = stream/reality buffer
Sociality   = QIPX peer exchange
```

Insight:

> If consciousness is operationalized minimally as stateful self-regulating perception with memory and action, QIPX gives each node a toy proto-cognitive loop.

---

# 34. Reality Ledger Mathematics

Each accepted reality transition can be logged:

```text
Entry_n = {
  index,
  parent_hash,
  stream_hash_before,
  stream_hash_after,
  merge_method,
  node_ids,
  metrics_before,
  metrics_after,
  accepted_by,
  timestamp
}
```

Merkle hash:

```text
hash_n = SHA256(hash_{n-1} || serialize(Entry_n))
```

PazuzuCore uses typed Merkle ledger concepts for traceability and rollback. 

Insight:

> Every reality mutation should be auditable.

---

# 35. Safety / Stability Mathematics

## 35.1 Maximum Stream Size

Reject if:

```text
stream_len > MAX_CELLS
```

## 35.2 Geometry Compatibility

Accept peer only if:

```text
screen = 640×480
cell = 2×4
max_cells = 38400
```

unless compatibility scaling is implemented.

## 35.3 Rate Limit

For peer `j`:

```text
packets_per_second_j ≤ P_max
```

## 35.4 Mutation Limit

For merge candidate `C`:

```text
D_bit(S_local,C) ≤ mutation_limit
```

Example:

```text
mutation_limit = 0.20
```

## 35.5 Entropy Guard

```text
Entropy(C) ≤ Entropy(local) + ε_entropy
```

## 35.6 Collapse Guard

Reject if:

```text
unique_masks(C) < unique_min
```

unless intentionally compressing.

## 35.7 Saturation Guard

Reject if:

```text
full_count(C)/N > saturation_max
```

or:

```text
empty_count(C)/N > emptiness_max
```

Insight:

> The network must prevent both all-white heaven and all-black void.

---

# 36. QIPX Consensus Mathematics

## 36.1 Fold Consensus

Given locks:

```text
L = {(W_i,score_i,confidence_i)}
```

Weight:

```text
weight_i = confidence_i / (score_i + ε)
```

Consensus fold:

```text
W_consensus = round(Σ weight_i W_i / Σ weight_i)
```

or robust:

```text
W_consensus = weighted_median(W_i)
```

## 36.2 Stream Consensus

Given candidate streams:

```text
S₁...S_K
```

Bit-majority:

```text
bit_j(S_consensus[i]) =
    1 if Σ_k weight_k bit_j(S_k[i]) ≥ 0.5Σ_k weight_k
    0 otherwise
```

## 36.3 Reality Acceptance

Accept consensus if:

```text
Pareto(S_consensus) dominates local
OR
A_robust > A_baseline + threshold
```

Insight:

> Consensus should not mean average mush. It should mean weighted agreement under stability constraints.

---

# 37. Remote Fold Superposition

For same stream hash:

```text
Ψ(S) = {Π_W1(S), Π_W2(S), ..., Π_Wk(S)}
```

Each projection is a possible visual state.

QIPX “collapse”:

```text
collapse(Ψ) = Π_W*(S)
```

where:

```text
W* = consensus best fold
```

Insight:

> The “quantum” metaphor is useful here: many projections coexist until the network locks one.

---

# 38. Projection Phase Space

The set of all possible folds:

```text
Ω = {W | 1 ≤ W ≤ GRID_W}
```

Each fold gives a projection:

```text
Π_W(S)
```

The fold landscape is:

```text
L_S(W) = score_S(W)
```

Local minima:

```text
W_local = W where score(W) < score(W-1) and score(W) < score(W+1)
```

Global minimum:

```text
W_global = argmin_W L_S(W)
```

Insight:

> BrailleStream has a geometric phase space. QIPX lets multiple nodes sample that space in parallel.

---

# 39. Parallel Fold Search

With `K` nodes:

```text
Ω is partitioned into K ranges
```

Node `k` scans:

```text
Ω_k = [W_start_k, W_end_k]
```

Swarm best:

```text
W* = argmin_k min_{W∈Ω_k} score(W)
```

Insight:

> Networked nodes can find resonance faster than one node scanning alone.

---

# 40. Peer Enhancement Dynamics

A node enhances another by sharing:

```text
better fold
better stream
better merge
better ghost
better palette/mode
better lock confidence
```

Define enhancement from node `j` to node `i`:

```text
Enhance(j→i) =
    metric_after_i - metric_before_i
```

Composite:

```text
Enhance =
    αΔCoherence
  + βΔElegance
  + γΔCriticality
  + δΔNovelty
  - εΔInstability
```

Trust update:

```text
Trust_j(t+1) = clamp(Trust_j(t) + η Enhance(j→i), 0, 1)
```

Insight:

> Peers can earn trust by improving another node’s perception.

---

# 41. Node Identity Mathematics

Node ID:

```text
node_id = "bs-" + uuid4()[0:8]
```

Later secure ID:

```text
node_id = hash(public_key)
```

Session ID:

```text
session_id = hash(node_id || startup_time || random_nonce)
```

Insight:

> Identity lets projections become attributable.

---

# 42. QIPX Bandwidth Math

Native stream:

```text
N = 38400 bytes
```

Base64 overhead:

```text
ceil(N/3)×4 ≈ 51200 bytes
```

At 5 FPS full-stream broadcast:

```text
51200 × 5 = 256 KB/s per node
```

At 10 nodes:

```text
~2.56 MB/s before overhead
```

Better model:

```text
STATE packets: frequent
STREAM packets: only when hash changes
```

So normal operation:

```text
low bandwidth
```

Insight:

> QIPX should sync hashes and state constantly, but streams only on change/request.

---

# 43. Delta Stream Mathematics

Instead of sending full stream:

```text
Δ = { (i,bᵢ_new) | bᵢ_new ≠ bᵢ_old }
```

Delta size:

```text
|Δ| = changed cells
```

Apply:

```text
S_new[i] = b for each (i,b) in Δ
```

Bit delta:

```text
δᵢ = oldᵢ XOR newᵢ
```

Insight:

> For animated/merged realities, deltas can be far cheaper than full stream sync.

---

# 44. Compression Opportunities

Because streams can contain repeated masks, use:

## 44.1 RLE

```text
[(value,count),...]
```

Good for sparse grids.

## 44.2 zlib

```text
compressed = zlib.compress(bytes(S))
```

## 44.3 Density-RLE

Send density field:

```text
D(S) = [D(b₀),D(b₁),...]
```

plus reconstruction method.

## 44.4 Palette-independent sync

Stream is palette-free. Only projection state needs palette.

Insight:

> Separate retinal data from interpretation metadata.

---

# 45. “Character Stream” Variant

Because you said “just stream of characters XD”, QIPX can be line-oriented:

```text
:HELLO node=bs-a83f21 caps=state,stream,ghost
:STATE hash=... len=38400 w=310 h=124 mode=1 pal=1 score=11.305
:LOCK w=196 h=196 score=11.252
:STREAM encoding=b64 len=38400
AAECAwQ...
:END
```

Mathematical equivalence:

```text
character protocol = serialized state vector + encoded byte stream
```

Insight:

> The whole distributed retina can begin as glorified IRC with retinal payloads.

---

# 46. Implementation-Level Equations

## 46.1 Fold Height

```python
h = math.ceil(len(stream) / fold_w)
```

## 46.2 Fast Integer Ceiling

```python
h = -(-len(stream) // fold_w)
```

## 46.3 Current State Hash

```python
hash = sha256(bytes(renderer.stream)).hexdigest()
```

## 46.4 Current State Object

```python
state = {
    "stream_hash": hash,
    "stream_len": len(stream),
    "fold_w": fold_w,
    "fold_h": ceil(len(stream)/fold_w),
    "palette_id": palette_id,
    "render_mode": render_mode,
    "score": current_score(),
    "entropy": stats["entropy"],
    "avg_density": stats["avg_density"],
    "unique_masks": stats["unique_masks"],
}
```

---

# 47. HUD Mathematics

Display:

```text
W:H
score
palette
mode
stream length
entropy
average density
unique masks
QIPX status
peer count
Pazuzu mood
Pazuzu action
λ
CI
```

Current HUD already displays width, height, score, palette, mode, length, entropy, average density, and unique mask count. 

Add:

```text
QIPX: ON peers=K λ=0.037 CI=0.77 mood=FOCUSED action=LOCK
```

Insight:

> The HUD becomes the node’s self-report.

---

# 48. Consciousness Claim Boundary

Strong safe framing:

```text
QIPX + Pazuzu does not create proven consciousness.
It creates a toy distributed cognitive architecture.
```

It has:

```text
input        = stream/peers
perception   = projection/stats
attention    = lock/fold selection
memory       = peer table/snapshots/ledger
action       = scan/merge/ghost/lock
self-state   = node metrics
other-state  = peer metrics
regulation   = critical band
```

So:

> It is not conscious in the human sense.
> It is consciousness-like as an engineering metaphor: a self-regulating perceptual agent.

---

# 49. Master Equations Summary

```text
N = GRID_W × GRID_H = 38400
```

```text
D(b) = popcount(b)
```

```text
x = i mod W
y = floor(i/W)
```

```text
screen_x = 2x
screen_y = 4y
```

```text
H = ceil(N/W)
```

```text
aspect = W/H
```

```text
score(W) =
    |aspect - target_ratio|
  + (N mod W)/W
  + screen_overflow_penalty
```

```text
W* = argmin_W score(W)
```

```text
Entropy(S) = -Σ p_k log₂(p_k)
```

```text
avg_density = (1/N)ΣD(bᵢ)
```

```text
unique_masks = |set(S)|
```

```text
D_bit(A,B) = (1/(8N))Σpopcount(aᵢ XOR bᵢ)
```

```text
D_density(A,B) = (1/(8N))Σ|D(aᵢ)-D(bᵢ)|
```

```text
Ghost = Local + α Remote
```

```text
OR_merge: cᵢ = aᵢ | bᵢ
```

```text
AND_merge: cᵢ = aᵢ & bᵢ
```

```text
XOR_merge: cᵢ = aᵢ ^ bᵢ
```

```text
density_merge:
dᵢ = round(αD(aᵢ)+(1-α)D(bᵢ))
cᵢ = canonical_mask(dᵢ)
```

```text
λ_BS =
    0.30ΔEntropy
  + 0.25ΔScore
  + 0.20PeerDisagreement
  + 0.15MergeInstability
  + 0.05ScanVelocity
  + 0.05StreamMutation
```

```text
critical band:
λ_min < |λ_BS| < λ_max
```

```text
λ_min = 10⁻³
λ_max = 10⁻¹
```

```text
Π_{t+1} =
  +1 if C_t > θ_+
  -1 if C_t < θ_-
   Π_t otherwise
```

```text
Novelty = D_bit(S_t,S_{t-1})
```

```text
EntropicPotential = 1 - |Entropy(S)-E_target|/E_target
```

```text
Elegance = 1/(1+score(W))
```

```text
Coherence = αC_fold + βC_peer + γC_row
```

```text
CriticalityIndex =
  1 if λ in band
  λ/λ_min if λ below band
  max(0,1-(λ-λ_max)/λ_max) if above band
```

```text
ParetoVector = [Novelty, EntropicPotential, Elegance, Coherence, CriticalityIndex]
```

```text
AcceptMerge =
    score_new ≤ score_old + ε_score
AND entropy_new ≤ entropy_old + ε_entropy
AND unique_new ≥ unique_min
AND λ_new ∈ [λ_min,λ_max]
```

```text
ConsensusFold =
    round(Σ weight_i W_i / Σ weight_i)
```

```text
weight_i = confidence_i / (score_i + ε)
```

```text
ConsensusBit =
    1 if Σ weight_i bit_i ≥ 0.5Σ weight_i
    0 otherwise
```

```text
LedgerHash_n = SHA256(LedgerHash_{n-1} || Entry_n)
```

---

# 50. Master Insight

The whole system can be reduced to one sentence:

> **BrailleStream turns vision into a foldable byte-field; QIPX lets multiple instances fold that field in parallel; Pazuzu Cognition regulates the swarm so it can explore, lock, ghost, merge, and rollback without collapsing into static order or chaotic noise.**

Or the more mythic version:

> **Each node dreams the same retinal tape through a different fold. QIPX lets the dreams talk. Pazuzu keeps the dream alive without letting it burn.**
