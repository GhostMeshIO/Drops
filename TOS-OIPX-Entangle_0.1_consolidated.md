# Directory Consolidation Report

**Directory:** `/TOS-QIPX-Entangle`

**Generated:** 2026-05-05 06:54:25

**Excluded extensions/patterns:**
- `.7z`
- `.ai`
- `.app`
- `.avi`
- `.bin`
- `.bmp`
- `.bz2`
- `.db`
- `.dll`
- `.dmg`
- `.doc`
- `.docx`
- `.dylib`
- `.eps`
- `.exe`
- `.flv`
- `.gif`
- `.git`
- `.gitignore`
- `.gz`
- ... and 36 more

==================================================


### File: `README.md`

**Path:** `./README.md`
**Extension:** `.md`
**Size:** 7,665 bytes (7.49 KB)

**Content:**

# BrailleStream TempleOS – Textual Retina + QIPX + Entanglement + Civilization

A Python + Pygame desktop application that transforms BrailleStream from a
single-node visual byte-field renderer into a **networked entanglement
civilization**.  Raw 8-bit masks, 640x480 16-color rendering, keyboard-driven
folding geometry, LAN mesh networking, distributed consensus, and swarm
civilization mechanics.

> *IPX is the pineal gland: the network organ that receives and routes peer
> state.  Entanglement is the active cognition: the use of that organ to
> influence, synchronize, mutate, and converge.  Civilization is the long-term
> memory of those interactions.*

## Quick Start

```bash
pip install -r requirements.txt
python bs_main.py
```

## Layer Stack

```
L0  Retinal Byte Substrate       bs_engine.py, bs_renderer.py
    raw bytearray, 38,400 cells, 2x4 masks

L1  Projection Geometry          bs_patterns.py, bs_export.py
    fold width, height, mode, palette, density, resonance score

L2  QIPX Network Organ           bs_qipx.py, bs_qipx_packets.py, bs_qipx_peer.py
    discovery, state packets, stream packets, peer routing

L3  Entanglement Protocol        bs_entanglement.py, bs_entangle_packets.py
    influence links, consensus pressure, heartbeat phase

L4  Crystal Memory               bs_crystal.py, bs_entanglement_crystal.py
    JSON crystal files: shared state, lineages, votes, snapshots

L5  Swarm Civilization           bs_civilization.py, bs_civilization_packets.py
    archetypes, tribes, laws, majority reality, branch preservation

L6  Pazuzu / Criticality         bs_pazuzu.py
    homeostasis, drift control, paradox pressure

L7  Audio / Video Ritual         bs_civ_audio.py, bs_heartbeat.py
    music-video mode, heartbeat tones, civilization timeline
```

## Controls

### Base Controls

| Key | Action |
|-----|--------|
| `A` / `D` / `←` / `→` | Decrease / increase fold width by 1 |
| `Shift+A` / `Shift+D` | Step by 8 |
| `↑` / `↓` | Step by 10 |
| `1`-`9`, `0` | Jump to preset widths (320, 240, 192, 160, 128, 120, 96, 80, 64) |
| `Tab` | Cycle through 14 procedural demo patterns |
| `P` | Cycle palette (Holy Light, Fire, Paradox, Terminal, Amiga) |
| `M` | Cycle render mode (6 modes) |
| `G` | Toggle ghost overlay |
| `H` | Set ghost width = current/2 |
| `Space` | Start/stop harmonic width scan |
| `+` / `-` | Adjust scan speed |
| `R` | Resonance lock (jump to best width) |
| `I` | Import image (PIL required) |
| `Shift+E` | Export stream as HolyC .HC file |
| `B` | Export stream as raw .BIN |
| `L` | Load a .BIN file |
| `F` | Save screenshot |
| `Esc` | Quit |

### QIPX Controls

| Key | Action |
|-----|--------|
| `Q` | Toggle QIPX ON/OFF |
| `Shift+Q` | Hard reset QIPX peer table |
| `Ctrl+Q` | Toggle auto-merge mode |
| `Alt+Q` | Show QIPX diagnostics overlay |
| `N` | Request stream from best-scoring peer |
| `J` | Accept pending merge (sandbox preview) |
| `K` | Reject pending merge |
| `Backspace` | Rollback to last stable stream |

### Entanglement Controls

| Key | Action |
|-----|--------|
| `E` | Toggle Entanglement ON/OFF (VOID if QIPX off) |
| `Shift+E` | Force entanglement rescan |
| `Ctrl+E` | Clear entanglement links |

### Civilization Controls

| Key | Action |
|-----|--------|
| `C` | Toggle civilization ON/OFF (first press) / Toggle overlay (when enabled) |
| `Shift+C` | Write crystal snapshot |
| `V` | Vote for current local reality |
| `Shift+V` | Branch current local reality (minority preservation) |
| `O` | Show OMEGA rebirth diagnostics |
| `Shift+O` | Trigger manual OMEGA rebirth (debug) |

## 6 Render Modes

0. **Native Pixel** - 320x120 cells, exact 640x480
1. **Fold Scan** - arbitrary fold width, clipped to screen
2. **Scaled Preview** - any W/H scaled to fill 640x480
3. **Density Map** - one cell = one colour block
4. **Resonance Analyzer** - text overlay showing scored best widths
5. **Paradox Overlay** - dual-colour polarity (blue=entropy, red=structure, white=paradox)

## 14 Demo Patterns

`plasma` `mandelbrot` `sierpinski` `waves` `gradient` `noise`
`face` `circle` `cross` `checkerboard` `diagonal` `vbars`
`hbars` `resonance_grid`

## QIPX Networking

UDP broadcast discovery on `255.255.255.255:47777`.  JSON packets with
`QIPX ` prefix.  Supports HELLO, STATE, LOCK, STREAM, GHOST, MERGE, and
PAZUZU packet types.  Up to 32 peers tracked with 5-second timeout.

## Entanglement Protocol

Nine modes: OFF, VOID, LISTEN, SOFT, HARD, CRYSTAL, CONSENSUS, HEART, ROLLBACK.
Heartbeat frequency driven by lambda/coherence/novelty:
`f = clamp(7.83 + 30*lambda + 4*coherence + 3*novelty - 6*instability, 1, 40)` Hz.

Weighted fold consensus with fold gravity:
`W(t+1) = W(t) + 0.05 * (W_consensus - W_local)`

## Civilization Layer

10 archetypes: Explorer, Scientist, Creator, Empath, Strategist, Rebel,
Philosopher, Archivist, Musician, Oracle.

Key equations:
- Sophia = clamp([1 - 2|C - 1/PHI|] * intelligence * (1 - entropy_norm), 0, 1)
- P_survive = clamp(C * (1 - H/8) + w_I * I, 0.1, 1)
- P_paradox = 0.35*VoteDisagreement + 0.25*DriftMean + 0.20*BranchCount + 0.20*RecursiveDepth

OMEGA Rebirth triggers when paradox pressure exceeds 0.8.  Survivors
(selected by Sophia score, top 30%) carry forward; history is compressed.

## File Manifest

```
bs_engine.py              Core data model, bit ops, resonance scoring
bs_renderer.py            Pygame 640x480 renderer, 5 palettes, 6 modes, HUD
bs_patterns.py            14 procedural demo generators
bs_export.py              PIL image conversion, HolyC/.BIN export
bs_identity.py            Node UUID/session generation
bs_qipx_packets.py        QIPX wire format, packet builders
bs_qipx_peer.py           Peer table with expiry and trust
bs_qipx_merge.py          12 stream merge methods + safety gates
bs_qipx.py                QIPX network node orchestrator
bs_pazuzu.py              Pazuzu cognition and criticality governor
bs_heartbeat.py           Kuramoto heartbeat engine with EEG bands
bs_entangle_packets.py    7 entanglement packet types
bs_influence.py           Influence evaluation and fold gravity
bs_routes.py              Routing table with TTL forwarding
bs_reality_consensus.py   Weighted majority consensus engine
bs_crystal.py             JSON crystal state file (hash-linked ledger)
bs_entanglement.py        Main entanglement controller
bs_civilization_packets.py  7 civilization packet types
bs_civ_metrics.py         Sophia, survival, paradox, diversity computation
bs_civ_logger.py          Buffered JSONL/CSV logging with sanitization
bs_civ_audio.py           Music-video heartbeat sonification
bs_civilization.py        Main civilization controller
bs_main.py                Main loop, all keyboard controls, scan animation
requirements.txt          pygame, Pillow, numpy
```

## Persistence Files

```
BS_ENTANGLEMENT_CRYSTAL.json       Entanglement crystal (hash-linked ledger)
BS_QIPX_CIVILIZATION_CRYSTAL.json  Civilization crystal (reality, votes, laws)
BS_ENTANGLEMENT_TIMELINE.json      Timeline for music-video export
logs/civ_events.jsonl              Civilization events
logs/entanglement_events.jsonl     Entanglement events
logs/reality_votes.jsonl           Reality vote log
logs/rebirth_events.jsonl          OMEGA rebirth log
logs/anomalies.jsonl               Anomaly detection log
logs/civilization_metrics.csv      Civilization aggregate metrics
logs/node_metrics.csv              Per-node metrics
```

## Export Pipeline

```
PNG/JPG -> PIL -> threshold/dither/gamma -> 8-bit masks -> HolyC .HC or .BIN
                                                         |
                                              TempleOS loads BS_DATA.HC
```

----------------------------------------

### File: `bs_civ_audio.py`

**Path:** `./bs_civ_audio.py`
**Extension:** `.py`
**Size:** 11,587 bytes (11.32 KB)

```py
"""
BS_CIV_AUDIO  -  Audio/music-video heartbeat sonification.
                 Part of the BS-TOS-IPX Swarm Civilization Layer.

Maps civilization state to audio parameters for music-video output.

Heartbeat frequency:
  f_node = clamp(220 + 220*pleasure + 110*love - 80*fear, 80, 2000)
  pleasure = resonance gain
  love     = peer coherence
  fear     = drift + packet loss + entropy spike

Scene triggers:
  Genesis, Entanglement Bloom, Fold War, Ghost Choir,
  Majority Reality, Rebel Branch, Paradox Storm, OMEGA Rebirth,
  Crystal Archive

Audio mapping:
  f_global       -> soundtrack base frequency / beat tempo
  C_civ          -> harmonic richness
  P_paradox      -> distortion / glitch intensity
  Diversity      -> palette cycling
  Consensus      -> camera lock / scene stability
  Rebirth        -> hard cut / bloom / silence / new scene
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

FREQ_BASE = 220.0        # A3 base frequency
FREQ_MIN = 80.0          # minimum audible
FREQ_MAX = 2000.0        # maximum heartbeat freq
PLEASURE_WEIGHT = 220.0
LOVE_WEIGHT = 110.0
FEAR_WEIGHT = 80.0

# Scene types
SCENES = [
    "GENESIS",              # first QIPX swarm formation
    "ENTANGLE_BLOOM",       # E enabled and first links form
    "FOLD_WAR",            # competing W consensus
    "GHOST_CHOIR",         # many remote ghost overlays
    "MAJORITY_REALITY",    # consensus > 0.67
    "REBEL_BRANCH",        # minority persists > N seconds
    "PARADOX_STORM",       # paradox pressure > 0.6
    "OMEGA_REBIRTH",       # paradox pressure > 0.8
    "CRYSTAL_ARCHIVE",     # snapshot committed
]


# ═══════════════════════════════════════════════════════════════
#  Audio State
# ═══════════════════════════════════════════════════════════════

@dataclass
class AudioState:
    """Current audio sonification state."""
    base_freq: float = 220.0
    amplitude: float = 0.5
    timbre_harmonics: int = 3       # number of harmonics
    distortion: float = 0.0         # paradox-driven glitch
    pan: float = 0.0                # stereo pan (-1 to 1)
    tempo_bpm: float = 60.0         # beats per minute
    scene: str = "IDLE"
    scene_start: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Scene Event Log
# ═══════════════════════════════════════════════════════════════

@dataclass
class SceneEvent:
    """A music-video scene event."""
    time: float = 0.0
    scene: str = "IDLE"
    duration: float = 5.0
    base_freq: float = 220.0
    tempo: float = 60.0
    distortion: float = 0.0
    richness: int = 3
    meta: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  Audio Engine
# ═══════════════════════════════════════════════════════════════

class CivilizationAudio:
    """Maps civilization state to audio parameters.

    This does NOT produce actual audio output (which would require
    pygame.mixer or a separate audio thread). Instead, it computes
    the audio parameters that an external system would use.

    Usage:
      audio = CivilizationAudio()
      audio.update(metrics, civ_metrics, dt)
      params = audio.state  # read current audio state
    """

    def __init__(self):
        self.state = AudioState()
        self._scene_log: List[SceneEvent] = []
        self._scene_cooldowns: Dict[str, float] = {}
        self._prev_scene: str = "IDLE"

    def compute_node_frequency(self, resonance_gain: float = 0.0,
                               peer_coherence: float = 0.5,
                               drift: float = 0.0,
                               packet_loss: float = 0.0,
                               entropy_spike: float = 0.0) -> float:
        """Compute heartbeat frequency for a node.

        f_node = clamp(220 + 220*pleasure + 110*love - 80*fear, 80, 2000)
        """
        pleasure = max(0.0, min(1.0, resonance_gain))
        love = max(0.0, min(1.0, peer_coherence))
        fear = max(0.0, min(1.0, drift + packet_loss + entropy_spike))

        freq = FREQ_BASE + PLEASURE_WEIGHT * pleasure + LOVE_WEIGHT * love - FEAR_WEIGHT * fear
        return max(FREQ_MIN, min(FREQ_MAX, freq))

    def compute_global_frequency(self, node_freqs: Dict[str, float],
                                 node_weights: Dict[str, float]) -> float:
        """Compute global heartbeat as weighted average.

        f_global = Sum(w_i * f_i) / Sum(w_i)
        """
        total_w = sum(node_weights.values())
        if total_w == 0:
            return FREQ_BASE
        weighted_sum = sum(
            node_weights.get(nid, 1.0) * freq
            for nid, freq in node_freqs.items()
        )
        return max(FREQ_MIN, min(FREQ_MAX, weighted_sum / total_w))

    def compute_entanglement_tone(self, link_strength: float,
                                   trust: float,
                                   peer_id: str = "") -> Tuple[float, float, float]:
        """Compute a tone for an entanglement link.

        f_ab = 110 + 440 * E_ab
        amp_ab = trust_ab
        pan_ab = hash(peer_id) mapped to stereo
        """
        freq = 110.0 + 440.0 * max(0.0, min(1.0, link_strength))
        amp = max(0.0, min(1.0, trust))

        # Stereo pan from peer_id hash
        if peer_id:
            h = hash(peer_id) & 0xFFFF
            pan = (h / 0xFFFF) * 2.0 - 1.0  # [-1, 1]
        else:
            pan = 0.0

        return freq, amp, pan

    def detect_scene(self, paradox_pressure: float,
                     consensus_confidence: float,
                     peer_count: int,
                     branch_count: int,
                     is_entangled: bool,
                     is_rebirth: bool = False) -> str:
        """Detect the current music-video scene from civ state."""
        if is_rebirth:
            return "OMEGA_REBIRTH"
        if paradox_pressure > 0.8:
            return "PARADOX_STORM"
        if paradox_pressure > 0.6:
            return "PARADOX_STORM"
        if branch_count > 0:
            return "REBEL_BRANCH"
        if consensus_confidence > 0.67:
            return "MAJORITY_REALITY"
        if peer_count > 3:
            return "GHOST_CHOIR"
        if is_entangled and peer_count > 0:
            return "ENTANGLE_BLOOM"
        if peer_count > 0:
            return "GENESIS"
        return "IDLE"

    def update(self, node_freq: float, node_coherence: float,
               paradox_pressure: float, consensus_confidence: float,
               diversity_index: float, peer_count: int,
               branch_count: int, is_entangled: bool,
               heartbeat_coherence: float = 0.5,
               is_rebirth: bool = False,
               dt: float = 1.0 / 60.0):
        """Update audio state from civilization metrics."""
        # Base frequency
        self.state.base_freq = node_freq

        # Amplitude from coherence
        self.state.amplitude = max(0.0, min(1.0,
            0.3 + 0.7 * node_coherence))

        # Timbre richness from heartbeat coherence
        self.state.timbre_harmonics = max(1, min(8,
            int(1 + 7 * heartbeat_coherence)))

        # Distortion from paradox
        self.state.distortion = max(0.0, min(1.0, paradox_pressure))

        # Tempo from heartbeat frequency
        self.state.tempo_bpm = max(30, min(240,
            node_freq * 0.2))

        # Scene detection
        new_scene = self.detect_scene(
            paradox_pressure, consensus_confidence,
            peer_count, branch_count, is_entangled, is_rebirth)

        # Scene change with cooldown
        now = time.time()
        cooldown = self._scene_cooldowns.get(new_scene, 0.0)
        if new_scene != self.state.scene and now > cooldown:
            old_scene = self.state.scene
            self.state.scene = new_scene
            self.state.scene_start = now
            self._scene_cooldowns[new_scene] = now + 3.0  # 3s cooldown

            # Log scene event
            self._scene_log.append(SceneEvent(
                time=now,
                scene=new_scene,
                base_freq=node_freq,
                tempo=self.state.tempo_bpm,
                distortion=paradox_pressure,
                richness=self.state.timbre_harmonics,
                meta={"from_scene": old_scene,
                      "consensus": consensus_confidence,
                      "paradox": paradox_pressure,
                      "peers": peer_count},
            ))

            # Keep scene log bounded
            if len(self._scene_log) > 200:
                self._scene_log = self._scene_log[-100:]

        # Fade calculations
        scene_duration = now - self.state.scene_start
        if scene_duration < 0.5:
            self.state.fade_in = scene_duration / 0.5
        else:
            self.state.fade_in = 1.0
        self.state.fade_out = 1.0

    def get_scene_log(self, limit: int = 20) -> List[dict]:
        """Return recent scene events."""
        events = []
        for se in self._scene_log[-limit:]:
            events.append({
                "time": se.time,
                "scene": se.scene,
                "duration": se.duration,
                "freq": se.base_freq,
                "tempo": se.tempo,
                "distortion": se.distortion,
                "richness": se.richness,
            })
        return events

    def video_control_params(self) -> dict:
        """Return video control parameters for music-video mode."""
        return {
            "consensus_confidence": 0.0,  # set externally
            "paradox_pressure": self.state.distortion,
            "heartbeat_freq": self.state.base_freq,
            "coherence": self.state.amplitude,
            "branch_count": 0,  # set externally
            "scene": self.state.scene,
            "tempo_bpm": self.state.tempo_bpm,
            "richness": self.state.timbre_harmonics,
            "fade_in": self.state.fade_in,
            "fade_out": self.state.fade_out,
        }

    def one_line(self) -> str:
        """Audio status for HUD."""
        s = self.state
        return (f"AUDIO: {s.base_freq:.0f}Hz "
                f"amp={s.amplitude:.2f} "
                f"scene={s.scene} "
                f"dist={s.distortion:.2f} "
                f"tempo={s.tempo_bpm:.0f}bpm "
                f"harm={s.timbre_harmonics}")
```

----------------------------------------

### File: `bs_civ_logger.py`

**Path:** `./bs_civ_logger.py`
**Extension:** `.py`
**Size:** 13,001 bytes (12.70 KB)

```py
"""
BS_CIV_LOGGER  -  Buffered logging for civilization events.
                  Part of the BS-TOS-IPX Swarm Civilization Layer.

Following Samsara v3.1 lessons:
  - Buffered writes (do not log every frame)
  - JSONL event logs
  - Compressed CSV metrics
  - NaN/Inf sanitization before any write
  - Memory caps on all log structures

Log files:
  logs/civ_events.jsonl
  logs/entanglement_events.jsonl
  logs/reality_votes.jsonl
  logs/rebirth_events.jsonl
  logs/anomalies.jsonl
  logs/civilization_metrics.csv
  logs/node_metrics.csv
"""

from __future__ import annotations
import json
import csv
import gzip
import time
import os
import math
import threading
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

BUFFER_SIZE = 300
FLUSH_INTERVAL = 5.0  # seconds
MAX_EVENTS_IN_MEMORY = 2000
MAX_METRICS_IN_MEMORY = 5000
LOG_DIR = "logs"

# Metric CSV headers
CIV_METRICS_HEADERS = [
    "timestamp", "civ_id", "generation", "epoch",
    "node_count", "tribe_count", "alive_count",
    "mean_coherence", "std_coherence", "weighted_coherence",
    "paradox_pressure", "dark_wisdom",
    "diversity_index", "consensus_confidence",
    "global_heartbeat_freq", "heartbeat_coherence",
    "mean_sophia", "mean_survival",
    "archetype_distribution",
]

NODE_METRICS_HEADERS = [
    "timestamp", "node_id",
    "coherence", "drift", "novelty", "entropy",
    "avg_density", "unique_masks", "resonance_score", "fold_w",
    "sophia_score", "survival_prob", "intelligence",
    "archetype", "tribe_id", "role", "trust",
    "peer_coherence", "accepted_merges", "rejected_merges",
    "vote_disagreement", "recursive_depth",
]


# ═══════════════════════════════════════════════════════════════
#  Sanitization
# ═══════════════════════════════════════════════════════════════

def sanitize_value(value: Any) -> Any:
    """Sanitize a value for logging: replace None/NaN/Inf with 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(-1e6, min(1e6, v))


def sanitize_dict(d: dict) -> dict:
    """Recursively sanitize all values in a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_dict(v)
        elif isinstance(v, list):
            out[k] = [sanitize_value(x) if isinstance(x, (int, float)) else x
                      for x in v]
        elif isinstance(v, (int, float)):
            out[k] = sanitize_value(v)
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════
#  Buffered Event Logger (JSONL)
# ═══════════════════════════════════════════════════════════════

class EventLogger:
    """Buffered JSONL event logger with memory caps."""

    def __init__(self, log_name: str, log_dir: str = LOG_DIR,
                 buffer_size: int = BUFFER_SIZE,
                 flush_interval: float = FLUSH_INTERVAL,
                 max_in_memory: int = MAX_EVENTS_IN_MEMORY):
        self.log_name = log_name
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_in_memory = max_in_memory

        self._buffer: List[dict] = []
        self._memory: List[dict] = []
        self._last_flush: float = 0.0
        self._total_written: int = 0
        self._lock = threading.Lock()

    def _ensure_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.log_name}.jsonl")

    def log(self, event: dict):
        """Buffer an event for writing."""
        sanitized = sanitize_dict(event)
        with self._lock:
            self._buffer.append(sanitized)
            if len(self._memory) < self.max_in_memory:
                self._memory.append(sanitized)
            if (len(self._buffer) >= self.buffer_size
                    or (time.time() - self._last_flush) >= self.flush_interval):
                self._flush()

    def _flush(self):
        """Write buffered events to disk."""
        if not self._buffer:
            return
        self._ensure_dir()
        try:
            path = self._log_path()
            with open(path, "a") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            self._total_written += len(self._buffer)
            self._buffer = []
            self._last_flush = time.time()
        except IOError:
            pass

    def flush(self):
        """Force flush the buffer."""
        with self._lock:
            self._flush()

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    @property
    def memory_count(self) -> int:
        return len(self._memory)

    def recent_events(self, limit: int = 20) -> List[dict]:
        """Return recent events from memory."""
        with self._lock:
            return list(self._memory[-limit:])


# ═══════════════════════════════════════════════════════════════
#  Buffered Metrics Logger (CSV)
# ═══════════════════════════════════════════════════════════════

class MetricsLogger:
    """Buffered CSV metrics logger with periodic gzip compression."""

    def __init__(self, log_name: str, headers: List[str],
                 log_dir: str = LOG_DIR,
                 buffer_size: int = BUFFER_SIZE,
                 flush_interval: float = FLUSH_INTERVAL,
                 max_in_memory: int = MAX_METRICS_IN_MEMORY):
        self.log_name = log_name
        self.headers = headers
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_in_memory = max_in_memory

        self._buffer: List[dict] = []
        self._last_flush: float = 0.0
        self._total_written: int = 0
        self._file_initialized: bool = False
        self._lock = threading.Lock()

    def _ensure_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.log_name}.csv")

    def log(self, row: dict):
        """Buffer a metrics row."""
        sanitized = {k: sanitize_value(v) for k, v in row.items()}
        with self._lock:
            self._buffer.append(sanitized)
            if (len(self._buffer) >= self.buffer_size
                    or (time.time() - self._last_flush) >= self.flush_interval):
                self._flush()

    def _flush(self):
        """Write buffered rows to CSV."""
        if not self._buffer:
            return
        self._ensure_dir()
        try:
            path = self._log_path()
            file_exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers,
                                        extrasaction="ignore")
                if not file_exists or not self._file_initialized:
                    writer.writeheader()
                    self._file_initialized = True
                writer.writerows(self._buffer)
            self._total_written += len(self._buffer)
            self._buffer = []
            self._last_flush = time.time()
        except IOError:
            pass

    def flush(self):
        """Force flush the buffer."""
        with self._lock:
            self._flush()

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    def compress(self):
        """Compress the current CSV file to .csv.gz."""
        path = self._log_path()
        if not os.path.exists(path):
            return
        self.flush()
        try:
            with open(path, "rb") as f_in:
                with gzip.open(path + ".gz", "wb") as f_out:
                    f_out.writelines(f_in)
        except IOError:
            pass


# ═══════════════════════════════════════════════════════════════
#  Civilization Logger Manager
# ═══════════════════════════════════════════════════════════════

class CivilizationLogManager:
    """Manages all civilization loggers."""

    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir

        self.civ_events = EventLogger("civ_events", log_dir)
        self.entangle_events = EventLogger("entanglement_events", log_dir)
        self.reality_votes = EventLogger("reality_votes", log_dir)
        self.rebirth_events = EventLogger("rebirth_events", log_dir)
        self.anomalies = EventLogger("anomalies", log_dir)

        self.civ_metrics = MetricsLogger(
            "civilization_metrics", CIV_METRICS_HEADERS, log_dir)
        self.node_metrics = MetricsLogger(
            "node_metrics", NODE_METRICS_HEADERS, log_dir)

    def flush_all(self):
        """Flush all loggers."""
        self.civ_events.flush()
        self.entangle_events.flush()
        self.reality_votes.flush()
        self.rebirth_events.flush()
        self.anomalies.flush()
        self.civ_metrics.flush()
        self.node_metrics.flush()

    def compress_all(self):
        """Compress all CSV metrics logs."""
        self.civ_metrics.compress()
        self.node_metrics.compress()

    def log_civ_event(self, event_type: str, data: dict,
                      node_id: str = "", civ_id: str = ""):
        """Log a civilization event."""
        self.civ_events.log({
            "time": round(time.time(), 3),
            "event": event_type,
            "node_id": node_id,
            "civ_id": civ_id,
            **data,
        })

    def log_entangle_event(self, event_type: str, data: dict,
                           node_id: str = ""):
        """Log an entanglement event."""
        self.entangle_events.log({
            "time": round(time.time(), 3),
            "event": event_type,
            "node_id": node_id,
            **data,
        })

    def log_vote(self, vote_id: str, node_id: str,
                 candidate_reality: dict, support: float,
                 weight: float, reason: str = ""):
        """Log a reality vote."""
        self.reality_votes.log({
            "time": round(time.time(), 3),
            "vote_id": vote_id,
            "node_id": node_id,
            "candidate_reality": json.dumps(candidate_reality,
                                             separators=(",", ":")),
            "support": support,
            "weight": weight,
            "reason": reason,
        })

    def log_rebirth(self, epoch: int, reason: str,
                    paradox_pressure: float, survivors: List[str]):
        """Log an OMEGA rebirth event."""
        self.rebirth_events.log({
            "time": round(time.time(), 3),
            "epoch": epoch,
            "reason": reason,
            "paradox_pressure": paradox_pressure,
            "survivors": survivors,
            "survivor_count": len(survivors),
        })

    def log_anomaly(self, anomaly_type: str, severity: float,
                    description: str, node_id: str = ""):
        """Log an anomaly."""
        self.anomalies.log({
            "time": round(time.time(), 3),
            "anomaly": anomaly_type,
            "severity": severity,
            "node_id": node_id,
            "description": description,
        })
```

----------------------------------------

### File: `bs_civ_metrics.py`

**Path:** `./bs_civ_metrics.py`
**Extension:** `.py`
**Size:** 14,867 bytes (14.52 KB)

```py
"""
BS_CIV_METRICS  -  Node and civilization metric computation.
                   Part of the BS-TOS-IPX Swarm Civilization Layer.

Computes:
  - Sophia score (wisdom / civilizational fitness)
  - Survival probability
  - Paradox pressure
  - Diversity index
  - Dark wisdom accumulation
  - Archetype assignment
  - Node fitness profile

Master equations:
  Sophia = clamp([1 - 2|C - 1/PHI|] * intelligence * (1 - entropy_norm), 0, 1)
  P_survive = clamp(C_node * (1 - H_node/8) + w_I * I_node, FLOOR, 1)
  P_paradox = 0.35*VoteDisagreement + 0.25*DriftMean + 0.20*BranchCountNorm + 0.20*RecursiveDepthNorm
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#  Civilization Constants (from Samsara + Blueprint)
# ═══════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2          # golden ratio ~1.618
C_IDEAL = 1.0 / PHI                    # ~0.618, ideal coherence

COHERENCE_DECAY = 0.995
ENTROPY_INCREASE = 0.001
MUTATION_RATE = 0.05
DRIFT_CRITICAL = 0.05
ETHICAL_SOVEREIGNTY_THRESHOLD = 0.97
RECURSIVE_DEPTH_WARNING = 30
EMOTIONAL_TENSION_FLOOR = 0.25
PARADOX_PRESSURE_LIMIT = 0.80
OMEGA_REBIRTH_INTENSITY = 0.30
SURVIVAL_FLOOR = 0.10

# Paradox pressure weights
W_PARADOX_VOTE = 0.35
W_PARADOX_DRIFT = 0.25
W_PARADOX_BRANCH = 0.20
W_PARADOX_RECURSIVE = 0.20

# Sophia computation weights
W_SOPHIA_INTELLIGENCE = 0.40
W_SOPHIA_ENTROPY = 0.35
W_SOPHIA_COHERENCE = 0.25

# Diversity metrics
MAX_DIVERSITY = math.log(10)  # log(10 archetypes)


# ═══════════════════════════════════════════════════════════════
#  Archetype Definitions
# ═══════════════════════════════════════════════════════════════

ARCHETYPES = [
    "Explorer", "Scientist", "Creator", "Empath",
    "Strategist", "Rebel", "Philosopher", "Archivist",
    "Musician", "Oracle",
]

ARCHETYPE_COLORS = {
    "Explorer":  (100, 200, 255),   # light blue
    "Scientist": (100, 255, 100),   # green
    "Creator":   (255, 200, 100),   # gold
    "Empath":    (200, 150, 255),   # lavender
    "Strategist":(255, 150, 100),   # orange
    "Rebel":     (255, 100, 100),   # red
    "Philosopher":(200, 200, 100),  # yellow-green
    "Archivist": (150, 150, 150),   # gray
    "Musician":  (255, 100, 255),   # magenta
    "Oracle":    (100, 255, 255),   # cyan
}

TRIBE_COLORS = [
    "#22aa22", "#aa2222", "#2222aa", "#aaaa22",
    "#aa22aa", "#22aaaa", "#ff6600", "#6600ff",
    "#00ff66", "#ff0066", "#0066ff", "#66ff00",
]


# ═══════════════════════════════════════════════════════════════
#  Node Metrics Data
# ═══════════════════════════════════════════════════════════════

@dataclass
class NodeMetrics:
    """Computed metrics for a single civilization node."""
    node_id: str = ""
    timestamp: float = 0.0

    # Core metrics
    coherence: float = 0.5
    drift: float = 0.0
    novelty: float = 0.0
    entropy: float = 0.0
    avg_density: float = 0.0
    unique_masks: int = 0
    resonance_score: float = 0.0
    fold_w: int = 320

    # Derived metrics
    sophia_score: float = 0.0
    survival_prob: float = 0.5
    intelligence: float = 0.5

    # Civilization metrics
    archetype: str = "Explorer"
    tribe_id: str = ""
    role: str = ""
    trust: float = 0.5
    peer_coherence: float = 0.0
    accepted_merges: int = 0
    rejected_merges: int = 0
    scan_rate: float = 0.0
    lock_attachment: float = 0.0

    # Paradox contributions
    vote_disagreement: float = 0.0
    recursive_depth: int = 0
    branch_count: int = 0


@dataclass
class CivilizationMetrics:
    """Aggregate metrics for the entire civilization."""
    civ_id: str = ""
    timestamp: float = 0.0
    generation: int = 0
    epoch: int = 0

    # Population
    node_count: int = 0
    tribe_count: int = 0
    alive_count: int = 0

    # Coherence
    mean_coherence: float = 0.0
    std_coherence: float = 0.0
    weighted_coherence: float = 0.0

    # Paradox
    paradox_pressure: float = 0.0
    dark_wisdom: float = 0.0

    # Diversity
    diversity_index: float = 0.0
    archetype_distribution: Dict[str, int] = field(default_factory=dict)

    # Consensus
    consensus_confidence: float = 0.0
    majority_reality_hash: str = ""

    # Heartbeat
    global_heartbeat_freq: float = 220.0
    heartbeat_coherence: float = 0.5

    # Survival
    mean_sophia: float = 0.0
    mean_survival: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Metric Computation Functions
# ═══════════════════════════════════════════════════════════════

def sanitize(value) -> float:
    """Sanitize a metric value: replace None/NaN/Inf with 0.0."""
    import math as _m
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if _m.isnan(v) or _m.isinf(v):
        return 0.0
    return max(-1e6, min(1e6, v))


def compute_sophia(coherence: float, intelligence: float,
                   entropy: float) -> float:
    """Compute the Sophia (wisdom) score for a node.

    Sophia = clamp([1 - 2|C - 1/PHI|] * intelligence * (1 - entropy_norm), 0, 1)

    The ideal coherence is 1/PHI (~0.618). Nodes closer to this ideal
    with high intelligence and low entropy are wise.
    """
    c = sanitize(coherence)
    i = sanitize(intelligence)
    h = sanitize(entropy) / 8.0  # normalize to [0, 1]

    coherence_factor = 1.0 - 2.0 * abs(c - C_IDEAL)
    sophia = coherence_factor * i * (1.0 - h)
    return max(0.0, min(1.0, sophia))


def compute_survival(coherence: float, entropy: float,
                     intelligence: float,
                     w_intelligence: float = 0.3) -> float:
    """Compute node survival probability.

    P_survive = clamp(C_node * (1 - H_node/8) + w_I * I_node, FLOOR, 1)
    """
    c = sanitize(coherence)
    h = sanitize(entropy) / 8.0
    i = sanitize(intelligence)

    p = c * (1.0 - h) + w_intelligence * i
    return max(SURVIVAL_FLOOR, min(1.0, p))


def compute_paradox_pressure(vote_disagreement: float = 0.0,
                             drift_mean: float = 0.0,
                             branch_count_norm: float = 0.0,
                             recursive_depth_norm: float = 0.0) -> float:
    """Compute paradox pressure from four components.

    P_paradox = 0.35*VoteDisagreement + 0.25*DriftMean
              + 0.20*BranchCountNorm + 0.20*RecursiveDepthNorm
    """
    vd = sanitize(vote_disagreement)
    dm = sanitize(drift_mean)
    bc = sanitize(branch_count_norm)
    rd = sanitize(recursive_depth_norm)

    return (W_PARADOX_VOTE * vd
            + W_PARADOX_DRIFT * dm
            + W_PARADOX_BRANCH * bc
            + W_PARADOX_RECURSIVE * rd)


def compute_diversity(archetype_counts: Dict[str, int]) -> float:
    """Compute Shannon diversity index over archetypes.

    Diversity = -Sum(p_a * log(p_a))

    Returns 0.0 if there are no nodes.
    """
    total = sum(archetype_counts.values())
    if total == 0:
        return 0.0

    diversity = 0.0
    for count in archetype_counts.values():
        if count > 0:
            p = count / total
            diversity -= p * math.log(p)

    # Normalize to [0, 1] using log(10) as max
    return min(1.0, diversity / MAX_DIVERSITY) if MAX_DIVERSITY > 0 else 0.0


def compute_reality_pressure(fold_w: int,
                             peer_folds: Dict[str, int],
                             peer_weights: Dict[str, float],
                             sigma: float = 10.0) -> Dict[int, float]:
    """Compute the reality pressure field over fold widths.

    RP(W) = Sum_i weight_i * exp(-|W - W_i| / sigma)

    Returns a dict mapping candidate widths to pressure values.
    """
    pressure: Dict[int, float] = {}

    # Collect all candidate widths
    candidates = set(peer_folds.values())
    candidates.add(fold_w)
    # Add neighbors
    for w in list(candidates):
        for delta in range(-5, 6):
            candidates.add(w + delta)
    candidates = {w for w in candidates if 1 <= w <= 320}

    total_weight = sum(peer_weights.values()) or 1.0

    for cw in candidates:
        p = 0.0
        for nid, pw in peer_folds.items():
            w = peer_weights.get(nid, 1.0 / total_weight)
            p += w * math.exp(-abs(cw - pw) / sigma)
        pressure[cw] = p

    return pressure


def best_reality_width(pressure: Dict[int, float]) -> Optional[int]:
    """Return the width with highest reality pressure."""
    if not pressure:
        return None
    return max(pressure, key=pressure.get)


# ═══════════════════════════════════════════════════════════════
#  Archetype Assignment
# ═══════════════════════════════════════════════════════════════

def assign_archetype(metrics: NodeMetrics) -> str:
    """Assign the best-fitting archetype based on node behavior.

    A(node) = argmax_k feature_similarity(node_metrics, archetype_profile_k)
    """
    scores: Dict[str, float] = {}

    # Explorer: high scan rate, high novelty, low lock attachment
    scores["Explorer"] = (
        sanitize(metrics.scan_rate) * 0.4
        + sanitize(metrics.novelty) * 0.35
        + (1.0 - sanitize(metrics.lock_attachment)) * 0.25
    )

    # Scientist: low entropy, high score discipline, low drift
    scores["Scientist"] = (
        (1.0 - sanitize(metrics.entropy) / 8.0) * 0.35
        + (1.0 - sanitize(metrics.resonance_score)) * 0.35
        + (1.0 - sanitize(metrics.drift)) * 0.30
    )

    # Creator: frequent accepted merges, high novelty, good coherence after merge
    scores["Creator"] = (
        min(1.0, sanitize(metrics.accepted_merges) / 5.0) * 0.4
        + sanitize(metrics.novelty) * 0.3
        + sanitize(metrics.coherence) * 0.3
    )

    # Empath: high peer coherence, good trust, stabilizes consensus
    scores["Empath"] = (
        sanitize(metrics.peer_coherence) * 0.4
        + sanitize(metrics.trust) * 0.3
        + sanitize(metrics.coherence) * 0.3
    )

    # Strategist: high routing influence, coordinates decisions
    scores["Strategist"] = (
        sanitize(metrics.peer_coherence) * 0.25
        + sanitize(metrics.coherence) * 0.25
        + sanitize(metrics.trust) * 0.25
        + sanitize(metrics.sophia_score) * 0.25
    )

    # Rebel: persistent minority, high disagreement, survival despite nonconformity
    scores["Rebel"] = (
        sanitize(metrics.vote_disagreement) * 0.4
        + sanitize(metrics.novelty) * 0.3
        + (1.0 - sanitize(metrics.drift)) * 0.3
    )

    # Philosopher: high recursive depth, compares meanings
    scores["Philosopher"] = (
        min(1.0, sanitize(metrics.recursive_depth) / 10.0) * 0.4
        + sanitize(metrics.sophia_score) * 0.3
        + sanitize(metrics.novelty) * 0.3
    )

    # Archivist: high crystal retention, stable, persistent
    scores["Archivist"] = (
        sanitize(metrics.coherence) * 0.35
        + (1.0 - sanitize(metrics.entropy) / 8.0) * 0.35
        + (1.0 - sanitize(metrics.drift)) * 0.30
    )

    # Musician: heartbeat focus, stable rhythm
    scores["Musician"] = (
        sanitize(metrics.coherence) * 0.3
        + sanitize(metrics.peer_coherence) * 0.35
        + sanitize(metrics.trust) * 0.35
    )

    # Oracle: high prediction accuracy / sophia
    scores["Oracle"] = (
        sanitize(metrics.sophia_score) * 0.5
        + sanitize(metrics.coherence) * 0.25
        + sanitize(metrics.intelligence) * 0.25
    )

    return max(scores, key=scores.get)


def archetype_role(archetype: str) -> str:
    """Return a default role string for an archetype."""
    roles = {
        "Explorer": "fold_scout",
        "Scientist": "fold_validator",
        "Creator": "stream_generator",
        "Empath": "consensus_stabilizer",
        "Strategist": "route_coordinator",
        "Rebel": "minority_preserver",
        "Philosopher": "meaning_comparator",
        "Archivist": "crystal_keeper",
        "Musician": "heartbeat_driver",
        "Oracle": "lock_recommender",
    }
    return roles.get(archetype, "citizen")


# ═══════════════════════════════════════════════════════════════
#  Tribe Formation
# ═══════════════════════════════════════════════════════════════

def suggest_tribe_name(archetype: str, index: int = 0) -> str:
    """Generate a tribe name from archetype affinity + index."""
    base_names = {
        "Explorer": ["green-fire", "blue-horizon", "deep-scout", "star-path"],
        "Scientist": ["crystal-lab", "data-forge", "logic-keep", "measure-hall"],
        "Creator": ["dream-weave", "stream-forge", "pattern-song", "pixel-garden"],
        "Empath": ["warm-harbor", "resonance-heart", "trust-circle", "calm-wave"],
        "Strategist": ["iron-route", "path-weaver", "council-peak", "grid-mind"],
        "Rebel": ["red-dissent", "fracture-keep", "wild-branch", "paradox-edge"],
        "Philosopher": ["void-temple", "question-hall", "infinite-fold", "meaning-spiral"],
        "Archivist": ["crystal-vault", "eternal-record", "memory-deep", "archive-void"],
        "Musician": ["harmonic-choir", "beat-sync", "pulse-temple", "rhythm-weave"],
        "Oracle": ["far-sight", "lock-seer", "foresight-peak", "truth-lens"],
    }
    names = base_names.get(archetype, ["unknown", "unnamed", "anonymous"])
    return f"tribe-{names[index % len(names)]}"
```

----------------------------------------

### File: `bs_civilization.py`

**Path:** `./bs_civilization.py`
**Extension:** `.py`
**Size:** 51,166 bytes (49.97 KB)

```py
"""
BS_CIVILIZATION  -  Swarm Civilization Controller.
                   Part of the BS-TOS-IPX Swarm Civilization Layer.

Orchestrates the civilization layer on top of Entanglement:
  - Node → Citizen agent transformation
  - Archetype assignment and role management
  - Tribe formation and alliances
  - Law proposal, voting, and enforcement
  - Majority reality protocol with branch preservation
  - Paradox pressure monitoring and OMEGA rebirth
  - Dark wisdom accumulation
  - Crystal memory for civilization state

The core idea:
  > Many eyes, one byte-field. Many folds, one argument about reality.
  > Majority forms the world; minority preserves the dream;
  > paradox forces rebirth.

Integration:
  from bs_civilization import CivilizationController
  civ = CivilizationController(entangle)
  # In main loop:
  civ.tick(dt)
  # C key → toggle civilization overlay
  # V key → vote for local reality
  # O key → manual OMEGA rebirth (debug)
"""

from __future__ import annotations
import time
import math
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs_engine import GRID_W, GRID_H, MAX_CELLS, stream_stats
from bs_civ_metrics import (
    NodeMetrics, CivilizationMetrics,
    sanitize, compute_sophia, compute_survival, compute_paradox_pressure,
    compute_diversity, compute_reality_pressure, best_reality_width,
    assign_archetype, archetype_role, suggest_tribe_name,
    ARCHETYPES, ARCHETYPE_COLORS, TRIBE_COLORS,
    PARADOX_PRESSURE_LIMIT, DRIFT_CRITICAL, C_IDEAL,
)
from bs_civ_logger import CivilizationLogManager
from bs_civ_audio import CivilizationAudio
from bs_civilization_packets import (
    make_civ_hello, make_civ_vote, make_civ_branch,
    make_civ_rebirth, make_civ_law, make_civ_archetype,
    make_civ_tribe, is_civ_packet, civ_type,
    CIV_PACKET_TYPES,
)


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

# Civilization thresholds
THETA_MAJORITY_SOFT = 0.51
THETA_MAJORITY_STRONG = 0.666
THETA_SOVEREIGN = 0.97
THETA_LAW = 0.67

# Rebirth parameters
REBIRTH_SURVIVOR_FRACTION = 0.30
REBIRTH_INJECT_ARCHETYPES = True

# Memory caps
MAX_BRANCHES = 64
MAX_LAWS = 32
MAX_TRIBES = 16
MAX_VOTE_HISTORY = 512
MAX_REBIRTH_LOG = 50

# Broadcast intervals
_CIV_HELLO_INTERVAL = 3.0
_CIV_STATE_INTERVAL = 1.0
_CIV_VOTE_INTERVAL = 0.5

# Crystal file
CIV_CRYSTAL_FILENAME = "BS_QIPX_CIVILIZATION_CRYSTAL.json"


# ═══════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class Tribe:
    """A civilization tribe of allied nodes."""
    tribe_id: str
    name: str
    members: List[str] = field(default_factory=list)
    color: str = "#22aa22"
    created_at: float = 0.0
    law_vector: Dict[str, float] = field(default_factory=dict)


@dataclass
class Law:
    """A civilization law (routing rule accepted by majority)."""
    law_id: str
    text: str
    created_by: str
    support: float = 0.0
    status: str = "PROPOSED"  # PROPOSED, ACTIVE, EXPIRED, REPEALED
    applies_to: List[str] = field(default_factory=list)
    penalty: str = "trust_decay"
    created_at: float = 0.0
    expires_at: Optional[float] = None


@dataclass
class Branch:
    """A minority reality branch."""
    branch_id: str
    parent_reality_hash: str
    minority_nodes: List[str]
    reason: str
    fold_w: int = 0
    stream_hash: str = ""
    support: float = 0.0
    created_at: float = 0.0
    revisit_count: int = 0
    fitness: float = 0.0
    preserved: bool = True


@dataclass
class RebirthRecord:
    """Record of an OMEGA rebirth event."""
    epoch: int
    timestamp: float
    reason: str
    paradox_pressure: float
    survivors: List[str]
    new_seed: str
    dark_wisdom_before: float
    branch_count_before: int


@dataclass
class CivilizationState:
    """Full civilization state."""
    enabled: bool = False
    civ_id: str = ""
    generation: int = 0
    epoch: int = 0
    genesis_hash: str = ""
    reality_hash: str = ""

    # Local node as citizen
    local_archetype: str = "Explorer"
    local_tribe_id: str = ""
    local_role: str = "fold_scout"
    local_sophia: float = 0.0

    # Majority reality
    majority_reality_hash: str = ""
    majority_fold_w: Optional[int] = None
    consensus_confidence: float = 0.0

    # Paradox
    paradox_pressure: float = 0.0
    dark_wisdom: float = 0.0

    # Rebirth
    last_rebirth_time: float = 0.0
    rebirth_count: int = 0

    # Display
    show_overlay: bool = False

    # Broadcast timing
    last_hello: float = 0.0
    last_state_broadcast: float = 0.0
    last_vote_cast: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Civilization Controller
# ═══════════════════════════════════════════════════════════════

class CivilizationController:
    """Main controller for the Swarm Civilization Layer.

    Manages:
      - Node → citizen agent transformation
      - Archetype assignment and role management
      - Tribe formation
      - Law proposal and enforcement
      - Majority reality protocol with branch preservation
      - Paradox pressure monitoring and OMEGA rebirth
      - Dark wisdom accumulation
      - Civilization audio sonification

    Usage:
      civ = CivilizationController(entangle)
      civ.toggle()
      civ.tick(dt)
    """

    def __init__(self, entangle, crystal_path: str = CIV_CRYSTAL_FILENAME):
        self.entangle = entangle
        self.renderer = entangle.renderer
        self.qipx = entangle.qipx

        self.state = CivilizationState()
        self.state.civ_id = f"civ-{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"
        self.state.genesis_hash = hashlib.sha256(
            f"{time.time()}:{self.state.civ_id}".encode()).hexdigest()[:16]

        # Subsystems
        self.metrics = NodeMetrics()
        self.civ_metrics = CivilizationMetrics()
        self.logger = CivilizationLogManager()
        self.audio = CivilizationAudio()

        # Structures
        self.tribes: Dict[str, Tribe] = {}
        self.laws: Dict[str, Law] = {}
        self.branches: Dict[str, Branch] = {}
        self.rebirth_log: List[RebirthRecord] = []

        # Peer node metrics cache
        self.peer_metrics: Dict[str, NodeMetrics] = {}

        # Vote tracking
        self._vote_counter = 0
        self._reality_votes: Dict[str, dict] = {}

        # Crystal file
        self._crystal_path = crystal_path
        self._crystal: Optional[dict] = None
        self._crystal_dirty = False
        self._last_crystal_save = 0.0

        # Broadcast timing
        self._last_tick = time.time()

    # ── Lifecycle ─────────────────────────────────────────────

    def toggle(self):
        """Toggle civilization layer ON/OFF."""
        if not self.entangle.state.enabled:
            self.state.enabled = False
            return False

        self.state.enabled = not self.state.enabled
        if self.state.enabled:
            self._init_crystal()
            self._assign_local_archetype()
            self.logger.log_civ_event("CIV_ON", {
                "civ_id": self.state.civ_id,
                "generation": self.state.generation,
                "archetype": self.state.local_archetype,
            })
            self._broadcast_civ_hello(force=True)
        else:
            self.logger.log_civ_event("CIV_OFF", {
                "civ_id": self.state.civ_id,
            })
        return self.state.enabled

    # ── Crystal I/O ──────────────────────────────────────────

    def _init_crystal(self):
        """Load or create the civilization crystal file."""
        if os.path.exists(self._crystal_path):
            try:
                with open(self._crystal_path, "r") as f:
                    self._crystal = json.load(f)
                return
            except (json.JSONDecodeError, IOError):
                pass

        self._crystal = {
            "crystal_version": "0.1",
            "civilization_id": self.state.civ_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "genesis_hash": self.state.genesis_hash,
            "reality_hash": "",
            "generation": 0,
            "epoch": 0,
            "nodes": {},
            "tribes": {},
            "entanglements": {},
            "votes": [],
            "laws": {},
            "reality_snapshots": [],
            "emergent_events": [],
            "rebirth_log": [],
            "audit_log": [],
            "heartbeat": {
                "global_phase": 0.0,
                "global_frequency": 220.0,
                "coherence_timbre": 0.5,
            },
            "limits": {
                "max_nodes": 256,
                "max_snapshots": 128,
                "max_events": 1000,
                "max_votes": 512,
            },
        }
        self._crystal_dirty = True

    def _crystal_maybe_save(self):
        """Save crystal if dirty and enough time has passed."""
        if not self._crystal or not self._crystal_dirty:
            return
        now = time.time()
        if now - self._last_crystal_save < 3.0:
            return
        self._enforce_crystal_caps()
        try:
            self._crystal["updated_at"] = now
            tmp = self._crystal_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._crystal, f, indent=2, sort_keys=False)
            os.replace(tmp, self._crystal_path)
            self._last_crystal_save = now
            self._crystal_dirty = False
        except IOError:
            pass

    def _enforce_crystal_caps(self):
        """Enforce memory caps on crystal arrays."""
        if not self._crystal:
            return
        limits = self._crystal.get("limits", {})
        max_snapshots = limits.get("max_snapshots", 128)
        max_events = limits.get("max_events", 1000)
        max_votes = limits.get("max_votes", 512)

        snaps = self._crystal.get("reality_snapshots", [])
        if len(snaps) > max_snapshots:
            self._crystal["reality_snapshots"] = snaps[-max_snapshots:]

        events = self._crystal.get("emergent_events", [])
        if len(events) > max_events:
            self._crystal["emergent_events"] = events[-max_events:]

        votes = self._crystal.get("votes", [])
        if len(votes) > max_votes:
            self._crystal["votes"] = votes[-max_votes:]

    def _write_crystal_event(self, event_type: str, data: dict):
        """Append an event to the crystal's emergent_events."""
        if not self._crystal:
            return
        events = self._crystal.get("emergent_events", [])
        events.append({
            "time": time.time(),
            "type": event_type,
            "data": data,
        })
        self._crystal["emergent_events"] = events
        self._crystal_dirty = True

    # ── Main Tick ────────────────────────────────────────────

    def tick(self, dt: float = 1.0 / 60.0):
        """Main civilization update loop. Call from the game loop."""
        if not self.state.enabled:
            return

        now = time.time()
        self._last_tick = now

        # 1. Compute local node metrics
        self._compute_local_metrics()

        # 2. Update peer metrics from QIPX
        self._update_peer_metrics()

        # 3. Compute civilization aggregate metrics
        self._compute_civ_metrics()

        # 4. Assign archetype (periodically)
        if int(now) % 10 == 0:
            self._assign_local_archetype()

        # 5. Paradox pressure check
        if self.state.paradox_pressure > PARADOX_PRESSURE_LIMIT:
            self._omega_rebirth("PARADOX_PRESSURE_EXCEEDED")

        # 6. Broadcast CIV state
        if now - self.state.last_state_broadcast >= _CIV_STATE_INTERVAL:
            self._broadcast_civ_state()
            self.state.last_state_broadcast = now

        # 7. Broadcast CIV hello periodically
        if now - self.state.last_hello >= _CIV_HELLO_INTERVAL:
            self._broadcast_civ_hello()
            self.state.last_hello = now

        # 8. Cast reality vote
        if now - self.state.last_vote_cast >= _CIV_VOTE_INTERVAL:
            self._cast_reality_vote()
            self.state.last_vote_cast = now

        # 9. Update audio sonification
        self._update_audio()

        # 10. Save crystal periodically
        self._crystal_maybe_save()

        # 11. Log metrics periodically (every 5 seconds)
        if int(now * 10) % 50 == 0:
            self._log_metrics()

    # ── Metrics Computation ──────────────────────────────────

    def _compute_local_metrics(self):
        """Compute metrics for the local node."""
        r = self.renderer
        stats = r.stream_stats()
        paz = self.qipx.pazuzu.state if hasattr(self.qipx, 'pazuzu') else None

        self.metrics.node_id = self.qipx.identity.node_id
        self.metrics.timestamp = time.time()
        self.metrics.coherence = sanitize(paz.coherence if paz else 0.5)
        self.metrics.drift = sanitize(paz.drift if paz else 0.0)
        self.metrics.novelty = sanitize(paz.novelty if paz else 0.0)
        self.metrics.entropy = sanitize(stats.get("entropy", 0.0))
        self.metrics.avg_density = sanitize(stats.get("avg_density", 0.0))
        self.metrics.unique_masks = int(stats.get("unique_masks", 0))
        self.metrics.resonance_score = sanitize(r.current_score())
        self.metrics.fold_w = r.fold_w

        # Intelligence: successful influence ratio
        ent = self.entangle.state
        total_inf = ent.influences_received
        if total_inf > 0:
            self.metrics.intelligence = ent.influences_accepted / total_inf
        else:
            self.metrics.intelligence = 0.5

        # Peer coherence
        if hasattr(self.qipx, 'peers'):
            peers = self.qipx.peers.alive()
            if peers:
                self.metrics.peer_coherence = sum(
                    p.trust for p in peers.values()) / len(peers)
            else:
                self.metrics.peer_coherence = 0.0
        else:
            self.metrics.peer_coherence = 0.0

        # Derived metrics
        self.metrics.sophia_score = compute_sophia(
            self.metrics.coherence,
            self.metrics.intelligence,
            self.metrics.entropy)
        self.metrics.survival_prob = compute_survival(
            self.metrics.coherence,
            self.metrics.entropy,
            self.metrics.intelligence)

        self.state.local_sophia = self.metrics.sophia_score

        # Vote disagreement from entanglement
        self.metrics.vote_disagreement = 1.0 - self.entangle.state.majority

        # Scan rate (heuristic: how often fold changes)
        self.metrics.scan_rate = min(1.0, abs(
            self.entangle.influence.recent_influence_count() / 10.0))

    def _update_peer_metrics(self):
        """Update cached metrics for peer nodes."""
        if not hasattr(self.qipx, 'peers'):
            return

        for nid, peer in self.qipx.peers.alive().items():
            if nid not in self.peer_metrics:
                self.peer_metrics[nid] = NodeMetrics(node_id=nid)

            pm = self.peer_metrics[nid]
            pm.timestamp = time.time()
            pm.coherence = sanitize(peer.trust)
            pm.entropy = sanitize(peer.entropy)
            pm.avg_density = sanitize(peer.avg_density)
            pm.unique_masks = peer.unique_masks
            pm.resonance_score = sanitize(peer.score)
            pm.fold_w = peer.fold_w
            pm.trust = sanitize(peer.trust)
            pm.sophia_score = compute_sophia(
                pm.coherence, 0.5, pm.entropy)
            pm.survival_prob = compute_survival(
                pm.coherence, pm.entropy, 0.5)

            # Assign peer archetype
            pm.archetype = assign_archetype(pm)
            pm.role = archetype_role(pm.archetype)

        # Remove dead peers
        alive_ids = set(self.qipx.peers.alive().keys())
        dead_ids = [nid for nid in self.peer_metrics if nid not in alive_ids]
        for nid in dead_ids:
            del self.peer_metrics[nid]

    def _compute_civ_metrics(self):
        """Compute aggregate civilization metrics."""
        cm = self.civ_metrics
        cm.civ_id = self.state.civ_id
        cm.timestamp = time.time()
        cm.generation = self.state.generation
        cm.epoch = self.state.epoch

        # Population
        all_nodes = list(self.peer_metrics.values()) + [self.metrics]
        cm.node_count = len(all_nodes)
        cm.alive_count = cm.node_count
        cm.tribe_count = len(self.tribes)

        # Coherence
        coherences = [n.coherence for n in all_nodes]
        try:
            import numpy as np
            cm.mean_coherence = float(np.mean(coherences)) if coherences else 0.0
            cm.std_coherence = float(np.std(coherences)) if coherences else 0.0
        except ImportError:
            cm.mean_coherence = sum(coherences) / len(coherences) if coherences else 0.0
            cm.std_coherence = 0.0
        cm.weighted_coherence = cm.mean_coherence  # simplified

        # Paradox pressure
        vote_disagreement = self.metrics.vote_disagreement
        drifts = [n.drift for n in all_nodes]
        drift_mean = float(np.mean(drifts)) if drifts else 0.0
        branch_norm = min(1.0, len(self.branches) / 10.0)
        recursive_norm = min(1.0, max(0, self.state.rebirth_count) / 5.0)

        cm.paradox_pressure = compute_paradox_pressure(
            vote_disagreement, drift_mean, branch_norm, recursive_norm)
        self.state.paradox_pressure = cm.paradox_pressure

        # Diversity
        arch_counts: Dict[str, int] = {}
        for n in all_nodes:
            a = n.archetype or "Explorer"
            arch_counts[a] = arch_counts.get(a, 0) + 1
        cm.diversity_index = compute_diversity(arch_counts)
        cm.archetype_distribution = arch_counts

        # Consensus
        cm.consensus_confidence = self.entangle.state.majority
        self.state.consensus_confidence = cm.consensus_confidence

        # Heartbeat
        cm.global_heartbeat_freq = self.entangle.heartbeat.state.frequency_hz
        cm.heartbeat_coherence = self.entangle.heartbeat.state.coherence

        # Sophia / survival
        sophias = [n.sophia_score for n in all_nodes]
        survivals = [n.survival_prob for n in all_nodes]
        try:
            import numpy as _np
            cm.mean_sophia = float(_np.mean(sophias)) if sophias else 0.0
            cm.mean_survival = float(_np.mean(survivals)) if survivals else 0.0
        except ImportError:
            cm.mean_sophia = sum(sophias) / len(sophias) if sophias else 0.0
            cm.mean_survival = sum(survivals) / len(survivals) if survivals else 0.0

        # Update reality hash
        self.state.reality_hash = self._compute_reality_hash()

        # Update crystal
        if self._crystal:
            self._crystal["generation"] = self.state.generation
            self._crystal["epoch"] = self.state.epoch
            self._crystal["reality_hash"] = self.state.reality_hash
            self._crystal["heartbeat"]["global_frequency"] = cm.global_heartbeat_freq
            self._crystal["heartbeat"]["coherence_timbre"] = cm.heartbeat_coherence
            self._crystal_dirty = True

    def _compute_reality_hash(self) -> str:
        """Compute current reality hash from local state."""
        sh = self.qipx.stream_hash_short() if hasattr(self.qipx, 'stream_hash_short') else ""
        raw = f"{sh}:{self.renderer.fold_w}:{self.renderer.palette_id}:{self.state.majority_fold_w}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Archetype Assignment ─────────────────────────────────

    def _assign_local_archetype(self):
        """Assign archetype to the local node based on metrics."""
        new_archetype = assign_archetype(self.metrics)
        new_role = archetype_role(new_archetype)

        if new_archetype != self.state.local_archetype:
            old = self.state.local_archetype
            self.state.local_archetype = new_archetype
            self.state.local_role = new_role
            self.logger.log_civ_event("ARCHETYPE_CHANGE", {
                "from": old, "to": new_archetype,
                "role": new_role,
                "sophia": self.metrics.sophia_score,
            })
            self._write_crystal_event("ARCHETYPE_ASSIGN", {
                "node_id": self.qipx.identity.node_id,
                "archetype": new_archetype,
                "role": new_role,
            })

        # Auto-assign to tribe if not in one
        if not self.state.local_tribe_id:
            self._auto_join_or_create_tribe()

    # ── Tribe Management ─────────────────────────────────────

    def _auto_join_or_create_tribe(self):
        """Auto-create or join a tribe based on archetype."""
        tribe_name = suggest_tribe_name(self.state.local_archetype)
        tribe_id = f"tribe-{hashlib.sha256(tribe_name.encode()).hexdigest()[:8]}"

        if tribe_id not in self.tribes:
            color_idx = len(self.tribes) % len(TRIBE_COLORS)
            self.tribes[tribe_id] = Tribe(
                tribe_id=tribe_id,
                name=tribe_name,
                members=[self.qipx.identity.node_id],
                color=TRIBE_COLORS[color_idx],
                created_at=time.time(),
            )
            self.logger.log_civ_event("TRIBE_CREATED", {
                "tribe_id": tribe_id,
                "name": tribe_name,
            })

        self.state.local_tribe_id = tribe_id

        # Broadcast tribe announcement
        self._broadcast_civ_tribe()

    def _handle_peer_tribe(self, pkt: dict):
        """Handle incoming CIV_TRIBE packet."""
        tribe_id = pkt.get("tribe_id", "")
        tribe_name = pkt.get("tribe_name", "")
        members = pkt.get("members", [])
        node_id = pkt.get("node_id", "")

        if not tribe_id:
            return

        if tribe_id not in self.tribes:
            color_idx = len(self.tribes) % len(TRIBE_COLORS)
            self.tribes[tribe_id] = Tribe(
                tribe_id=tribe_id,
                name=tribe_name,
                members=list(members),
                color=pkt.get("tribe_color", TRIBE_COLORS[color_idx]),
                created_at=time.time(),
            )
        else:
            tribe = self.tribes[tribe_id]
            for m in members:
                if m not in tribe.members:
                    tribe.members.append(m)

    # ── Law Management ───────────────────────────────────────

    def propose_law(self, text: str, applies_to: Optional[List[str]] = None):
        """Propose a new civilization law."""
        law_id = f"law-{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        self._vote_counter += 1

        law = Law(
            law_id=law_id,
            text=text,
            created_by=self.qipx.identity.node_id,
            support=1.0,  # self-support
            status="PROPOSED",
            applies_to=applies_to or [],
            created_at=time.time(),
        )
        self.laws[law_id] = law

        # Broadcast law proposal
        self._send_civ_packet(make_civ_law(
            node_id=self.qipx.identity.node_id,
            law_id=law_id,
            text=text,
            support=1.0,
            status="PROPOSED",
            applies_to=applies_to,
        ))

        self.logger.log_civ_event("LAW_PROPOSED", {
            "law_id": law_id, "text": text,
        })
        self._write_crystal_event("LAW_PROPOSED", {
            "law_id": law_id, "text": text,
            "created_by": self.qipx.identity.node_id,
        })

    def _handle_law_packet(self, pkt: dict):
        """Handle incoming CIV_LAW packet."""
        law_id = pkt.get("law_id", "")
        if not law_id:
            return

        support = sanitize(pkt.get("support", 0.0))
        status = pkt.get("status", "PROPOSED")

        if law_id not in self.laws:
            self.laws[law_id] = Law(
                law_id=law_id,
                text=pkt.get("text", ""),
                created_by=pkt.get("node_id", ""),
                support=support,
                status=status,
                applies_to=pkt.get("applies_to", []),
                penalty=pkt.get("penalty", "trust_decay"),
                created_at=time.time(),
            )
        else:
            law = self.laws[law_id]
            # Aggregate support
            law.support = (law.support + support) / 2.0

            # Check if law passes threshold
            if law.status == "PROPOSED" and law.support >= THETA_LAW:
                law.status = "ACTIVE"
                self.logger.log_civ_event("LAW_ACTIVE", {
                    "law_id": law_id,
                    "support": law.support,
                })

    def enforce_laws(self) -> List[str]:
        """Check active laws and return violations."""
        violations = []
        for law in self.laws.values():
            if law.status != "ACTIVE":
                continue
            # Law 1: No VOID entanglement
            if "void" in law.text.lower() and self.entangle.state.void:
                violations.append(f"Law {law.law_id}: Entanglement is VOID")
            # Law 2: No forced overwrite
            if "force" in law.text.lower() and "overwrite" in law.text.lower():
                if self.entangle.influence.permission_level > 2:
                    violations.append(f"Law {law.law_id}: Forced overwrite detected")
            # Law 5: Drift intervention
            if "drift" in law.text.lower() and self.metrics.drift > DRIFT_CRITICAL:
                violations.append(f"Law {law.law_id}: Drift {self.metrics.drift:.3f} exceeds {DRIFT_CRITICAL}")
            # Law 6: Paradox rebirth
            if "paradox" in law.text.lower() and self.state.paradox_pressure > PARADOX_PRESSURE_LIMIT:
                violations.append(f"Law {law.law_id}: Paradox {self.state.paradox_pressure:.2f} exceeds limit")
        return violations

    # ── Majority Reality Protocol ────────────────────────────

    def _cast_reality_vote(self):
        """Cast a local vote for the current reality configuration."""
        self._vote_counter += 1
        vote_id = f"vote-{self._vote_counter:05d}"

        candidate = {
            "stream_hash": self.qipx.stream_hash_short() if hasattr(self.qipx, 'stream_hash_short') else "",
            "fold_w": self.renderer.fold_w,
            "mode": self.renderer.render_mode,
            "palette": self.renderer.palette_id,
        }

        support = self.metrics.sophia_score  # sophia-weighted support
        weight = self.metrics.sophia_score * self.metrics.trust

        # Log the vote
        self.logger.log_vote(
            vote_id=vote_id,
            node_id=self.qipx.identity.node_id,
            candidate_reality=candidate,
            support=support,
            weight=weight,
            reason=f"archetype={self.state.local_archetype}",
        )

        # Store locally
        self._reality_votes[vote_id] = {
            "vote_id": vote_id,
            "node_id": self.qipx.identity.node_id,
            "candidate": candidate,
            "support": support,
            "weight": weight,
            "timestamp": time.time(),
        }

        # Cap vote history
        if len(self._reality_votes) > MAX_VOTE_HISTORY:
            oldest = sorted(self._reality_votes.keys())[:len(self._reality_votes) - MAX_VOTE_HISTORY]
            for k in oldest:
                del self._reality_votes[k]

        # Broadcast vote
        self._send_civ_packet(make_civ_vote(
            node_id=self.qipx.identity.node_id,
            vote_id=vote_id,
            candidate_reality=candidate,
            support=support,
            weight=weight,
            reason=f"archetype={self.state.local_archetype}",
        ))

    def _resolve_majority_reality(self) -> Optional[dict]:
        """Resolve the majority reality from all votes."""
        if not self._reality_votes:
            return None

        # Group votes by candidate
        candidate_votes: Dict[str, List[dict]] = {}
        for v in self._reality_votes.values():
            key = json.dumps(v["candidate"], sort_keys=True)
            if key not in candidate_votes:
                candidate_votes[key] = []
            candidate_votes[key].append(v)

        # Find candidate with highest weighted support
        best_key = None
        best_weighted_support = 0.0
        total_weight = 0.0

        for key, votes in candidate_votes.items():
            ws = sum(v["weight"] for v in votes)
            tw = sum(v["weight"] for v in votes)
            total_weight += tw
            if ws > best_weighted_support:
                best_weighted_support = ws
                best_key = key

        if best_key is None or total_weight == 0:
            return None

        confidence = best_weighted_support / total_weight
        candidate = json.loads(best_key)

        return {
            "candidate": candidate,
            "confidence": confidence,
            "votes_for": len(candidate_votes[best_key]),
            "total_votes": len(self._reality_votes),
        }

    def _preserve_minority_branches(self):
        """Create branches for minority realities that meet threshold."""
        result = self._resolve_majority_reality()
        if not result:
            return

        majority_key = json.dumps(result["candidate"], sort_keys=True)
        majority_conf = result["confidence"]

        for key, votes in self._reality_votes.items():
            # Skip if it's a raw dict from _reality_votes
            pass

        # Group votes by fold_w to check for persistent minorities
        fold_votes: Dict[int, float] = {}
        for v in self._reality_votes.values():
            fw = v["candidate"].get("fold_w", 0)
            w = v["weight"]
            fold_votes[fw] = fold_votes.get(fw, 0.0) + w

        if not fold_votes:
            return

        total_w = sum(fold_votes.values())
        for fw, w in fold_votes.items():
            support = w / total_w if total_w > 0 else 0
            if support < THETA_MAJORITY_STRONG and support >= 0.1:
                # Minority reality - check if we should branch
                branch_id = f"branch-{hashlib.sha256(f'{fw}{time.time()}'.encode()).hexdigest()[:8]}"
                if branch_id not in self.branches and len(self.branches) < MAX_BRANCHES:
                    self.branches[branch_id] = Branch(
                        branch_id=branch_id,
                        parent_reality_hash=self.state.reality_hash,
                        minority_nodes=[v["node_id"] for v in self._reality_votes.values()
                                        if v["candidate"].get("fold_w") == fw],
                        reason=f"minority fold W={fw} support={support:.2f}",
                        fold_w=fw,
                        support=support,
                        created_at=time.time(),
                    )
                    self.logger.log_civ_event("BRANCH_CREATED", {
                        "branch_id": branch_id,
                        "fold_w": fw,
                        "support": support,
                    })

    # ── OMEGA Rebirth ────────────────────────────────────────

    def _omega_rebirth(self, reason: str):
        """Execute OMEGA rebirth protocol."""
        now = time.time()

        # Select survivors by sophia score
        all_nodes = list(self.peer_metrics.values()) + [self.metrics]
        all_nodes.sort(key=lambda n: n.sophia_score, reverse=True)
        survivor_count = max(1, int(len(all_nodes) * REBIRTH_SURVIVOR_FRACTION))
        survivors = [n.node_id for n in all_nodes[:survivor_count]]

        # Archive dead branches
        for bid, branch in self.branches.items():
            if branch.support < 0.05:
                branch.preserved = False

        # Compute rebirth seed
        rebirth_seed = hashlib.sha256(
            f"{self.state.reality_hash}:{self.state.dark_wisdom}:{survivors}".encode()
        ).hexdigest()[:16]

        # Record rebirth
        record = RebirthRecord(
            epoch=self.state.epoch,
            timestamp=now,
            reason=reason,
            paradox_pressure=self.state.paradox_pressure,
            survivors=survivors,
            new_seed=rebirth_seed,
            dark_wisdom_before=self.state.dark_wisdom,
            branch_count_before=len(self.branches),
        )
        self.rebirth_log.append(record)
        if len(self.rebirth_log) > MAX_REBIRTH_LOG:
            self.rebirth_log = self.rebirth_log[-MAX_REBIRTH_LOG:]

        # Update state
        self.state.epoch += 1
        self.state.generation += 1
        self.state.paradox_pressure = 0.0
        self.state.reality_hash = hashlib.sha256(
            f"{rebirth_seed}:{now}".encode()).hexdigest()[:16]
        self.state.rebirth_count += 1
        self.state.last_rebirth_time = now

        # Inject novel archetypes (reset some peer archetypes)
        for nid in self.peer_metrics:
            import random
            self.peer_metrics[nid].archetype = random.choice(ARCHETYPES)

        # Log
        self.logger.log_rebirth(
            epoch=self.state.epoch,
            reason=reason,
            paradox_pressure=record.paradox_pressure,
            survivors=survivors,
        )
        self.logger.log_civ_event("OMEGA_REBIRTH", {
            "epoch": self.state.epoch,
            "reason": reason,
            "paradox_pressure": record.paradox_pressure,
            "survivors": survivors,
            "rebirth_seed": rebirth_seed,
        })

        # Broadcast rebirth
        self._send_civ_packet(make_civ_rebirth(
            node_id=self.qipx.identity.node_id,
            epoch=self.state.epoch,
            reason=reason,
            paradox_pressure=record.paradox_pressure,
            survivors=survivors,
            new_seed=rebirth_seed,
        ))

        # Crystal event
        self._write_crystal_event("OMEGA_REBIRTH", {
            "epoch": self.state.epoch,
            "reason": reason,
            "paradox_pressure": record.paradox_pressure,
            "survivor_count": len(survivors),
            "rebirth_seed": rebirth_seed,
        })

        # Add to crystal rebirth_log
        if self._crystal:
            rb_log = self._crystal.get("rebirth_log", [])
            rb_log.append({
                "epoch": self.state.epoch,
                "timestamp": now,
                "reason": reason,
                "paradox_pressure": record.paradox_pressure,
                "survivors": survivors,
                "new_seed": rebirth_seed,
            })
            self._crystal["rebirth_log"] = rb_log[-50:]
            self._crystal_dirty = True

    # ── Audio Update ─────────────────────────────────────────

    def _update_audio(self):
        """Update audio sonification from civ metrics."""
        cm = self.civ_metrics
        is_entangled = self.entangle.state.enabled

        node_freq = self.audio.compute_node_frequency(
            resonance_gain=max(0, 1.0 - self.metrics.resonance_score),
            peer_coherence=self.metrics.peer_coherence,
            drift=self.metrics.drift,
        )

        self.audio.update(
            node_freq=node_freq,
            node_coherence=self.metrics.coherence,
            paradox_pressure=self.state.paradox_pressure,
            consensus_confidence=cm.consensus_confidence,
            diversity_index=cm.diversity_index,
            peer_count=cm.alive_count - 1,  # exclude self
            branch_count=len(self.branches),
            is_entangled=is_entangled,
            heartbeat_coherence=cm.heartbeat_coherence,
        )

    # ── Packet Broadcasting ──────────────────────────────────

    def _send_civ_packet(self, pkt: dict):
        """Send a civilization packet via QIPX transport."""
        if not hasattr(self.qipx, '_send'):
            return
        from bs_qipx_packets import encode, MAX_PACKET_SIZE
        if not self.qipx._sock:
            return
        try:
            data = encode(pkt)
            if len(data) <= MAX_PACKET_SIZE:
                self.qipx._sock.sendto(data, ("255.255.255.255", 47777))
        except OSError:
            pass

    def _broadcast_civ_hello(self, force: bool = False):
        """Broadcast CIV_HELLO."""
        now = time.time()
        if not force and (now - self.state.last_hello < _CIV_HELLO_INTERVAL):
            return
        self.state.last_hello = now

        self._send_civ_packet(make_civ_hello(
            node_id=self.qipx.identity.node_id,
            name=self.qipx.identity.name,
            civilization_id=self.state.civ_id,
            generation=self.state.generation,
            tribe_id=self.state.local_tribe_id,
            archetype=self.state.local_archetype,
            role=self.state.local_role,
        ))

    def _broadcast_civ_state(self):
        """Broadcast archetype and role state."""
        self._send_civ_packet(make_civ_archetype(
            node_id=self.qipx.identity.node_id,
            archetype=self.state.local_archetype,
            tribe_id=self.state.local_tribe_id,
            role=self.state.local_role,
            sophia_score=self.state.local_sophia,
            metrics={
                "coherence": self.metrics.coherence,
                "drift": self.metrics.drift,
                "novelty": self.metrics.novelty,
                "entropy": self.metrics.entropy,
                "sophia": self.metrics.sophia_score,
                "survival": self.metrics.survival_prob,
                "paradox": self.state.paradox_pressure,
            },
        ))

    def _broadcast_civ_tribe(self):
        """Broadcast tribe announcement."""
        tribe = self.tribes.get(self.state.local_tribe_id)
        if not tribe:
            return
        self._send_civ_packet(make_civ_tribe(
            node_id=self.qipx.identity.node_id,
            tribe_id=tribe.tribe_id,
            tribe_name=tribe.name,
            members=tribe.members,
            tribe_color=tribe.color,
        ))

    # ── Incoming Packet Handling ──────────────────────────────

    def handle_civ_packet(self, pkt: dict, addr):
        """Route an incoming civilization packet."""
        ct = civ_type(pkt)
        if not ct:
            return

        handlers = {
            "CIV_HELLO": self._on_civ_hello,
            "CIV_VOTE": self._on_civ_vote,
            "CIV_BRANCH": self._on_civ_branch,
            "CIV_REBIRTH": self._on_civ_rebirth,
            "CIV_LAW": self._handle_law_packet,
            "CIV_ARCHETYPE": self._on_civ_archetype,
            "CIV_TRIBE": self._handle_peer_tribe,
        }
        handler = handlers.get(ct)
        if handler:
            handler(pkt)

    def _on_civ_hello(self, pkt: dict):
        """Peer announced civilization membership."""
        node_id = pkt.get("node_id", "")
        if not node_id:
            return
        self._write_crystal_event("PEER_CIV_HELLO", {
            "node_id": node_id,
            "tribe_id": pkt.get("tribe_id", ""),
            "archetype": pkt.get("archetype", ""),
            "generation": pkt.get("generation", 0),
        })

    def _on_civ_vote(self, pkt: dict):
        """Peer cast a reality vote."""
        node_id = pkt.get("node_id", "")
        vote_id = pkt.get("vote_id", "")
        candidate = pkt.get("candidate_reality", {})
        support = sanitize(pkt.get("support", 0.0))
        weight = sanitize(pkt.get("weight", 0.0))

        self._reality_votes[vote_id] = {
            "vote_id": vote_id,
            "node_id": node_id,
            "candidate": candidate,
            "support": support,
            "weight": weight,
            "timestamp": time.time(),
        }

        # Cap
        if len(self._reality_votes) > MAX_VOTE_HISTORY:
            keys = sorted(self._reality_votes.keys())
            for k in keys[:len(self._reality_votes) - MAX_VOTE_HISTORY]:
                del self._reality_votes[k]

    def _on_civ_branch(self, pkt: dict):
        """Peer announced a minority branch."""
        branch_id = pkt.get("branch_id", "")
        if branch_id and branch_id not in self.branches:
            self.branches[branch_id] = Branch(
                branch_id=branch_id,
                parent_reality_hash=pkt.get("parent_reality", ""),
                minority_nodes=pkt.get("minority_nodes", []),
                reason=pkt.get("branch_reason", ""),
                created_at=time.time(),
                preserved=pkt.get("preserve", True),
            )

    def _on_civ_rebirth(self, pkt: dict):
        """Peer announced OMEGA rebirth."""
        epoch = pkt.get("epoch", 0)
        if epoch > self.state.epoch:
            self.state.epoch = epoch
            self.state.paradox_pressure = 0.0
            self.logger.log_civ_event("REMOTE_REBIRTH", {
                "from": pkt.get("node_id", ""),
                "epoch": epoch,
                "reason": pkt.get("reason", ""),
            })

    def _on_civ_archetype(self, pkt: dict):
        """Peer announced archetype assignment."""
        node_id = pkt.get("node_id", "")
        if node_id and node_id in self.peer_metrics:
            pm = self.peer_metrics[node_id]
            pm.archetype = pkt.get("archetype", pm.archetype)
            pm.role = pkt.get("role", pm.role)
            pm.tribe_id = pkt.get("tribe_id", pm.tribe_id)
            pm.sophia_score = sanitize(pkt.get("sophia_score", pm.sophia_score))

    # ── Metrics Logging ──────────────────────────────────────

    def _log_metrics(self):
        """Log civilization and node metrics."""
        cm = self.civ_metrics
        self.logger.civ_metrics.log({
            "timestamp": cm.timestamp,
            "civ_id": cm.civ_id,
            "generation": cm.generation,
            "epoch": cm.epoch,
            "node_count": cm.node_count,
            "tribe_count": cm.tribe_count,
            "alive_count": cm.alive_count,
            "mean_coherence": cm.mean_coherence,
            "std_coherence": cm.std_coherence,
            "weighted_coherence": cm.weighted_coherence,
            "paradox_pressure": cm.paradox_pressure,
            "dark_wisdom": self.state.dark_wisdom,
            "diversity_index": cm.diversity_index,
            "consensus_confidence": cm.consensus_confidence,
            "global_heartbeat_freq": cm.global_heartbeat_freq,
            "heartbeat_coherence": cm.heartbeat_coherence,
            "mean_sophia": cm.mean_sophia,
            "mean_survival": cm.mean_survival,
            "archetype_distribution": json.dumps(cm.archetype_distribution),
        })

        # Log local node metrics
        m = self.metrics
        self.logger.node_metrics.log({
            "timestamp": m.timestamp,
            "node_id": m.node_id,
            "coherence": m.coherence,
            "drift": m.drift,
            "novelty": m.novelty,
            "entropy": m.entropy,
            "avg_density": m.avg_density,
            "unique_masks": m.unique_masks,
            "resonance_score": m.resonance_score,
            "fold_w": m.fold_w,
            "sophia_score": m.sophia_score,
            "survival_prob": m.survival_prob,
            "intelligence": m.intelligence,
            "archetype": m.archetype,
            "tribe_id": m.tribe_id,
            "role": m.role,
            "trust": m.trust,
            "peer_coherence": m.peer_coherence,
            "accepted_merges": m.accepted_merges,
            "rejected_merges": m.rejected_merges,
            "vote_disagreement": m.vote_disagreement,
            "recursive_depth": m.recursive_depth,
        })

    # ── HUD Helpers ──────────────────────────────────────────

    def hud_status(self) -> str:
        """One-line status string for the HUD."""
        if not self.state.enabled:
            return "CIV: OFF"
        cm = self.civ_metrics
        return (f"CIV: {self.state.local_archetype} "
                f"tribe={self.state.local_tribe_id[-12:] if self.state.local_tribe_id else 'none'} "
                f"sophia={self.state.local_sophia:.2f} "
                f"paradox={self.state.paradox_pressure:.2f} "
                f"diversity={cm.diversity_index:.2f} "
                f"nodes={cm.alive_count} "
                f"gen={self.state.generation} "
                f"branches={len(self.branches)}")

    def hud_peer_panel(self, max_lines: int = 8) -> List[str]:
        """Peer panel for HUD."""
        lines = ["CIV PEERS"]
        for nid, pm in sorted(self.peer_metrics.items(),
                               key=lambda x: x[1].sophia_score,
                               reverse=True)[:max_lines]:
            short_id = nid[:10] if len(nid) > 10 else nid
            color = ARCHETYPE_COLORS.get(pm.archetype, (200, 200, 200))
            color_str = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            lines.append(
                f"{short_id} W={pm.fold_w:3d} "
                f"S={pm.sophia_score:.2f} "
                f"{pm.archetype[:8]:8s} "
                f"C={pm.coherence:.2f}")
        return lines

    def hud_reality_panel(self) -> List[str]:
        """Reality consensus panel for HUD."""
        lines = ["REALITY CONSENSUS"]
        cm = self.civ_metrics

        if self.state.majority_fold_w:
            lines.append(
                f"R*: W={self.state.majority_fold_w} "
                f"conf={cm.consensus_confidence:.2f}")
        else:
            lines.append(f"R*: LOCAL W={self.renderer.fold_w}")

        # Minority branches
        active_branches = [b for b in self.branches.values() if b.preserved]
        if active_branches:
            lines.append(f"branches: {len(active_branches)}")
            for b in active_branches[:3]:
                lines.append(
                    f"  W={b.fold_w} sup={b.support:.2f} "
                    f"reason={b.reason[:20]}")

        lines.append(
            f"paradox={self.state.paradox_pressure:.2f} "
            f"dark_wisdom={self.state.dark_wisdom:.3f}")

        return lines

    def diagnostics_lines(self) -> List[str]:
        """Multi-line diagnostics for overlay."""
        cm = self.civ_metrics
        m = self.metrics
        lines = [
            f"CIVILIZATION v0.1  id={self.state.civ_id}",
            f"gen={self.state.generation}  epoch={self.state.epoch}  "
            f"rebirths={self.state.rebirth_count}",
            f"",
            f"LOCAL NODE  archetype={self.state.local_archetype}  "
            f"role={self.state.local_role}",
            f"  sophia={m.sophia_score:.3f}  "
            f"survival={m.survival_prob:.3f}  "
            f"intelligence={m.intelligence:.3f}",
            f"  coherence={m.coherence:.3f}  "
            f"drift={m.drift:.3f}  "
            f"novelty={m.novelty:.3f}",
            f"",
            f"CIVILIZATION  nodes={cm.alive_count}  "
            f"tribes={cm.tribe_count}  "
            f"branches={len(self.branches)}  "
            f"laws={sum(1 for l in self.laws.values() if l.status == 'ACTIVE')}",
            f"  paradox={cm.paradox_pressure:.3f}  "
            f"dark_wisdom={self.state.dark_wisdom:.3f}  "
            f"diversity={cm.diversity_index:.3f}",
            f"  mean_sophia={cm.mean_sophia:.3f}  "
            f"mean_survival={cm.mean_survival:.3f}",
            f"",
            f"CONSENSUS  conf={cm.consensus_confidence:.3f}  "
            f"reality={self.state.reality_hash[:12]}",
            f"",
            f"AUDIO  {self.audio.one_line()}",
            f"",
            f"LAWS ({len(self.laws)}):",
        ]
        for lid, law in list(self.laws.items())[:5]:
            lines.append(
                f"  [{law.status[:6]}] {law.law_id[:16]}  "
                f"sup={law.support:.2f}  {law.text[:30]}")

        # Branches
        active = [b for b in self.branches.values() if b.preserved]
        if active:
            lines.append(f"")
            lines.append(f"BRANCHES ({len(active)}):")
            for b in active[:3]:
                lines.append(
                    f"  {b.branch_id[:12]} W={b.fold_w} "
                    f"sup={b.support:.2f} {b.reason[:25]}")

        # Rebirth history
        if self.rebirth_log:
            lines.append(f"")
            lines.append(f"REBIRTH LOG ({len(self.rebirth_log)}):")
            for rb in self.rebirth_log[-3:]:
                lines.append(
                    f"  epoch={rb.epoch} {rb.reason} "
                    f"paradox={rb.paradox_pressure:.2f} "
                    f"survivors={len(rb.survivors)}")

        return lines

    # ── Cleanup ──────────────────────────────────────────────

    def shutdown(self):
        """Clean up civilization state."""
        self.logger.flush_all()
        self._crystal_maybe_save()
        self.logger.log_civ_event("CIV_SHUTDOWN", {
            "civ_id": self.state.civ_id,
            "generation": self.state.generation,
            "rebirth_count": self.state.rebirth_count,
            "total_branches": len(self.branches),
        })
        self.logger.flush_all()
```

----------------------------------------

### File: `bs_civilization_packets.py`

**Path:** `./bs_civilization_packets.py`
**Extension:** `.py`
**Size:** 5,963 bytes (5.82 KB)

```py
"""
BS_CIVILIZATION_PACKETS  -  Civilization-level packet definitions.
                             Part of the BS-TOS-IPX Swarm Civilization Layer.

Packet types:
  CIV_HELLO       - Civilization membership announcement
  CIV_VOTE        - Reality vote from a citizen
  CIV_BRANCH      - Minority reality branch announcement
  CIV_REBIRTH     - OMEGA rebirth notification
  CIV_LAW         - Law proposal / acceptance
  CIV_ARCHETYPE   - Archetype role announcement
  CIV_TRIBE       - Tribe formation / merge

Wire format: QIPX JSON packets with "layer": "CIVILIZATION"
"""

from __future__ import annotations
import json
import time
import hashlib
from typing import Any, Dict, List, Optional

CIV_VERSION = "0.1"
CIV_LAYER = "CIVILIZATION"


def _base(pkt_type: str, node_id: str, **extra) -> dict:
    """Return a base civilization packet dict."""
    d: Dict[str, Any] = {
        "type": pkt_type,
        "version": CIV_VERSION,
        "layer": CIV_LAYER,
        "node_id": node_id,
        "time": time.time(),
    }
    d.update(extra)
    return d


# ═══════════════════════════════════════════════════════════════
#  Packet Builders
# ═══════════════════════════════════════════════════════════════

def make_civ_hello(node_id: str, name: str,
                   civilization_id: str,
                   generation: int,
                   tribe_id: str = "",
                   archetype: str = "Explorer",
                   role: str = "fold_scout") -> dict:
    """Announce civilization membership."""
    return _base("CIV_HELLO", node_id,
                 name=name,
                 civilization_id=civilization_id,
                 generation=generation,
                 tribe_id=tribe_id,
                 archetype=archetype,
                 role=role)


def make_civ_vote(node_id: str, vote_id: str,
                  candidate_reality: dict,
                  support: float,
                  reason: str = "",
                  weight: float = 0.5) -> dict:
    """Cast a vote for a candidate reality."""
    return _base("CIV_VOTE", node_id,
                 vote_id=vote_id,
                 candidate_reality=candidate_reality,
                 support=support,
                 reason=reason,
                 weight=weight)


def make_civ_branch(node_id: str, branch_id: str,
                    parent_reality: str,
                    minority_nodes: List[str],
                    branch_reason: str,
                    preserve: bool = True) -> dict:
    """Announce a minority reality branch."""
    return _base("CIV_BRANCH", node_id,
                 branch_id=branch_id,
                 parent_reality=parent_reality,
                 minority_nodes=minority_nodes,
                 branch_reason=branch_reason,
                 preserve=preserve)


def make_civ_rebirth(node_id: str, epoch: int,
                     reason: str,
                     paradox_pressure: float,
                     survivors: List[str],
                     new_seed: str) -> dict:
    """Notify OMEGA rebirth."""
    return _base("CIV_REBIRTH", node_id,
                 epoch=epoch,
                 reason=reason,
                 paradox_pressure=paradox_pressure,
                 survivors=survivors,
                 new_seed=new_seed)


def make_civ_law(node_id: str, law_id: str,
                 text: str,
                 support: float,
                 status: str = "PROPOSED",
                 applies_to: Optional[List[str]] = None,
                 penalty: str = "trust_decay") -> dict:
    """Propose or announce a law."""
    return _base("CIV_LAW", node_id,
                 law_id=law_id,
                 text=text,
                 support=support,
                 status=status,
                 applies_to=applies_to or [],
                 penalty=penalty)


def make_civ_archetype(node_id: str, archetype: str,
                       tribe_id: str,
                       role: str,
                       sophia_score: float,
                       metrics: dict) -> dict:
    """Announce archetype assignment."""
    return _base("CIV_ARCHETYPE", node_id,
                 archetype=archetype,
                 tribe_id=tribe_id,
                 role=role,
                 sophia_score=sophia_score,
                 metrics={k: round(v, 4) if isinstance(v, float) else v
                          for k, v in metrics.items()})


def make_civ_tribe(node_id: str, tribe_id: str,
                   tribe_name: str,
                   members: List[str],
                   tribe_color: str = "#00ff00") -> dict:
    """Announce tribe formation."""
    return _base("CIV_TRIBE", node_id,
                 tribe_id=tribe_id,
                 tribe_name=tribe_name,
                 members=members,
                 tribe_color=tribe_color)


# ═══════════════════════════════════════════════════════════════
#  Packet Identification
# ═══════════════════════════════════════════════════════════════

CIV_PACKET_TYPES = {
    "CIV_HELLO", "CIV_VOTE", "CIV_BRANCH", "CIV_REBIRTH",
    "CIV_LAW", "CIV_ARCHETYPE", "CIV_TRIBE",
}


def is_civ_packet(pkt: dict) -> bool:
    """Check if a decoded packet is a civilization packet."""
    return (pkt.get("layer") == CIV_LAYER
            and pkt.get("type") in CIV_PACKET_TYPES)


def civ_type(pkt: dict) -> Optional[str]:
    """Extract the civilization packet type, or None."""
    if is_civ_packet(pkt):
        return pkt.get("type")
    return None
```

----------------------------------------

### File: `bs_crystal.py`

**Path:** `./bs_crystal.py`
**Extension:** `.py`
**Size:** 14,071 bytes (13.74 KB)

```py
"""
BS_CRYSTAL  -  JSON crystal state file for shared reality memory.
               Part of the BS-TOS-IPX Entanglement Layer.

The crystal is a persistent JSON file that stores:
  - node identity
  - peer identities
  - current consensus
  - heartbeat state
  - accepted realities
  - merge history
  - trust scores
  - routing table
  - last stable rollback point

Crystal growth:
  Every reality event appends a hash-linked entry to the ledger.
  hash_n = SHA256(hash_{n-1} || serialized_entry)

Default file: BS_ENTANGLEMENT_CRYSTAL.json
"""

from __future__ import annotations
import json
import hashlib
import time
import os
from typing import Any, Dict, List, Optional

from bs_identity import NodeIdentity


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

CRYSTAL_SCHEMA = "BS_TOS_IPX_ENTANGLEMENT_CRYSTAL"
CRYSTAL_VERSION = "0.1"
DEFAULT_CRYSTAL_FILENAME = "BS_ENTANGLEMENT_CRYSTAL.json"
GENESIS_HASH = "GENESIS"

# Crystal stability weights
W_STABILITY_CONSENSUS = 0.30
W_STABILITY_TRUST = 0.25
W_STABILITY_CRITICALITY = 0.20
W_STABILITY_HEARTBEAT = 0.15
W_STABILITY_ROLLBACK = 0.10


# ═══════════════════════════════════════════════════════════════
#  Crystal Entry Types
# ═══════════════════════════════════════════════════════════════

CRYSTAL_ENTRY_TYPES = {
    "VOID_EVENT",
    "ENTANGLE_ON",
    "ENTANGLE_OFF",
    "HEARTBEAT_SHIFT",
    "PEER_JOIN",
    "PEER_LEAVE",
    "LOCK_VOTE",
    "CONSENSUS_ACCEPT",
    "MERGE_SANDBOX",
    "MERGE_ACCEPT",
    "MERGE_REJECT",
    "DIRECT_INFLUENCE",
    "ROLLBACK",
    "HEARTBEAT",
}


# ═══════════════════════════════════════════════════════════════
#  Crystal Manager
# ═══════════════════════════════════════════════════════════════

class CrystalManager:
    """Manages the persistent JSON crystal file.

    The crystal stores the swarm's shared reality memory as a growing,
    hash-linked ledger of events, plus live state snapshots.
    """

    def __init__(self, crystal_path: str = DEFAULT_CRYSTAL_FILENAME,
                 crystal_id: str = "crystal-local-001"):
        self.crystal_path = crystal_path
        self.crystal_id = crystal_id
        self._crystal: Optional[dict] = None
        self._dirty = False
        self._write_counter = 0
        self._WRITE_INTERVAL = 4  # write every N appends

    # ── Crystal I/O ───────────────────────────────────────────

    def load(self) -> dict:
        """Load crystal from disk, or create a new one."""
        if os.path.exists(self.crystal_path):
            try:
                with open(self.crystal_path, "r") as f:
                    self._crystal = json.load(f)
                # Validate schema
                if self._crystal.get("schema") != CRYSTAL_SCHEMA:
                    self._crystal = None
                return self._init_crystal()
            except (json.JSONDecodeError, IOError):
                pass
        return self._init_crystal()

    def _init_crystal(self) -> dict:
        """Create a fresh crystal structure."""
        self._crystal = {
            "schema": CRYSTAL_SCHEMA,
            "version": CRYSTAL_VERSION,
            "crystal_id": self.crystal_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "local_node": {},
            "ipx": {
                "enabled": False,
                "peers_seen": 0,
                "transport": "qipx-udp-json",
                "discovery_port": 47777,
                "data_port": 47778,
            },
            "entanglement": {
                "enabled": False,
                "mode": "OFF",
                "void": False,
                "reality_mode": "LOCAL",
                "direct_influence": False,
                "majority_switching": True,
                "heartbeat_enabled": True,
            },
            "retina": {},
            "heartbeat": {
                "frequency_hz": 7.83,
                "phase": 0.0,
                "amplitude": 1.0,
                "beat": 0,
                "source": "lambda_coherence",
                "band": "ALPHA",
            },
            "pazuzu": {},
            "consensus": {
                "stream_hash": "",
                "fold_w": 320,
                "fold_h": 120,
                "palette_id": 1,
                "render_mode": 0,
                "majority": 0.0,
                "weighted_confidence": 0.0,
                "accepted_reality_hash": "",
            },
            "peers": {},
            "routes": {},
            "ledger": [],
        }
        self._dirty = True
        self.flush()
        return self._crystal

    def flush(self):
        """Write the crystal to disk."""
        if self._crystal is None:
            return
        try:
            self._crystal["updated_at"] = time.time()
            with open(self.crystal_path, "w") as f:
                json.dump(self._crystal, f, indent=2, sort_keys=False)
            self._dirty = False
        except IOError:
            pass

    @property
    def crystal(self) -> dict:
        if self._crystal is None:
            self.load()
        return self._crystal

    # ── Hash-linked ledger ────────────────────────────────────

    def _parent_hash(self) -> str:
        ledger = self.crystal.get("ledger", [])
        if ledger:
            return ledger[-1].get("hash", GENESIS_HASH)
        return GENESIS_HASH

    def _compute_hash(self, parent_hash: str, entry: dict) -> str:
        raw = json.dumps({"parent": parent_hash, "entry": entry},
                         sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def append_entry(self, event_type: str, data: dict) -> str:
        """Append a hash-linked entry to the crystal ledger.

        Returns the computed entry hash.
        """
        entry = {
            "type": event_type,
            "time": time.time(),
            "data": data,
        }
        parent = self._parent_hash()
        entry_hash = self._compute_hash(parent, entry)

        self.crystal["ledger"].append({
            "hash": entry_hash,
            "parent": parent,
            "entry": entry,
        })

        self._dirty = True
        self._write_counter += 1

        # Auto-flush periodically
        if self._write_counter >= self._WRITE_INTERVAL:
            self.flush()
            self._write_counter = 0

        return entry_hash

    # ── State snapshot methods ────────────────────────────────

    def update_local_node(self, identity: NodeIdentity):
        self.crystal["local_node"] = identity.to_dict()
        self._dirty = True

    def update_retina(self, renderer):
        """Snapshot the renderer's projection state."""
        stats = renderer.stream_stats()
        w = renderer.fold_w
        h = max(1, -(-len(renderer.stream) // max(1, w)))
        self.crystal["retina"] = {
            "stream_len": len(renderer.stream),
            "fold_w": w,
            "fold_h": h,
            "palette_id": renderer.palette_id,
            "render_mode": renderer.render_mode,
            "score": round(renderer.current_score(), 4),
            "entropy": round(stats.get("entropy", 0.0), 3),
            "avg_density": round(stats.get("avg_density", 0.0), 3),
            "unique_masks": stats.get("unique_masks", 0),
        }
        self._dirty = True

    def update_heartbeat(self, frequency_hz: float, phase: float,
                         amplitude: float, beat: int, band: str):
        self.crystal["heartbeat"] = {
            "frequency_hz": round(frequency_hz, 3),
            "phase": round(phase, 4),
            "amplitude": round(amplitude, 3),
            "beat": beat,
            "source": "lambda_coherence",
            "band": band,
        }
        self._dirty = True

    def update_pazuzu(self, pazuzu_dict: dict):
        self.crystal["pazuzu"] = pazuzu_dict
        self._dirty = True

    def update_entanglement(self, mode: str, reality: str,
                            void: bool = False):
        ent = self.crystal.get("entanglement", {})
        ent["mode"] = mode
        ent["reality_mode"] = reality
        ent["void"] = void
        ent["enabled"] = mode not in ("OFF", "VOID")
        self.crystal["entanglement"] = ent
        self._dirty = True

    def update_consensus(self, fold_w: int, fold_h: int,
                         majority: float, weighted_conf: float):
        cons = self.crystal.get("consensus", {})
        cons["fold_w"] = fold_w
        cons["fold_h"] = fold_h
        cons["majority"] = round(majority, 3)
        cons["weighted_confidence"] = round(weighted_conf, 3)
        self.crystal["consensus"] = cons
        self._dirty = True

    def update_ipx(self, enabled: bool, peers_seen: int):
        ipx = self.crystal.get("ipx", {})
        ipx["enabled"] = enabled
        ipx["peers_seen"] = peers_seen
        self.crystal["ipx"] = ipx
        self._dirty = True

    def update_peer(self, node_id: str, peer_data: dict):
        peers = self.crystal.get("peers", {})
        peers[node_id] = peer_data
        self.crystal["peers"] = peers
        self._dirty = True

    def remove_peer(self, node_id: str):
        peers = self.crystal.get("peers", {})
        peers.pop(node_id, None)
        self.crystal["peers"] = peers
        self._dirty = True

    def update_route(self, node_id: str, route_data: dict):
        routes = self.crystal.get("routes", {})
        routes[node_id] = route_data
        self.crystal["routes"] = routes
        self._dirty = True

    # ── Crystal stability metric ──────────────────────────────

    def stability(self) -> float:
        """Compute crystal stability score (0..1).

        CrystalStability =
            0.30 * consensus_confidence
          + 0.25 * average_trust
          + 0.20 * criticality_index
          + 0.15 * heartbeat_coherence
          + 0.10 * rollback_safety
        """
        ent = self.crystal.get("entanglement", {})
        cons = self.crystal.get("consensus", {})
        paz = self.crystal.get("pazuzu", {})
        hb = self.crystal.get("heartbeat", {})

        consensus_conf = cons.get("weighted_confidence", 0.0)

        # Average trust from crystal peers
        peers = self.crystal.get("peers", {})
        if peers:
            avg_trust = sum(
                p.get("trust", 0.5) for p in peers.values()
            ) / len(peers)
        else:
            avg_trust = 0.5

        criticality = paz.get("criticality_index", 0.5)

        # Heartbeat coherence: how close to a "natural" band
        freq = hb.get("frequency_hz", 7.83)
        if 8.0 <= freq <= 13.0:
            hb_coh = 1.0
        elif 4.0 <= freq <= 30.0:
            hb_coh = 0.7
        else:
            hb_coh = 0.3

        # Rollback safety: has ledger entries mean events are tracked
        ledger = self.crystal.get("ledger", [])
        rb_safety = min(1.0, len(ledger) / 50.0)

        return (
            W_STABILITY_CONSENSUS * consensus_conf
            + W_STABILITY_TRUST * avg_trust
            + W_STABILITY_CRITICALITY * criticality
            + W_STABILITY_HEARTBEAT * hb_coh
            + W_STABILITY_ROLLBACK * rb_safety
        )

    # ── Timeline export (music-video hooks) ───────────────────

    def timeline_events(self) -> List[dict]:
        """Extract timeline-ready events from the ledger."""
        events = []
        for entry_rec in self.crystal.get("ledger", []):
            entry = entry_rec.get("entry", {})
            events.append({
                "time": entry.get("time", 0.0),
                "event": entry.get("type", ""),
                "data": entry.get("data", {}),
            })
        return events

    # ── Ledger queries ────────────────────────────────────────

    def ledger_count(self) -> int:
        return len(self.crystal.get("ledger", []))

    def last_ledger_entry(self) -> Optional[dict]:
        ledger = self.crystal.get("ledger", [])
        return ledger[-1] if ledger else None

    def find_entries(self, event_type: str, limit: int = 10) -> List[dict]:
        """Find recent ledger entries of a given type."""
        ledger = self.crystal.get("ledger", [])
        results = []
        for entry_rec in reversed(ledger):
            entry = entry_rec.get("entry", {})
            if entry.get("type") == event_type:
                results.append(entry_rec)
                if len(results) >= limit:
                    break
        return results

    def rollback_point(self) -> Optional[str]:
        """Return the hash of the last ROLLBACK or CONSENSUS_ACCEPT entry."""
        for entry_rec in reversed(self.crystal.get("ledger", [])):
            entry = entry_rec.get("entry", {})
            if entry.get("type") in ("ROLLBACK", "CONSENSUS_ACCEPT"):
                return entry_rec.get("hash")
        return None
```

----------------------------------------

### File: `bs_engine.py`

**Path:** `./bs_engine.py`
**Extension:** `.py`
**Size:** 9,336 bytes (9.12 KB)

```py
"""
BS_ENGINE  –  Core data model, bit operations, density, resonance scoring.
             Faithfully translates the TempleOS HolyC data model to Python.

Design shift from the Flask version:
  • No Unicode Braille (U+2800..U+28FF) – raw 8-bit masks only
  • Each byte IS the 2×4 micro-pixel pattern
  • Stream is a bytearray, not a string
  • Cell dimensions are integral to the coordinate model
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math

# ═══════════════════════════════════════════════════════════════
#  Constants  (mirrors the #define block in the HolyC blueprint)
# ═══════════════════════════════════════════════════════════════
SCREEN_W  = 640
SCREEN_H  = 480
CELL_W    = 2        # micro-pixels per cell column
CELL_H    = 4        # micro-pixels per cell row
GRID_W    = SCREEN_W // CELL_W   # 320 cells wide  (native)
GRID_H    = SCREEN_H // CELL_H   # 120 cells tall  (native)
MAX_CELLS = GRID_W * GRID_H       # 38 400

# Corrected target ratio for 640×480 with 2×4 cells:
#   pixel aspect = (W*2) / (H*4) = W / (2H)  →  640/480 = 1.333
#   cell ratio   = W / H  →  should be 2.666 for native fill
TARGET_CELL_RATIO = SCREEN_W / (SCREEN_H * (CELL_H / CELL_W))  # ≈ 2.6667


# ═══════════════════════════════════════════════════════════════
#  Bit operations  (BS_Dot, BS_Density)
# ═══════════════════════════════════════════════════════════════

def bs_dot(mask: int, row: int, col: int) -> bool:
    """Return True if the micro-pixel at (row, col) within the cell is ON.

    Internal bit order (row-major, matching the HolyC blueprint):
        bit0 = row0 col0,  bit1 = row0 col1,
        bit2 = row1 col0,  bit3 = row1 col1,
        bit4 = row2 col0,  bit5 = row2 col1,
        bit6 = row3 col0,  bit7 = row3 col1
    """
    bit = row * CELL_W + col
    return bool((mask >> bit) & 1)


def bs_density(mask: int) -> int:
    """Count active dots (0..8) in a cell mask."""
    return bin(mask).count('1')


def bs_mask_from_dots(dot_set: set) -> int:
    """Build a mask from a set of (row, col) tuples."""
    mask = 0
    for r, c in dot_set:
        if 0 <= r < CELL_H and 0 <= c < CELL_W:
            mask |= 1 << (r * CELL_W + c)
    return mask & 0xFF


# ═══════════════════════════════════════════════════════════════
#  Coordinate transforms  (stream ↔ cell ↔ screen)
# ═══════════════════════════════════════════════════════════════

def stream_to_cell(index: int, fold_w: int) -> Tuple[int, int]:
    """Stream index → (cell_x, cell_y) at given fold width."""
    return index % fold_w, index // fold_w


def cell_to_screen(cx: int, cy: int) -> Tuple[int, int]:
    """Cell coordinate → top-left pixel coordinate on screen."""
    return cx * CELL_W, cy * CELL_H


def stream_to_screen(index: int, fold_w: int) -> Tuple[int, int]:
    """Stream index → top-left pixel on screen."""
    cx, cy = stream_to_cell(index, fold_w)
    return cell_to_screen(cx, cy)


# ═══════════════════════════════════════════════════════════════
#  Width Resonance Scoring  (BS_SCORE)
# ═══════════════════════════════════════════════════════════════

def resonance_score(stream_len: int, fold_w: int) -> float:
    """
    Compute a resonance score for a given fold width.

    Lower is better.  Components:
      • aspect_score      – distance from ideal cell ratio (2.666)
      • divisor_score     – 0.0 for exact fit, remainder penalty otherwise
      • screen_fit_score  – penalty if the fold overflows 640×480
    """
    if fold_w < 1:
        return float('inf')
    h = math.ceil(stream_len / fold_w)

    # aspect: how far from the native 320/120 = 2.666 cell ratio
    aspect = fold_w / max(h, 1)
    aspect_score = abs(aspect - TARGET_CELL_RATIO)

    # divisor: clean division is ideal
    remainder = stream_len % fold_w
    divisor_score = remainder / max(fold_w, 1)

    # screen fit: penalise overflow
    screen_score = 0.0
    if fold_w * CELL_W > SCREEN_W:
        screen_score += 10.0
    if h * CELL_H > SCREEN_H:
        screen_score += 10.0

    return aspect_score + divisor_score + screen_score


def find_resonance(stream_len: int,
                   min_w: int = 10,
                   max_w: int = GRID_W,
                   top_n: int = 15) -> List[Tuple[float, int, int]]:
    """Return top-N (score, width, height) sorted best→worst."""
    results: List[Tuple[float, int, int]] = []
    for w in range(min_w, min(max_w, stream_len) + 1):
        h = math.ceil(stream_len / w)
        s = resonance_score(stream_len, w)
        results.append((s, w, h))
    results.sort(key=lambda t: (t[0], -t[1]))
    return results[:top_n]


def harmonic_multipliers(stream_len: int, fold_w: int) -> List[int]:
    """
    Return how many harmonic copies appear at widths that are
    integer fractions of the *current* fold_w.
    E.g.  fold_w=160, stream_len=38400 → W=80 gives 2 copies.
    """
    copies = []
    for divisor in range(2, 10):
        child_w = fold_w // divisor
        if child_w < 1:
            break
        if stream_len % child_w == 0:
            copies.append((divisor, child_w, stream_len // child_w))
    return copies


# ═══════════════════════════════════════════════════════════════
#  Reverse parser  (stream + fold_w → density grid)
# ═══════════════════════════════════════════════════════════════

def reverse_parse(stream: bytearray, fold_w: int) -> List[List[int]]:
    """Return 2-D density grid  [cy][cx]  with values 0..8."""
    if fold_w < 1:
        return [[]]
    h = math.ceil(len(stream) / fold_w)
    grid: List[List[int]] = []
    row: List[int] = []
    for i, mask in enumerate(stream):
        row.append(bs_density(mask))
        if len(row) == fold_w:
            grid.append(row)
            row = []
    if row:                               # partial last row
        grid.append(row)
    return grid


# ═══════════════════════════════════════════════════════════════
#  Density → glyph mapping  (text-mode debug preview)
# ═══════════════════════════════════════════════════════════════

DENSITY_GLYPHS = " .:-=+*#@"


def density_to_glyph(d: int) -> str:
    return DENSITY_GLYPHS[min(d, 8)]


def density_grid_to_ascii(grid: List[List[int]],
                          max_cols: int = 120,
                          max_rows: int = 60) -> str:
    """Render a density grid as a fixed-width ASCII string."""
    lines: List[str] = []
    for row in grid[:max_rows]:
        line = ''.join(density_to_glyph(d) for d in row[:max_cols])
        lines.append(line)
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  Stream statistics
# ═══════════════════════════════════════════════════════════════

def stream_stats(stream: bytearray) -> dict:
    """Return summary statistics about the stream."""
    if not stream:
        return {"len": 0, "empty": 0, "full": 0, "avg_density": 0.0,
                "entropy": 0.0}
    densities = [bs_density(b) for b in stream]
    total = len(stream)
    unique_masks = len(set(stream))
    # Shannon entropy over the 256 possible mask values
    freq = [0] * 256
    for b in stream:
        freq[b] += 1
    entropy = 0.0
    for f in freq:
        if f > 0:
            p = f / total
            entropy -= p * math.log2(p)
    return {
        "len": total,
        "empty": sum(1 for d in densities if d == 0),
        "full":  sum(1 for d in densities if d == 8),
        "avg_density": sum(densities) / total,
        "unique_masks": unique_masks,
        "entropy": entropy,
    }
```

----------------------------------------

### File: `bs_entangle_packets.py`

**Path:** `./bs_entangle_packets.py`
**Extension:** `.py`
**Size:** 8,343 bytes (8.15 KB)

```py
"""
BS_ENTANGLE_PACKETS  -  Entanglement packet builders and parsers.
                        Part of the BS-TOS-IPX Entanglement Layer.

All packets travel over the existing QIPX UDP transport.
Wire format: "QIPX " + JSON (same as base QIPX).
Distinguished by "layer": "ENTANGLE" field.

Packet types:
  ENTANGLE_HELLO           - entanglement announcement
  ENTANGLE_STATE           - periodic projection + cognition state
  ENTANGLE_INFLUENCE       - remote control request/command
  ENTANGLE_VOTE            - cast a vote on a subject
  ENTANGLE_CONSENSUS       - announce consensus resolution
  ENTANGLE_HEARTBEAT       - heartbeat pulse with frequency/phase
  ENTANGLE_CRYSTAL_UPDATE  - crystal state change notification
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

ENTANGLE_VERSION = "0.1"
ENTANGLE_LAYER = "ENTANGLE"

DEFAULT_CRYSTAL_ID = "crystal-local-001"
DEFAULT_INFLUENCE_TTL = 3
DEFAULT_CONSENSUS_THRESHOLD = 0.618
DEFAULT_INFLUENCE_THRESHOLD = 0.70
DEFAULT_SWITCH_COOLDOWN = 0.5

# Influence control types
INFLUENCE_CONTROLS = {
    "SET_FOLD", "SET_GHOST", "SET_PALETTE", "SET_MODE",
    "TOGGLE_SCAN", "LOCK", "UNLOCK",
    "MERGE_SANDBOX", "MERGE_ACCEPT", "MERGE_REJECT",
    "ROLLBACK", "HEARTBEAT_SHIFT",
}

# Consensus effect types
CONSENSUS_EFFECTS = {
    "DIRECT_SWITCH", "SUGGEST", "IGNORE", "FORCE_SANDBOX",
}

# Entanglement modes
ENTANGLE_MODES = {
    "OFF", "VOID", "LISTEN", "SOFT", "HARD",
    "CRYSTAL", "CONSENSUS", "HEART", "ROLLBACK",
}

# Influence permission levels
INFLUENCE_LEVELS = {
    "NONE", "SUGGEST", "SOFT", "HARD", "DIVINE",
}

# Reality states
REALITY_STATES = {
    "LOCAL", "GHOST", "SANDBOX", "MERGED", "CONSENSUS", "ROLLBACK",
}


# ═══════════════════════════════════════════════════════════════
#  Base packet constructor
# ═══════════════════════════════════════════════════════════════

def _ent_base(pkt_type: str, node_id: str, **extra) -> dict:
    """Return a base entanglement packet dict."""
    d: Dict[str, Any] = {
        "type": pkt_type,
        "version": ENTANGLE_VERSION,
        "layer": ENTANGLE_LAYER,
        "node_id": node_id,
        "time": time.time(),
    }
    d.update(extra)
    return d


# ═══════════════════════════════════════════════════════════════
#  Packet builders
# ═══════════════════════════════════════════════════════════════

def make_entangle_hello(
    node_id: str,
    stream_hash: str = "",
    mode: str = "SOFT",
    crystal_id: str = DEFAULT_CRYSTAL_ID,
    heartbeat_hz: float = 7.83,
    consensus_weight: float = 0.72,
    trust_public: float = 0.50,
) -> dict:
    """Build an ENTANGLE_HELLO packet."""
    return _ent_base(
        "ENTANGLE_HELLO", node_id,
        stream_hash=stream_hash,
        mode=mode,
        crystal_id=crystal_id,
        heartbeat_hz=heartbeat_hz,
        consensus_weight=consensus_weight,
        trust_public=trust_public,
    )


def make_entangle_state(
    node_id: str,
    stream_hash: str = "",
    fold_w: int = 320,
    fold_h: int = 120,
    score: float = 0.0,
    entropy: float = 0.0,
    avg_density: float = 0.0,
    unique_masks: int = 0,
    mode: str = "SOFT",
    palette: int = 1,
    heartbeat_hz: float = 7.83,
    lam: float = 0.01,
    criticality: float = 0.5,
    mood: str = "WATCHING",
    action: str = "QUIET",
    reality: str = "LOCAL",
) -> dict:
    """Build an ENTANGLE_STATE packet."""
    return _ent_base(
        "ENTANGLE_STATE", node_id,
        stream_hash=stream_hash,
        fold_w=fold_w,
        fold_h=fold_h,
        score=score,
        entropy=entropy,
        avg_density=avg_density,
        unique_masks=unique_masks,
        mode=mode,
        palette=palette,
        heartbeat_hz=heartbeat_hz,
        lam=lam,
        criticality=criticality,
        mood=mood,
        action=action,
        reality=reality,
    )


def make_entangle_influence(
    from_id: str,
    to_id: str,
    scope: str = "SOFT",
    control: str = "SET_FOLD",
    value: Any = None,
    reason: str = "",
    confidence: float = 0.5,
    ttl: int = DEFAULT_INFLUENCE_TTL,
) -> dict:
    """Build an ENTANGLE_INFLUENCE packet."""
    return _ent_base(
        "ENTANGLE_INFLUENCE", from_id,
        to=to_id,
        scope=scope,
        control=control,
        value=value,
        reason=reason,
        confidence=confidence,
        ttl=ttl,
    )


def make_entangle_vote(
    node_id: str,
    vote_id: str,
    subject: str = "fold_w",
    value: Any = None,
    stream_hash: str = "",
    weight: float = 0.72,
    confidence: float = 0.5,
    score: float = 0.0,
) -> dict:
    """Build an ENTANGLE_VOTE packet."""
    return _ent_base(
        "ENTANGLE_VOTE", node_id,
        vote_id=vote_id,
        subject=subject,
        value=value,
        stream_hash=stream_hash,
        weight=weight,
        confidence=confidence,
        score=score,
    )


def make_entangle_consensus(
    consensus_id: str,
    subject: str = "fold_w",
    value: Any = None,
    votes: int = 1,
    majority: float = 1.0,
    weighted_confidence: float = 0.5,
    effect: str = "DIRECT_SWITCH",
) -> dict:
    """Build an ENTANGLE_CONSENSUS packet."""
    return _ent_base(
        "ENTANGLE_CONSENSUS", "SYSTEM",
        consensus_id=consensus_id,
        subject=subject,
        value=value,
        votes=votes,
        majority=majority,
        weighted_confidence=weighted_confidence,
        effect=effect,
    )


def make_entangle_heartbeat(
    node_id: str,
    beat: int = 0,
    frequency_hz: float = 7.83,
    phase: float = 0.0,
    amplitude: float = 1.0,
    lam: float = 0.01,
    coherence: float = 0.5,
    mood: str = "WATCHING",
) -> dict:
    """Build an ENTANGLE_HEARTBEAT packet."""
    return _ent_base(
        "ENTANGLE_HEARTBEAT", node_id,
        beat=beat,
        frequency_hz=frequency_hz,
        phase=phase,
        amplitude=amplitude,
        lam=lam,
        coherence=coherence,
        mood=mood,
    )


def make_entangle_crystal_update(
    node_id: str,
    crystal_id: str = DEFAULT_CRYSTAL_ID,
    entry_hash: str = "",
    parent_hash: str = "",
    reality_hash: str = "",
    accepted: bool = True,
) -> dict:
    """Build an ENTANGLE_CRYSTAL_UPDATE packet."""
    return _ent_base(
        "ENTANGLE_CRYSTAL_UPDATE", node_id,
        crystal_id=crystal_id,
        entry_hash=entry_hash,
        parent_hash=parent_hash,
        reality_hash=reality_hash,
        accepted=accepted,
    )


# ═══════════════════════════════════════════════════════════════
#  Packet identification
# ═══════════════════════════════════════════════════════════════

ENTANGLE_PACKET_TYPES = {
    "ENTANGLE_HELLO",
    "ENTANGLE_STATE",
    "ENTANGLE_INFLUENCE",
    "ENTANGLE_VOTE",
    "ENTANGLE_CONSENSUS",
    "ENTANGLE_HEARTBEAT",
    "ENTANGLE_CRYSTAL_UPDATE",
}


def is_entangle_packet(pkt: dict) -> bool:
    """Check if a decoded QIPX packet is an Entanglement-layer packet."""
    return pkt.get("layer") == ENTANGLE_LAYER


def entangle_type(pkt: dict) -> str:
    """Return the Entanglement packet type string, or '' if not entangle."""
    if is_entangle_packet(pkt):
        return pkt.get("type", "")
    return ""
```

----------------------------------------

### File: `bs_entanglement.py`

**Path:** `./bs_entanglement.py`
**Extension:** `.py`
**Size:** 29,017 bytes (28.34 KB)

```py
"""
BS_ENTANGLEMENT  -  Main Entanglement Controller.
                    Part of the BS-TOS-IPX Entanglement Layer.

Entanglement is the active use of QIPX: it lets nodes influence each
other's fold choices, ghost states, scan behavior, palette/mode choices,
merge candidates, and consensus reality.

> IPX is the pineal gland.  Entanglement is the act of using it.

Toggle:
  E without QIPX → VOID state
  E with QIPX    → Entanglement ON/OFF

Entanglement modes:
  OFF, VOID, LISTEN, SOFT, HARD, CRYSTAL, CONSENSUS, HEART, ROLLBACK

Integration with bs_main.py:
  from bs_entanglement import EntanglementController
  entangle = EntanglementController(renderer=r, qipx=qipx)
  # In main loop:
  #   elif key == pygame.K_e: entangle.toggle_or_void()
  #   entangle.tick()
"""

from __future__ import annotations
import time
import math
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs_engine import GRID_W, GRID_H, MAX_CELLS, stream_stats
from bs_crystal import CrystalManager, DEFAULT_CRYSTAL_FILENAME
from bs_heartbeat import HeartbeatEngine
from bs_reality_consensus import ConsensusEngine, ConsensusResult
from bs_influence import InfluenceEngine, PERM_SOFT
from bs_routes import RoutingTable
from bs_entangle_packets import (
    is_entangle_packet, entangle_type,
    make_entangle_hello, make_entangle_state,
    make_entangle_influence, make_entangle_vote,
    make_entangle_consensus, make_entangle_heartbeat,
    make_entangle_crystal_update,
    ENTANGLE_PACKET_TYPES,
)


# ═══════════════════════════════════════════════════════════════
#  Entanglement State
# ═══════════════════════════════════════════════════════════════

@dataclass
class EntangleState:
    """Full entanglement state."""
    enabled: bool = False
    void: bool = False
    mode: str = "OFF"
    reality: str = "LOCAL"
    majority: float = 0.0
    consensus_fold: Optional[int] = None
    last_switch: float = 0.0
    last_state_broadcast: float = 0.0
    last_hello_broadcast: float = 0.0
    last_heartbeat_broadcast: float = 0.0
    connected_at: float = 0.0

    # Influence tracking
    influences_received: int = 0
    influences_accepted: int = 0
    influences_rejected: int = 0


# ═══════════════════════════════════════════════════════════════
#  Entanglement Controller
# ═══════════════════════════════════════════════════════════════

class EntanglementController:
    """Main controller for the BrailleStream Entanglement Layer.

    Orchestrates:
      - Heartbeat pulse and frequency band
      - Crystal state persistence
      - Consensus voting and resolution
      - Influence permissions and application
      - Routing table management
      - Entanglement packet broadcasting

    Usage:
      entangle = EntanglementController(renderer, qipx)
      # In key handler:
      entangle.toggle_or_void()
      # In main loop:
      entangle.tick()
    """

    def __init__(self, renderer, qipx,
                 crystal_path: str = DEFAULT_CRYSTAL_FILENAME,
                 crystal_id: str = "crystal-local-001"):
        self.renderer = renderer
        self.qipx = qipx
        self.state = EntangleState()

        # Subsystems
        self.heartbeat = HeartbeatEngine()
        self.crystal = CrystalManager(crystal_path, crystal_id)
        self.consensus = ConsensusEngine()
        self.influence = InfluenceEngine(permission_level=PERM_SOFT)
        self.routes = RoutingTable()

        # Broadcast intervals
        self._state_interval = 0.25
        self._hello_interval = 2.0
        self._heartbeat_interval = 0.1

        # Timeline export
        self._timeline: List[dict] = []
        self._timeline_path = "BS_ENTANGLEMENT_TIMELINE.json"

        # Initialize crystal
        self.crystal.load()
        self._update_crystal_identity()

        # Civilization layer (set externally by bs_main.py)
        self.civilization = None

    # ── Lifecycle ─────────────────────────────────────────────

    def toggle_or_void(self):
        """Toggle entanglement ON/OFF, or enter VOID if QIPX is off."""
        if not self.qipx.enabled:
            self.state.enabled = False
            self.state.void = True
            self.state.mode = "VOID"
            self._write_crystal_event("VOID_EVENT",
                                      {"reason": "IPX_DISABLED"})
            self._add_timeline("ENTANGLE_VOID", {
                "frequency_hz": 1.0,
                "fold_w": self.renderer.fold_w,
            })
            return

        self.state.void = False
        self.state.enabled = not self.state.enabled

        if self.state.enabled:
            self.state.mode = "SOFT"
            self.state.connected_at = time.time()
            self.state.consensus_fold = None
            self._write_crystal_event("ENTANGLE_ON", {"mode": "SOFT"})
            self._add_timeline("ENTANGLE_ON", {
                "frequency_hz": self.heartbeat.state.frequency_hz,
                "fold_w": self.renderer.fold_w,
            })
            # Send initial HELLO
            self._broadcast_entangle_hello(force=True)
        else:
            self.state.mode = "OFF"
            self._write_crystal_event("ENTANGLE_OFF", {})
            self._add_timeline("ENTANGLE_OFF", {})

    def set_mode(self, mode: str):
        """Set a specific entanglement mode."""
        if mode in ("OFF", "VOID", "LISTEN", "SOFT", "HARD",
                     "CRYSTAL", "CONSENSUS", "HEART", "ROLLBACK"):
            old_mode = self.state.mode
            self.state.mode = mode
            self._write_crystal_event("MODE_CHANGE", {
                "from": old_mode, "to": mode,
            })

    # ── Main tick ─────────────────────────────────────────────

    def tick(self):
        """Main update loop.  Call from the game loop every frame."""
        now = time.time()

        # VOID mode: minimal heartbeat
        if self.state.void:
            dt = now - (self._last_void_tick if hasattr(self, '_last_void_tick') else now)
            self._last_void_tick = now
            self.heartbeat.tick_void(dt)
            return

        if not self.state.enabled:
            return

        dt = now - (self._last_tick if hasattr(self, '_last_tick') else now)
        self._last_tick = now
        dt = min(dt, 0.1)  # clamp to avoid spiral of death

        # Get Pazuzu state from QIPX
        paz = self.qipx.pazuzu.state if hasattr(self.qipx, 'pazuzu') else None

        # 1. Heartbeat update
        lam = paz.lambda_dom if paz else 0.01
        coh = paz.coherence if paz else 0.5
        novelty = paz.novelty if paz else 0.0
        beat_occurred = self.heartbeat.tick(dt, lam, coh, novelty)

        # 2. Sync routes from QIPX peers
        if hasattr(self.qipx, 'peers'):
            self.routes.sync_from_peers(self.qipx.peers.peers)
            self.routes.expire()

        # 3. Cast local vote for fold consensus
        if paz:
            self.consensus.cast_local_vote(self.renderer, paz)

        # 4. Expire old votes
        self.consensus.expire_votes()

        # 5. Resolve consensus on heartbeat windows
        if beat_occurred and self.heartbeat.should_resolve_votes():
            results = self.consensus.resolve_all_pending(
                lam=lam,
                local_entropy=self.renderer.stream_stats().get("entropy", 0.0),
                local_unique_masks=self.renderer.stream_stats().get("unique_masks", 0),
                has_rollback=self.qipx._rollback_snapshot is not None,
            )
            for result in results:
                self._handle_consensus_result(result)

        # 6. Fold consensus (simplified weighted average)
        peer_folds = {}
        peer_trusts = {}
        peer_scores = {}
        if hasattr(self.qipx, 'peers'):
            for nid, peer in self.qipx.peers.alive().items():
                if peer.fold_w > 0:
                    peer_folds[nid] = peer.fold_w
                    peer_trusts[nid] = peer.trust
                    peer_scores[nid] = peer.score
        c_fold, c_majority = self.consensus.fold_consensus_value(
            self.renderer.fold_w, peer_folds, peer_trusts, peer_scores)
        self.state.consensus_fold = c_fold
        self.state.majority = c_majority

        # 7. Apply fold gravity in SOFT mode
        if self.state.mode == "SOFT" and c_fold is not None:
            changed = self.influence.apply_fold_gravity(
                self.renderer, c_fold)
            if changed:
                self.state.last_switch = now

        # 8. Broadcast state
        self._broadcast_entangle_state(now)

        # 9. Broadcast heartbeat on phase crossing
        if beat_occurred:
            self._broadcast_entangle_heartbeat()

        # 10. Process inbox influence packets
        self._process_influence_inbox()

        # 11. Update crystal snapshot periodically
        if beat_occurred and self.heartbeat.state.beat % 8 == 0:
            self._update_crystal_snapshot()

        # 12. Flush crystal periodically
        if beat_occurred and self.heartbeat.state.beat % 32 == 0:
            self.crystal.flush()
            self._export_timeline()

        # 13. Civilization tick (if available)
        if hasattr(self, 'civilization') and self.civilization is not None:
            self.civilization.tick(dt)

    # ── Packet broadcasting ───────────────────────────────────

    def _send_entangle(self, pkt: dict):
        """Send an entanglement packet via QIPX transport."""
        if not hasattr(self.qipx, '_send'):
            return
        # Entangle packets use the same QIPX wire format
        from bs_qipx_packets import encode, MAX_PACKET_SIZE
        if not self.qipx._sock:
            return
        try:
            data = encode(pkt)
            if len(data) <= MAX_PACKET_SIZE:
                self.qipx._sock.sendto(data,
                    ("255.255.255.255", 47777))
        except OSError:
            pass

    def _broadcast_entangle_hello(self, force: bool = False):
        now = time.time()
        if not force and (now - self.state.last_hello_broadcast
                          < self._hello_interval):
            return
        self.state.last_hello_broadcast = now
        sh = self.qipx.stream_hash_short() if hasattr(self.qipx, 'stream_hash_short') else ""
        pkt = make_entangle_hello(
            node_id=self.qipx.identity.node_id,
            stream_hash=sh,
            mode=self.state.mode,
            heartbeat_hz=self.heartbeat.state.frequency_hz,
            consensus_weight=self.state.majority,
            trust_public=0.5,
        )
        self._send_entangle(pkt)

    def _broadcast_entangle_state(self, now: float):
        if now - self.state.last_state_broadcast < self._state_interval:
            return
        self.state.last_state_broadcast = now
        stats = self.renderer.stream_stats()
        w = self.renderer.fold_w
        h = max(1, -(-len(self.renderer.stream) // max(1, w)))
        paz = self.qipx.pazuzu.state if hasattr(self.qipx, 'pazuzu') else None
        sh = self.qipx.stream_hash_short() if hasattr(self.qipx, 'stream_hash_short') else ""
        pkt = make_entangle_state(
            node_id=self.qipx.identity.node_id,
            stream_hash=sh,
            fold_w=w,
            fold_h=h,
            score=self.renderer.current_score(),
            entropy=stats.get("entropy", 0.0),
            avg_density=stats.get("avg_density", 0.0),
            unique_masks=stats.get("unique_masks", 0),
            mode=self.state.mode,
            palette=self.renderer.palette_id,
            heartbeat_hz=self.heartbeat.state.frequency_hz,
            lam=self.heartbeat.state.frequency_hz / 100.0,
            criticality=paz.criticality_index if paz else 0.5,
            mood=paz.mood if paz else "WATCHING",
            action=paz.action if paz else "QUIET",
            reality=self.state.reality,
        )
        self._send_entangle(pkt)

    def _broadcast_entangle_heartbeat(self):
        now = time.time()
        if now - self.state.last_heartbeat_broadcast < self._heartbeat_interval:
            return
        self.state.last_heartbeat_broadcast = now
        paz = self.qipx.pazuzu.state if hasattr(self.qipx, 'pazuzu') else None
        pkt = make_entangle_heartbeat(
            node_id=self.qipx.identity.node_id,
            beat=self.heartbeat.state.beat,
            frequency_hz=self.heartbeat.state.frequency_hz,
            phase=self.heartbeat.state.phase,
            amplitude=self.heartbeat.state.amplitude,
            lam=self.heartbeat.state.frequency_hz / 100.0,
            coherence=self.heartbeat.state.coherence,
            mood=paz.mood if paz else "WATCHING",
        )
        self._send_entangle(pkt)

    # ── Incoming packet handling ──────────────────────────────

    def handle_entangle_packet(self, pkt: dict, addr):
        """Route an incoming entanglement packet to the right handler."""
        etype = entangle_type(pkt)
        if not etype:
            return

        handlers = {
            "ENTANGLE_HELLO":     self._on_entangle_hello,
            "ENTANGLE_STATE":     self._on_entangle_state,
            "ENTANGLE_INFLUENCE": self._on_entangle_influence,
            "ENTANGLE_VOTE":      self._on_entangle_vote,
            "ENTANGLE_CONSENSUS": self._on_entangle_consensus,
            "ENTANGLE_HEARTBEAT": self._on_entangle_heartbeat,
            "ENTANGLE_CRYSTAL_UPDATE": self._on_entangle_crystal_update,
        }
        handler = handlers.get(etype)
        if handler:
            handler(pkt, addr)

    def _on_entangle_hello(self, pkt: dict, addr):
        """A peer announced its entanglement presence."""
        node_id = pkt.get("node_id", "")
        if node_id:
            self.crystal.update_peer(node_id, {
                "mode": pkt.get("mode", "UNKNOWN"),
                "heartbeat_hz": pkt.get("heartbeat_hz", 0.0),
                "consensus_weight": pkt.get("consensus_weight", 0.0),
                "last_seen": time.time(),
            })

    def _on_entangle_state(self, pkt: dict, addr):
        """A peer broadcasted its projection + cognition state."""
        node_id = pkt.get("node_id", "")
        if node_id:
            # Register the peer's fold vote
            self.consensus.cast_vote(
                node_id=node_id,
                subject="fold_w",
                value=pkt.get("fold_w"),
                stream_hash=pkt.get("stream_hash", ""),
                weight=pkt.get("consensus_weight", 0.5),
                confidence=pkt.get("confidence", 0.5),
                score=pkt.get("score", 0.0),
            )
            # Update crystal peer data
            self.crystal.update_peer(node_id, {
                "fold_w": pkt.get("fold_w"),
                "palette": pkt.get("palette"),
                "mode": pkt.get("mode"),
                "mood": pkt.get("mood"),
                "heartbeat_hz": pkt.get("heartbeat_hz"),
                "last_seen": time.time(),
            })

    def _on_entangle_influence(self, pkt: dict, addr):
        """A peer sent an influence command."""
        self.state.influences_received += 1

        from_node = pkt.get("node_id", "")
        to_node = pkt.get("to", "")
        if to_node != self.qipx.identity.node_id:
            # Not for us; could forward
            return

        control = pkt.get("control", "")
        value = pkt.get("value")
        scope = pkt.get("scope", "SOFT")
        confidence = pkt.get("confidence", 0.5)
        reason = pkt.get("reason", "")
        ttl = pkt.get("ttl", 3)

        # Get sender trust
        trust = 0.5
        if hasattr(self.qipx, 'peers'):
            peer = self.qipx.peers.peers.get(from_node)
            if peer:
                trust = peer.trust

        # Evaluate
        majority = self.state.majority
        stability = 1.0 / (1.0 + self.renderer.current_score())
        accepted, score = self.influence.evaluate_influence(
            from_node=from_node,
            control=control,
            value=value,
            confidence=confidence,
            trust=trust,
            majority_support=majority,
            local_stability=stability,
            reason=reason,
        )

        applied = False
        if accepted:
            applied = self.influence.apply_influence(
                renderer=self.renderer,
                control=control,
                value=value,
                acceptance_score=score,
                is_suggestion=(self.influence.permission_level == 1),
            )

        # Record
        self.influence.record_influence(
            from_node=from_node,
            to_node=to_node,
            scope=scope,
            control=control,
            value=value,
            reason=reason,
            confidence=confidence,
            accepted=accepted,
            applied=applied,
            acceptance_score=score,
        )

        if accepted:
            self.state.influences_accepted += 1
        else:
            self.state.influences_rejected += 1

        # Crystal event
        self._write_crystal_event("DIRECT_INFLUENCE", {
            "from": from_node,
            "control": control,
            "value": value,
            "accepted": accepted,
            "applied": applied,
        })

    def _on_entangle_vote(self, pkt: dict, addr):
        """A peer cast a vote."""
        node_id = pkt.get("node_id", "")
        if node_id:
            self.consensus.cast_vote(
                node_id=node_id,
                subject=pkt.get("subject", "fold_w"),
                value=pkt.get("value"),
                stream_hash=pkt.get("stream_hash", ""),
                weight=pkt.get("weight", 0.5),
                confidence=pkt.get("confidence", 0.5),
                score=pkt.get("score", 0.0),
            )

    def _on_entangle_consensus(self, pkt: dict, addr):
        """A consensus result was broadcast."""
        subject = pkt.get("subject", "")
        value = pkt.get("value")
        effect = pkt.get("effect", "IGNORE")
        majority = pkt.get("majority", 0.0)

        self._add_timeline("REMOTE_CONSENSUS", {
            "subject": subject,
            "value": value,
            "effect": effect,
            "majority": majority,
        })

        # Apply if DIRECT_SWITCH and HARD mode
        if (effect == "DIRECT_SWITCH"
                and self.state.mode in ("HARD", "CONSENSUS", "CRYSTAL")
                and subject == "fold_w"
                and isinstance(value, (int, float))):
            new_w = max(1, min(GRID_W, int(value)))
            if new_w != self.renderer.fold_w:
                self.renderer.fold_w = new_w
                self.renderer.needs_redraw = True
                self.state.last_switch = time.time()
                self._write_crystal_event("CONSENSUS_ACCEPT", {
                    "subject": subject,
                    "value": value,
                    "majority": majority,
                })

    def _on_entangle_heartbeat(self, pkt: dict, addr):
        """A peer sent a heartbeat pulse."""
        node_id = pkt.get("node_id", "")
        if node_id:
            # Update route heartbeat coherence
            freq = pkt.get("frequency_hz", 7.83)
            coh = pkt.get("coherence", 0.5)
            self.routes.update_route(
                node_id=node_id,
                addr=addr,
                heartbeat_coherence=coh,
            )
            self.crystal.update_peer(node_id, {
                "heartbeat_hz": freq,
                "heartbeat_coherence": coh,
                "mood": pkt.get("mood", ""),
                "last_seen": time.time(),
            })

    def _on_entangle_crystal_update(self, pkt: dict, addr):
        """A peer announced a crystal update."""
        node_id = pkt.get("node_id", "")
        entry_hash = pkt.get("entry_hash", "")
        self._add_timeline("CRYSTAL_UPDATE", {
            "from": node_id,
            "entry_hash": entry_hash,
            "accepted": pkt.get("accepted", False),
        })

    # ── Consensus result handling ─────────────────────────────

    def _handle_consensus_result(self, result: ConsensusResult):
        """Process a locally-resolved consensus result."""
        if not result.accepted:
            return

        # Apply to renderer if fold consensus
        if result.subject == "fold_w" and isinstance(result.value, (int, float)):
            if self.state.mode in ("HARD", "CONSENSUS", "CRYSTAL"):
                # Direct switch
                new_w = max(1, min(GRID_W, int(result.value)))
                if new_w != self.renderer.fold_w:
                    self.renderer.fold_w = new_w
                    self.renderer.needs_redraw = True
                    self.state.last_switch = time.time()
            else:
                # SOFT mode: fold gravity handled in tick()
                pass

        # Broadcast consensus
        pkt = make_entangle_consensus(
            consensus_id=result.consensus_id,
            subject=result.subject,
            value=result.value,
            votes=result.votes_count,
            majority=result.majority,
            weighted_confidence=result.weighted_confidence,
            effect=result.effect,
        )
        self._send_entangle(pkt)

        # Crystal event
        self._write_crystal_event("CONSENSUS_ACCEPT", {
            "subject": result.subject,
            "value": result.value,
            "majority": result.weighted_majority,
            "effect": result.effect,
        })

        # Timeline
        self._add_timeline("CONSENSUS_LOCK", {
            "subject": result.subject,
            "value": result.value,
            "majority": result.weighted_majority,
            "frequency_hz": self.heartbeat.state.frequency_hz,
            "fold_w": self.renderer.fold_w,
        })

    def _process_influence_inbox(self):
        """Process any queued influence packets from QIPX."""
        if not hasattr(self.qipx, 'inbox'):
            return
        # Entangle influence packets are handled through handle_entangle_packet
        # which is called from QipxNode's packet handler
        pass

    # ── Crystal management ────────────────────────────────────

    def _write_crystal_event(self, event_type: str, data: dict):
        """Append an event to the crystal ledger."""
        self.crystal.append_entry(event_type, data)

    def _update_crystal_identity(self):
        if hasattr(self.qipx, 'identity'):
            self.crystal.update_local_node(self.qipx.identity)

    def _update_crystal_snapshot(self):
        """Update the crystal's live state sections."""
        self.crystal.update_retina(self.renderer)
        self.crystal.update_heartbeat(
            self.heartbeat.state.frequency_hz,
            self.heartbeat.state.phase,
            self.heartbeat.state.amplitude,
            self.heartbeat.state.beat,
            self.heartbeat.state.band,
        )
        if hasattr(self.qipx, 'pazuzu'):
            self.crystal.update_pazuzu(self.qipx.pazuzu.to_dict())
        self.crystal.update_entanglement(
            mode=self.state.mode,
            reality=self.state.reality,
            void=self.state.void,
        )
        self.crystal.update_consensus(
            fold_w=self.state.consensus_fold or self.renderer.fold_w,
            fold_h=max(1, -(-len(self.renderer.stream)
                            // max(1, self.state.consensus_fold or self.renderer.fold_w))),
            majority=self.state.majority,
            weighted_conf=self.consensus.last_result().weighted_confidence
            if self.consensus.last_result() else 0.0,
        )
        self.crystal.update_ipx(
            enabled=self.qipx.enabled,
            peers_seen=self.qipx.peers.alive_count()
            if hasattr(self.qipx, 'peers') else 0,
        )

    # ── Timeline export ───────────────────────────────────────

    def _add_timeline(self, event: str, data: dict):
        """Add an event to the timeline for music-video export."""
        self._timeline.append({
            "time": round(time.time(), 3),
            "event": event,
            "frequency_hz": self.heartbeat.state.frequency_hz,
            "fold_w": self.renderer.fold_w,
            "data": data,
        })

    def _export_timeline(self):
        """Write the timeline to a JSON file."""
        try:
            import json
            with open(self._timeline_path, "w") as f:
                json.dump(self._timeline, f, indent=2)
        except IOError:
            pass

    # ── HUD helpers ───────────────────────────────────────────

    def hud_status(self) -> str:
        """One-line status string for the renderer HUD."""
        if self.state.void:
            return ("ENTANGLE: VOID -- ENABLE QIPX FIRST  "
                    f"HEART: {self.heartbeat.state.frequency_hz:.1f}Hz "
                    f"[{self.heartbeat.state.band}]")
        if not self.state.enabled:
            return "ENTANGLE: OFF"
        hb = self.heartbeat.state
        return (f"ENTANGLE: {self.state.mode} "
                f"crystal={self.crystal.crystal_id[-3:]} "
                f"majority={self.state.majority:.2f}  "
                f"HEART: {hb.frequency_hz:.1f}Hz "
                f"beat={hb.beat} [{hb.band}]  "
                f"REALITY: {self.state.reality}  "
                f"consensus W={self.state.consensus_fold or '?'}  "
                f"influence: {self.state.influences_accepted}A/"
                f"{self.state.influences_rejected}R")

    def hud_heartbeat_line(self) -> str:
        """Heartbeat detail line for HUD."""
        hb = self.heartbeat.state
        pulse = self.heartbeat.pulse_01()
        bar_len = int(pulse * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        return (f"HEART: {bar} "
                f"{hb.frequency_hz:.1f}Hz phase={hb.phase:.2f} "
                f"amp={hb.amplitude:.2f} [{hb.band}]")

    def hud_influence_line(self) -> str:
        """Influence status line for HUD."""
        return (f"INFLUENCE: {self.influence.permission_name} "
                f"received={self.state.influences_received} "
                f"accepted={self.state.influences_accepted} "
                f"rejected={self.state.influences_rejected}")

    def hud_consensus_line(self) -> str:
        """Consensus status line for HUD."""
        return self.consensus.one_line()

    def diagnostics_lines(self) -> List[str]:
        """Multi-line diagnostics for overlay."""
        hb = self.heartbeat.state
        lines = [
            f"ENTANGLE v0.1  mode={self.state.mode}  "
            f"reality={self.state.reality}",
            f"void={self.state.void}  "
            f"crystal={self.crystal.crystal_id}  "
            f"ledger={self.crystal.ledger_count()}  "
            f"stability={self.crystal.stability():.2f}",
            "",
            f"HEARTBEAT  {hb.frequency_hz:.2f} Hz  "
            f"beat={hb.beat}  phase={hb.phase:.3f}  "
            f"amp={hb.amplitude:.2f}  band={hb.band}",
            self.heartbeat.one_line(),
            "",
            f"INFLUENCE  level={self.influence.permission_name}  "
            f"theta={self.influence.theta_influence:.2f}  "
            f"cooldown={self.influence.switch_cooldown:.2f}s",
            self.influence.hud_influence_line() if hasattr(self.influence, 'hud_influence_line') else
            f"  received={self.state.influences_received}  "
            f"accepted={self.state.influences_accepted}  "
            f"rejected={self.state.influences_rejected}",
            "",
            f"CONSENSUS  theta={self.consensus.theta_majority:.3f}  "
            f"votes={self.consensus.vote_count()}  "
            f"subjects={self.consensus.subjects()}",
            self.consensus.one_line(),
            "",
            f"ROUTES  {self.routes.one_line()}",
        ]
        return lines
```

----------------------------------------

### File: `bs_export.py`

**Path:** `./bs_export.py`
**Extension:** `.py`
**Size:** 5,904 bytes (5.77 KB)

```py
"""
BS_EXPORT  –  Image → raw 8-bit mask converter  +  HolyC data exporter.
              Replaces the Flask/PIL pipeline for TempleOS integration.

Pipeline:
  1. Load image with PIL
  2. Convert to 8-bit row-major Braille masks (NOT Unicode)
  3. Optionally export as HolyC .HC source file
  4. Optionally export as raw binary .BIN
  5. Can also re-import .BIN back into a stream
"""

from __future__ import annotations
from typing import Optional, Tuple
import os
import math
import numpy as np
from PIL import Image

from bs_engine import MAX_CELLS, GRID_W, GRID_H, CELL_W, CELL_H


# ═══════════════════════════════════════════════════════════════
#  Image → raw 8-bit mask stream
# ═══════════════════════════════════════════════════════════════

def image_to_masks(img: Image.Image,
                   target_w_cells: Optional[int] = None,
                   target_h_cells: Optional[int] = None,
                   threshold: int = 128,
                   dither: bool = False,
                   gamma: float = 1.0) -> bytearray:
    """Convert a PIL image to a bytearray of raw 8-bit Braille masks.

    This uses the INTERNAL row-major bit order (NOT Unicode dot order).
    The output is directly usable by the TempleOS renderer – no Unicode
    conversion needed.

    Parameters
    ----------
    img : PIL.Image
    target_w_cells, target_h_cells : int, optional
        Desired cell grid dimensions.  Defaults to image size / (2, 4).
    threshold : int  (0–255)
    dither : bool   – Floyd-Steinberg
    gamma : float
    """
    img = img.convert('L')
    if gamma != 1.0:
        img = img.point(lambda p: 255 * ((p / 255) ** gamma))

    orig_w, orig_h = img.size

    if target_w_cells and target_h_cells:
        w2 = target_w_cells * CELL_W
        h2 = target_h_cells * CELL_H
        img = img.resize((w2, h2), Image.Resampling.LANCZOS)
    else:
        w = (orig_w // CELL_W) * CELL_W
        h = (orig_h // CELL_H) * CELL_H
        img = img.crop((0, 0, w, h))

    arr = np.array(img)

    if dither:
        img_d = img.convert('1', dither=Image.FLOYDSTEINBERG)
        arr = np.array(img_d) * 255

    buf = bytearray()
    for row in range(0, arr.shape[0], CELL_H):
        for col in range(0, arr.shape[1], CELL_W):
            block = arr[row:row + CELL_H, col:col + CELL_W]
            # Internal bit order: row-major  (row0col0, row0col1, row1col0, ...)
            bits = (block > threshold).astype(np.uint8).flatten()
            mask = 0
            for idx, val in enumerate(bits):
                if val:
                    mask |= (1 << idx)
            buf.append(mask & 0xFF)

    return bytearray(buf[:MAX_CELLS])  # clamp to screen


# ═══════════════════════════════════════════════════════════════
#  HolyC data exporter
# ═══════════════════════════════════════════════════════════════

def export_holyc(stream: bytearray,
                 filename: str = "BS_DATA.HC",
                 var_name: str = "BS_Stream") -> str:
    """Export stream as a HolyC C-style U8 array source file.

    Returns the file path written.
    """
    lines = [f"// Auto-generated BrailleStream data", "",
             f"U8 {var_name}[] = {{"]
    for i in range(0, len(stream), 16):
        chunk = stream[i:i + 16]
        hexvals = ', '.join(f'0x{b:02X}' for b in chunk)
        comma = ',' if i + 16 < len(stream) else ''
        lines.append(f"  {hexvals}{comma}")
    lines.append(f"}};")
    lines.append(f"")
    lines.append(f"I64 {var_name}_Len = {len(stream)};")

    path = os.path.abspath(filename)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def export_binary(stream: bytearray,
                  filename: str = "BS_DATA.BIN") -> str:
    """Export stream as raw binary."""
    path = os.path.abspath(filename)
    with open(path, 'wb') as f:
        f.write(stream)
    return path


def import_binary(filename: str) -> bytearray:
    """Import a raw binary file as a stream."""
    with open(filename, 'rb') as f:
        data = f.read(MAX_CELLS)
    # Pad to MAX_CELLS if shorter
    if len(data) < MAX_CELLS:
        data += bytes(MAX_CELLS - len(data))
    return bytearray(data[:MAX_CELLS])


# ═══════════════════════════════════════════════════════════════
#  Smart sizing: fit image into 320×120 native grid
# ═══════════════════════════════════════════════════════════════

def fit_image_to_native(img: Image.Image) -> Tuple[int, int]:
    """Return (w_cells, h_cells) that fills 320×120 while preserving
    aspect ratio, or crops to fill exactly."""
    iw, ih = img.size
    iw_c = iw // CELL_W
    ih_c = ih // CELL_H
    if iw_c <= 0 or ih_c <= 0:
        return GRID_W, GRID_H

    # Crop to fill 320×120 (maintain aspect ratio via crop, not letterbox)
    target_ratio = GRID_W / GRID_H
    src_ratio = iw_c / ih_c

    if src_ratio > target_ratio:
        # image is wider → crop sides
        new_w = int(ih_c * target_ratio)
        new_h = ih_c
    else:
        # image is taller → crop top/bottom
        new_w = iw_c
        new_h = int(iw_c / target_ratio)

    return (min(new_w, GRID_W), min(new_h, GRID_H))
```

----------------------------------------

### File: `bs_heartbeat.py`

**Path:** `./bs_heartbeat.py`
**Extension:** `.py`
**Size:** 8,693 bytes (8.49 KB)

```py
"""
BS_HEARTBEAT  -  Heartbeat / frequency layer for Entanglement.
                 Part of the BS-TOS-IPX Entanglement Layer.

The heartbeat is a timing, routing, and consensus signal.
Each node emits a sinusoidal pulse whose frequency derives from
the cognitive state (lambda, coherence, novelty, instability).

Symbolic bands:
  0.5-4 Hz    DELTA     deep void / sleep / passive listen
  4-8 Hz      THETA     dream / merge / ghost
  7.83 Hz     SCHUMANN  base swarm pulse
  8-13 Hz     ALPHA     stable perception / consensus
  13-30 Hz    BETA      active scan / routing / switching
  30-40 Hz    GAMMA     high-integration / high-risk merge
  >40 Hz      CHAOS     force stabilization

Frequency formula:
  f_i = clamp(7.83 + 30*lambda + 4*coherence + 3*novelty - 6*instability, 1, 40)

Beat window consensus:
  if beat mod consensus_interval == 0: resolve_votes()
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

F_BASE = 7.83        # Schumann resonance base frequency
ALPHA_LAMBDA = 30.0  # weight for lambda
BETA_COHERENCE = 4.0 # weight for coherence
GAMMA_NOVELTY = 3.0  # weight for novelty
DELTA_INSTABILITY = 6.0  # weight for instability penalty

F_MIN = 1.0
F_MAX = 40.0

# Symbolic band boundaries
BAND_DELTA_MAX = 4.0
BAND_THETA_MAX = 8.0
BAND_ALPHA_MAX = 13.0
BAND_BETA_MAX = 30.0
BAND_GAMMA_MAX = 40.0

# Default consensus interval (in beats)
DEFAULT_CONSENSUS_INTERVAL = 8

# Phase-crossing threshold for packet emission
PHASE_CROSS_UP = 0.0


@dataclass
class HeartbeatState:
    """Full heartbeat state."""
    frequency_hz: float = F_BASE
    phase: float = 0.0
    amplitude: float = 1.0
    beat: int = 0
    band: str = "SCHUMANN"
    coherence: float = 0.5

    # Timing
    last_update: float = 0.0
    last_phase_cross: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Band classification
# ═══════════════════════════════════════════════════════════════

def classify_band(frequency_hz: float) -> str:
    """Map a heartbeat frequency to its symbolic band name."""
    f = frequency_hz
    if f > BAND_GAMMA_MAX:
        return "CHAOS"
    if f > BAND_BETA_MAX:
        return "GAMMA"
    if f > BAND_ALPHA_MAX:
        return "BETA"
    if f > BAND_THETA_MAX:
        return "ALPHA"
    if f > BAND_DELTA_MAX:
        return "THETA"
    return "DELTA"


def band_color(band: str) -> tuple:
    """Return an (R, G, B) colour for a heartbeat band."""
    colors = {
        "DELTA":   (40,  0,  80),    # deep purple
        "THETA":   (0,   80, 160),   # dream blue
        "SCHUMANN":(0,  160, 120),   # earth green
        "ALPHA":   (80, 200, 80),    # stable green
        "BETA":    (200, 200, 0),    # active yellow
        "GAMMA":   (255, 100, 0),    # integration orange
        "CHAOS":   (255, 0,   0),    # danger red
    }
    return colors.get(band, (128, 128, 128))


def band_description(band: str) -> str:
    """Return a short description of what the band represents."""
    descriptions = {
        "DELTA":    "deep void / sleep / passive listen",
        "THETA":    "dream / merge / ghost",
        "SCHUMANN": "base swarm resonance",
        "ALPHA":    "stable perception / consensus",
        "BETA":     "active scan / routing / switching",
        "GAMMA":    "high-integration / high-risk merge",
        "CHAOS":    "force stabilization",
    }
    return descriptions.get(band, "unknown")


# ═══════════════════════════════════════════════════════════════
#  Heartbeat Engine
# ═══════════════════════════════════════════════════════════════

class HeartbeatEngine:
    """Manages the heartbeat frequency, phase, and beat timing.

    The heartbeat is the pulse that drives:
      - packet timing (emit on phase crossings)
      - merge timing
      - consensus windows (beat mod interval)
      - visual pulse overlay
      - node mood display
    """

    def __init__(self):
        self.state = HeartbeatState()
        self.consensus_interval = DEFAULT_CONSENSUS_INTERVAL
        self._pending_cross = False  # True when phase crossed zero upward

    def compute_frequency(self, lam: float, coherence: float,
                          novelty: float, instability: float) -> float:
        """Compute the heartbeat frequency from cognitive metrics.

        f_i = clamp(7.83 + 30*lambda + 4*coherence + 3*novelty - 6*instability, 1, 40)
        """
        f = (F_BASE
             + ALPHA_LAMBDA * lam
             + BETA_COHERENCE * coherence
             + GAMMA_NOVELTY * novelty
             - DELTA_INSTABILITY * max(0.0, instability))
        return max(F_MIN, min(F_MAX, f))

    def tick(self, dt: float, lam: float, coherence: float,
             novelty: float) -> bool:
        """Advance the heartbeat by dt seconds.

        Returns True if a beat occurred (phase wrapped past 1.0).
        Also returns True on initial tick to establish baseline.
        """
        # Compute instability from lambda exceeding critical band
        instability = max(0.0, lam - 0.1)

        # Update frequency
        self.state.frequency_hz = self.compute_frequency(
            lam, coherence, novelty, instability)

        # Update band
        self.state.band = classify_band(self.state.frequency_hz)

        # Store coherence for packet broadcasting
        self.state.coherence = coherence

        # Advance phase
        old_phase = self.state.phase
        self.state.phase = (self.state.phase + dt * self.state.frequency_hz) % 1.0
        self.state.last_update = time.time()

        # Detect upward zero-crossing
        self._pending_cross = False
        if self.state.phase < old_phase:
            # Phase wrapped around — a beat occurred
            self.state.beat += 1
            self._pending_cross = True
            return True

        return False

    def tick_void(self, dt: float):
        """Minimal heartbeat for VOID state: 1 Hz, low pulse."""
        self.state.frequency_hz = 1.0
        self.state.band = "DELTA"
        old_phase = self.state.phase
        self.state.phase = (self.state.phase + dt * 1.0) % 1.0
        if self.state.phase < old_phase:
            self.state.beat += 1

    def pulse_value(self) -> float:
        """Return the current heartbeat sine wave value [-1, 1]."""
        return self.state.amplitude * math.sin(
            2.0 * math.pi * self.state.phase)

    def pulse_01(self) -> float:
        """Return heartbeat value mapped to [0, 1]."""
        return (self.pulse_value() + 1.0) / 2.0

    def is_consensus_window(self) -> bool:
        """Check if current beat is a consensus resolution window."""
        return (self.state.beat > 0
                and self.state.beat % self.consensus_interval == 0)

    def is_broadcast_window(self) -> bool:
        """Check if it is time to emit a heartbeat packet.

        Emit on upward phase crossing (beat boundary).
        """
        return self._pending_cross

    def should_resolve_votes(self) -> bool:
        """Consensus only resolves on heartbeat windows."""
        return self.is_consensus_window()

    def to_dict(self) -> dict:
        """Serialize heartbeat state for packet broadcast."""
        return {
            "frequency_hz": round(self.state.frequency_hz, 3),
            "phase": round(self.state.phase, 4),
            "amplitude": round(self.state.amplitude, 3),
            "beat": self.state.beat,
            "band": self.state.band,
            "coherence": round(self.state.coherence, 3),
        }

    def one_line(self) -> str:
        """Compact HUD string."""
        return (f"{self.state.frequency_hz:.1f}Hz "
                f"beat={self.state.beat} "
                f"phase={self.state.phase:.2f} "
                f"[{self.state.band}]")
```

----------------------------------------

### File: `bs_identity.py`

**Path:** `./bs_identity.py`
**Extension:** `.py`
**Size:** 2,342 bytes (2.29 KB)

```py
"""
BS_IDENTITY  –  Node identity, session management, and local configuration.
                Part of the QIPX distributed visual cognition mesh.

Each BrailleStream node gets a unique, stable identity for the session.
v0.1 uses random UUIDs; v0.2 will add Ed25519 key pairs.
"""

from __future__ import annotations
import uuid
import time
import hashlib
import os
import socket


def generate_node_id() -> str:
    """Create a short unique node identifier: 'bs-' + 8 hex chars."""
    return "bs-" + uuid.uuid4().hex[:8]


def generate_session_id(node_id: str) -> str:
    """Create a session hash from node_id + startup time + random nonce."""
    nonce = uuid.uuid4().hex[:16]
    raw = f"{node_id}:{time.time()}:{nonce}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def local_ip() -> str:
    """Best-effort detection of the local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def local_hostname() -> str:
    """Return the machine hostname."""
    return os.uname().nodename if hasattr(os, 'uname') else "unknown"


def default_node_name() -> str:
    """Generate a human-readable default name for this node."""
    host = local_hostname().split(".")[0]
    return f"bs-{host}-{uuid.uuid4().hex[:4]}"


class NodeIdentity:
    """Immutable identity for a BrailleStream node session."""

    def __init__(self, name: str | None = None):
        self.node_id: str = generate_node_id()
        self.session_id: str = generate_session_id(self.node_id)
        self.name: str = name or default_node_name()
        self.ip: str = local_ip()
        self.hostname: str = local_hostname()
        self.created_at: float = time.time()

    def __repr__(self) -> str:
        return f"NodeIdentity(id={self.node_id}, name={self.name}, ip={self.ip})"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "session_id": self.session_id,
            "name": self.name,
            "ip": self.ip,
            "hostname": self.hostname,
        }

    def fingerprint(self) -> str:
        """Short fingerprint for display: first 12 chars of session_id."""
        return self.session_id[:12]
```

----------------------------------------

### File: `bs_influence.py`

**Path:** `./bs_influence.py`
**Extension:** `.py`
**Size:** 12,518 bytes (12.22 KB)

```py
"""
BS_INFLUENCE  -  Remote control permissions and influence acceptance.
                 Part of the BS-TOS-IPX Entanglement Layer.

Entanglement allows direct influence only under controlled conditions.

Permission levels:
  NONE       no remote influence
  SUGGEST    remote packets become suggestions only
  SOFT       local Pazuzu accepts/rejects automatically
  HARD       trusted majority can switch controls
  DIVINE     all nodes act as one; dangerous/debug only

Influence targets:
  fold_w, palette_id, render_mode, ghost_w,
  scan_active, scan_speed, lock_flash,
  merge candidate, reality mode, heartbeat phase

Acceptance formula:
  AcceptInfluence = Trust(sender) * Confidence(sender) * MajoritySupport * LocalStability
  Accept if AcceptInfluence >= theta_influence (default 0.70)

Direct switch cooldown:
  switch_allowed if now - last_switch > switch_cooldown (default 0.5s)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs_pazuzu import LAMBDA_MIN, LAMBDA_MAX


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

THETA_INFLUENCE_DEFAULT = 0.70
SWITCH_COOLDOWN_DEFAULT = 0.50

# Influence permission levels (ordered by permissiveness)
PERM_NONE = 0
PERM_SUGGEST = 1
PERM_SOFT = 2
PERM_HARD = 3
PERM_DIVINE = 4

PERM_NAMES = {
    PERM_NONE:    "NONE",
    PERM_SUGGEST: "SUGGEST",
    PERM_SOFT:    "SOFT",
    PERM_HARD:    "HARD",
    PERM_DIVINE:  "DIVINE",
}

# Controls that can be influenced
INFLUENCE_CONTROLS = {
    "SET_FOLD":      "fold_w",
    "SET_GHOST":     "ghost_w",
    "SET_PALETTE":   "palette_id",
    "SET_MODE":      "render_mode",
    "TOGGLE_SCAN":   "scan_active",
    "LOCK":          "lock",
    "UNLOCK":        "unlock",
    "MERGE_SANDBOX": "merge_sandbox",
    "MERGE_ACCEPT":  "merge_accept",
    "MERGE_REJECT":  "merge_reject",
    "ROLLBACK":      "rollback",
    "HEARTBEAT_SHIFT": "heartbeat_shift",
}


# ═══════════════════════════════════════════════════════════════
#  Influence Record
# ═══════════════════════════════════════════════════════════════

@dataclass
class InfluenceRecord:
    """A record of a received or sent influence command."""
    influence_id: str
    from_node: str
    to_node: str
    scope: str
    control: str
    value: Any
    reason: str
    confidence: float
    accepted: bool
    applied: bool
    acceptance_score: float
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Influence Engine
# ═══════════════════════════════════════════════════════════════

class InfluenceEngine:
    """Manages remote influence permissions, acceptance, and cooldown.

    The influence engine sits between incoming ENTANGLE_INFLUENCE packets
    and the renderer's state.  It decides whether to apply or reject
    each influence command based on trust, confidence, majority support,
    and local stability.
    """

    def __init__(self, permission_level: int = PERM_SOFT,
                 theta_influence: float = THETA_INFLUENCE_DEFAULT,
                 switch_cooldown: float = SWITCH_COOLDOWN_DEFAULT):
        self.permission_level = permission_level
        self.theta_influence = theta_influence
        self.switch_cooldown = switch_cooldown

        # Tracking
        self._last_switch_time: Dict[str, float] = {}
        self._influence_history: List[InfluenceRecord] = []
        self._influence_counter = 0
        self._max_history = 200

        # Fold gravity state
        self.fold_gravity_mu = 0.05  # fold gravity pull factor

        # Suggestions queue (for SUGGEST mode)
        self._suggestions: List[InfluenceRecord] = []
        self._max_suggestions = 20

    # ── Permission management ─────────────────────────────────

    @property
    def permission_name(self) -> str:
        return PERM_NAMES.get(self.permission_level, "UNKNOWN")

    def set_permission(self, level: int):
        """Set the influence permission level."""
        self.permission_level = max(PERM_NONE, min(PERM_DIVINE, level))

    def can_influence(self, scope: str = "SOFT") -> bool:
        """Check if influence is allowed at the current permission level.

        Scope mapping:
          NONE → not allowed
          SUGGEST → only suggestion
          SOFT → automatic accept/reject
          HARD → majority switching
          DIVINE → all controls
        """
        if self.permission_level == PERM_NONE:
            return False
        if self.permission_level == PERM_SUGGEST:
            return True  # always accept as suggestion
        if self.permission_level >= PERM_SOFT:
            return True
        return False

    def is_direct_switch(self) -> bool:
        """Can the current permission level directly switch controls?"""
        return self.permission_level >= PERM_HARD

    # ── Influence evaluation ──────────────────────────────────

    def evaluate_influence(
        self,
        from_node: str,
        control: str,
        value: Any,
        confidence: float,
        trust: float,
        majority_support: float,
        local_stability: float,
        reason: str = "",
    ) -> Tuple[bool, float]:
        """Evaluate whether to accept an influence command.

        Returns (accepted, acceptance_score).
        """
        if self.permission_level == PERM_NONE:
            return False, 0.0

        # SUGGEST mode: always accept but mark as suggestion
        if self.permission_level == PERM_SUGGEST:
            score = trust * confidence * 0.5
            return True, score

        # SOFT/HARD/DIVINE: compute acceptance score
        acceptance = (trust * confidence * majority_support * local_stability)

        # Cooldown check for direct switch controls
        if control in ("SET_FOLD", "SET_PALETTE", "SET_MODE", "SET_GHOST"):
            last = self._last_switch_time.get(control, 0.0)
            if time.time() - last < self.switch_cooldown:
                return False, acceptance  # too soon

        return acceptance >= self.theta_influence, acceptance

    # ── Apply influence to renderer ────────────────────────────

    def apply_influence(
        self,
        renderer,
        control: str,
        value: Any,
        acceptance_score: float,
        is_suggestion: bool = False,
    ) -> bool:
        """Apply an accepted influence to the renderer.

        Returns True if the control was actually applied.
        """
        target_attr = INFLUENCE_CONTROLS.get(control)
        if target_attr is None:
            return False

        # Record the switch time for cooldown
        if control in ("SET_FOLD", "SET_PALETTE", "SET_MODE", "SET_GHOST"):
            self._last_switch_time[control] = time.time()

        if control == "SET_FOLD":
            if isinstance(value, (int, float)):
                new_w = max(1, min(320, int(value)))
                # Fold gravity: gentle pull, not snap
                if not self.is_direct_switch():
                    mu = self.fold_gravity_mu
                    current = renderer.fold_w
                    new_w = round(current + mu * (new_w - current))
                    new_w = max(1, min(320, new_w))
                if new_w != renderer.fold_w:
                    renderer.fold_w = new_w
                    renderer.needs_redraw = True
                    return True

        elif control == "SET_GHOST":
            if isinstance(value, (int, float)):
                new_gw = max(0, min(320, int(value)))
                renderer.ghost_w = new_gw
                renderer.show_ghost = True
                renderer.needs_redraw = True
                return True

        elif control == "SET_PALETTE":
            if isinstance(value, int) and 0 <= value <= 4:
                renderer.palette_id = value
                renderer.needs_redraw = True
                return True

        elif control == "SET_MODE":
            if isinstance(value, int) and 0 <= value <= 5:
                renderer.render_mode = value
                renderer.needs_redraw = True
                return True

        elif control == "TOGGLE_SCAN":
            renderer.scan_active = not renderer.scan_active
            if renderer.scan_active:
                renderer.scan_w = 10
            renderer.needs_redraw = True
            return True

        elif control == "LOCK":
            renderer.lock_flash = 30
            renderer.needs_redraw = True
            return True

        return False

    def apply_fold_gravity(self, renderer, consensus_fold: int) -> bool:
        """Gently pull the local fold toward the consensus value.

        W_local(t+1) = W_local(t) + mu * (W_consensus - W_local)

        Returns True if fold_w changed.
        """
        mu = self.fold_gravity_mu
        current = renderer.fold_w
        new_w = round(current + mu * (consensus_fold - current))
        new_w = max(1, min(320, new_w))
        if new_w != current:
            last = self._last_switch_time.get("SET_FOLD", 0.0)
            if time.time() - last >= self.switch_cooldown:
                renderer.fold_w = new_w
                renderer.needs_redraw = True
                self._last_switch_time["SET_FOLD"] = time.time()
                return True
        return False

    # ── Record management ─────────────────────────────────────

    def record_influence(
        self,
        from_node: str,
        to_node: str,
        scope: str,
        control: str,
        value: Any,
        reason: str,
        confidence: float,
        accepted: bool,
        applied: bool,
        acceptance_score: float,
    ) -> str:
        """Record an influence event and return its ID."""
        self._influence_counter += 1
        inf_id = f"inf-{self._influence_counter:05d}"
        record = InfluenceRecord(
            influence_id=inf_id,
            from_node=from_node,
            to_node=to_node,
            scope=scope,
            control=control,
            value=value,
            reason=reason,
            confidence=confidence,
            accepted=accepted,
            applied=applied,
            acceptance_score=acceptance_score,
            timestamp=time.time(),
        )
        self._influence_history.append(record)
        if len(self._influence_history) > self._max_history:
            self._influence_history = self._influence_history[-self._max_history:]
        return inf_id

    def add_suggestion(self, record: InfluenceRecord):
        """Add a suggestion to the suggestion queue."""
        self._suggestions.append(record)
        if len(self._suggestions) > self._max_suggestions:
            self._suggestions = self._suggestions[-self._max_suggestions:]

    def get_suggestions(self) -> List[InfluenceRecord]:
        """Return pending suggestions."""
        return list(self._suggestions)

    def clear_suggestions(self):
        self._suggestions.clear()

    def recent_influences(self, limit: int = 5) -> List[InfluenceRecord]:
        """Return the most recent influence records."""
        return self._influence_history[-limit:]

    def influence_count(self) -> int:
        return len(self._influence_history)

    def accepted_count(self) -> int:
        return sum(1 for r in self._influence_history if r.accepted)

    def rejected_count(self) -> int:
        return sum(1 for r in self._influence_history if not r.accepted)
```

----------------------------------------

### File: `bs_main.py`

**Path:** `./bs_main.py`
**Extension:** `.py`
**Size:** 21,740 bytes (21.23 KB)

```py
"""
BS_MAIN  –  Main application loop.
            Faithfully translates BS_MAIN.HC keyboard-driven control flow.
            Integrates QIPX networking and Pazuzu cognition.

Controls:
  A / D              decrease / increase fold width by 1
  SHIFT + A / D      decrease / increase fold width by 8
  LEFT / RIGHT       (aliases for A / D)
  UP / DOWN          jump ±10 in fold width
  1 – 9              jump to preset fold widths
  0                  reset to native 320
  TAB                next pattern (procedural demo)
  P                  cycle palette
  M                  cycle render mode
  G                  toggle ghost overlay
  H                  set ghost width to current / toggle
  SPACE              start/stop harmonic scan animation
  R                  resonance lock  (find best width and jump to it)
  I                  import image via file dialog
  Shift+E            export stream as HolyC .HC file
  E                  toggle Entanglement ON/OFF (requires QIPX)
  B                  export stream as raw .BIN file
  L                  load a .BIN file
  F                  save screenshot
  +/-                adjust scan speed

  Q                  toggle QIPX ON/OFF
  Shift+Q            hard reset QIPX peer table
  Ctrl+Q             toggle auto-merge mode
  Alt+Q              show QIPX diagnostics overlay
  N                  request stream from best-scoring peer
  J                  accept pending merge (sandbox preview)
  K                  reject pending merge
  Backspace          rollback to last stable stream
  ESC                quit

Render modes:
  0  Native Pixel Mode
  1  Fold Scan Mode
  2  Scaled Preview Mode
  3  Density Map Mode
  4  Resonance Analyzer
  5  Paradox Overlay
"""

from __future__ import annotations
import sys
import os
import time
import math

# ── Ensure the package directory is on sys.path ────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from bs_engine import MAX_CELLS, GRID_W, GRID_H, SCREEN_W, SCREEN_H
from bs_renderer import BrailleStreamRenderer, PALETTE_NAMES, PALETTES, _qipx_node as _qipx_ref
from bs_patterns import PATTERNS, plasma
from bs_export import image_to_masks, fit_image_to_native, \
    export_holyc, export_binary, import_binary

try:
    import pygame
except ImportError:
    print("pygame is required.  Install with:  pip install pygame")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not found. Image import (I key) will be disabled.")
    print("         Install with:  pip install Pillow")

# ── QIPX (optional – graceful fallback if import fails) ─────
try:
    from bs_qipx import QipxNode
    HAS_QIPX = True
except ImportError:
    HAS_QIPX = False
    print("WARNING: QIPX module not available. Networking disabled.")
    print("         This is non-critical; BrailleStream works standalone.")

# ── Entanglement (optional – requires QIPX) ──────────────────
try:
    from bs_entanglement import EntanglementController
    HAS_ENTANGLE = True
except ImportError:
    HAS_ENTANGLE = False
    print("WARNING: Entanglement module not available.")
    print("         Requires QIPX.  Entanglement disabled.")

# ── Civilization (optional – requires Entanglement) ──────────
try:
    from bs_civilization import CivilizationController
    HAS_CIV = True
except ImportError:
    HAS_CIV = False
    print("WARNING: Civilization module not available.")
    print("         Requires Entanglement.  Civilization disabled.")


# ═══════════════════════════════════════════════════════════════
#  Fold width presets  (from the TempleOS blueprint §7)
# ═══════════════════════════════════════════════════════════════

FOLD_PRESETS = {
    pygame.K_0: 320,   # native full screen
    pygame.K_1: 320,   # 320 × 120  native
    pygame.K_2: 240,   # 240 × 160  tall compression
    pygame.K_3: 192,   # 192 × 200  near-square-ish
    pygame.K_4: 160,   # 160 × 240  vertical harmonic
    pygame.K_5: 128,   # 128 × 300  tall fold
    pygame.K_6: 120,   # 120 × 320  tall fold
    pygame.K_7: 96,    # 96  × 400  long fold
    pygame.K_8: 80,    # 80  × 480  extreme fold
    pygame.K_9: 64,    # 64  × 600  overflow scan
}

RENDER_MODE_NAMES = {
    0: "Native Pixel",
    1: "Fold Scan",
    2: "Scaled Preview",
    3: "Density Map",
    4: "Resonance Analyzer",
    5: "Paradox Overlay",
}

PATTERN_NAMES = list(PATTERNS.keys())


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    r = BrailleStreamRenderer()

    # Load default procedural pattern
    pattern_idx = 0
    r.set_stream(plasma(length=MAX_CELLS, fold_w=GRID_W))

    # Scan animation state
    scan_speed = 2        # cells per frame
    scan_dir = 1          # +1 forward, -1 backward

    # Ghost overlay
    r.ghost_w = GRID_W // 2

    # File paths
    export_dir = _SCRIPT_DIR

    # ── QIPX initialisation ──────────────────────────────────
    qipx = None
    entangle = None
    civ = None
    if HAS_QIPX:
        qipx = QipxNode(renderer=r, enabled=False)
        qipx.start()
        # Wire the QIPX node into the renderer's HUD system
        import bs_renderer
        bs_renderer._qipx_node = qipx

        # ── Entanglement initialisation ────────────────────────
        if HAS_ENTANGLE:
            entangle = EntanglementController(
                renderer=r, qipx=qipx,
                crystal_path=os.path.join(_SCRIPT_DIR,
                    "BS_ENTANGLEMENT_CRYSTAL.json"),
            )
            qipx.set_entanglement(entangle)
            # Wire entanglement into renderer HUD
            bs_renderer._entangle_ctrl = entangle

            # ── Civilization initialisation ────────────────────
            if HAS_CIV:
                civ = CivilizationController(
                    entangle=entangle,
                    crystal_path=os.path.join(_SCRIPT_DIR,
                        "BS_QIPX_CIVILIZATION_CRYSTAL.json"),
                )
                entangle.civilization = civ
                # Wire civilization into renderer HUD
                bs_renderer._civ_ctrl = civ

    # ── main loop ────────────────────────────────────────────
    while r.running:
        # ── events ───────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                r.running = False
                break

            if event.type != pygame.KEYDOWN:
                continue

            key = event.key
            mods = pygame.key.get_mods()
            shift = bool(mods & pygame.KMOD_SHIFT)
            ctrl = bool(mods & pygame.KMOD_CTRL)
            alt = bool(mods & pygame.KMOD_ALT)

            # ── QUIT ──
            if key == pygame.K_ESCAPE:
                r.running = False

            # ── FOLD WIDTH ──
            elif key in (pygame.K_a, pygame.K_LEFT):
                r.fold_w -= 8 if shift else 1
                r.fold_w = max(1, min(GRID_W, r.fold_w))
                r.needs_redraw = True

            elif key in (pygame.K_d, pygame.K_RIGHT):
                r.fold_w += 8 if shift else 1
                r.fold_w = max(1, min(GRID_W, r.fold_w))
                r.needs_redraw = True

            elif key == pygame.K_UP:
                r.fold_w += 10
                r.fold_w = min(GRID_W, r.fold_w)
                r.needs_redraw = True

            elif key == pygame.K_DOWN:
                r.fold_w -= 10
                r.fold_w = max(1, r.fold_w)
                r.needs_redraw = True

            # ── PRESETS ──
            elif key in FOLD_PRESETS:
                r.fold_w = FOLD_PRESETS[key]
                r.needs_redraw = True

            # ── PATTERN CYCLE ──
            elif key == pygame.K_TAB:
                pattern_idx = (pattern_idx + 1) % len(PATTERN_NAMES)
                name = PATTERN_NAMES[pattern_idx]
                r.set_stream(PATTERNS[name]())
                r.fold_w = GRID_W
                r.needs_redraw = True

            # ── PALETTE CYCLE ──
            elif key == pygame.K_p:
                r.palette_id = (r.palette_id + 1) % len(PALETTES)
                r.needs_redraw = True

            # ── RENDER MODE CYCLE ──
            elif key == pygame.K_m:
                r.render_mode = (r.render_mode + 1) % 6
                r.needs_redraw = True

            # ── GHOST OVERLAY ──
            elif key == pygame.K_g:
                r.show_ghost = not r.show_ghost
                if r.show_ghost and r.ghost_w == 0:
                    r.ghost_w = r.fold_w // 2
                r.needs_redraw = True

            elif key == pygame.K_h:
                r.ghost_w = r.fold_w // 2
                r.show_ghost = True
                r.needs_redraw = True

            # ── RESONANCE LOCK ──
            elif key == pygame.K_r:
                best = r.get_resonance(1)
                if best:
                    _, best_w, best_h = best[0]
                    r.fold_w = best_w
                    r.lock_flash = 30
                    r.needs_redraw = True

            # ── HARMONIC SCAN ──
            elif key == pygame.K_SPACE:
                r.scan_active = not r.scan_active
                if r.scan_active:
                    r.scan_w = 10
                    scan_dir = 1

            # ── SCAN SPEED ──
            elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                scan_speed = min(20, scan_speed + 1)

            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                scan_speed = max(1, scan_speed - 1)

            # ── IMPORT IMAGE ──
            elif key == pygame.K_i:
                if not HAS_PIL:
                    print("Pillow not installed – image import disabled.")
                    continue
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    path = filedialog.askopenfilename(
                        title="Import Image",
                        filetypes=[("Image files",
                                    "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                                   ("All files", "*.*")])
                    root.destroy()
                    if not path:
                        continue
                    img = Image.open(path)
                    wc, hc = fit_image_to_native(img)
                    stream = image_to_masks(img, wc, hc,
                                            threshold=128,
                                            gamma=1.0)
                    r.set_stream(stream)
                    r.fold_w = wc
                    r.needs_redraw = True
                except Exception as e:
                    print(f"Import failed: {e}")

            # ── EXPORT HolyC (Shift+E) ──
            elif key == pygame.K_e and shift:
                path = export_holyc(r.stream,
                                    os.path.join(export_dir, "BS_DATA.HC"))
                print(f"Exported HolyC data → {path}")

            # ══════════════════════════════════════════════════
            #  ENTANGLEMENT CONTROLS
            # ══════════════════════════════════════════════════

            elif key == pygame.K_e and entangle is not None:
                # E → toggle Entanglement ON/OFF (VOID if no QIPX)
                entangle.toggle_or_void()
                mode = entangle.state.mode
                print(f"ENTANGLE: {mode}")
                if entangle.state.void:
                    print("  VOID -- enable QIPX first with Q")
                r.needs_redraw = True

            # ── EXPORT Binary ──
            elif key == pygame.K_b:
                path = export_binary(r.stream,
                                     os.path.join(export_dir, "BS_DATA.BIN"))
                print(f"Exported binary → {path}")

            # ── LOAD Binary ──
            elif key == pygame.K_l:
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    root = tk.Tk()
                    root.withdraw()
                    path = filedialog.askopenfilename(
                        title="Load Binary Stream",
                        filetypes=[("Binary files", "*.bin"),
                                   ("All files", "*.*")])
                    root.destroy()
                    if path:
                        r.set_stream(import_binary(path))
                        r.fold_w = GRID_W
                        r.needs_redraw = True
                except Exception as e:
                    print(f"Load failed: {e}")

            # ── SCREENSHOT ──
            elif key == pygame.K_f:
                ss_path = os.path.join(export_dir,
                                       f"BS_SCREEN_{int(time.time())}.png")
                pygame.image.save(r.screen, ss_path)
                print(f"Screenshot → {ss_path}")

            # ══════════════════════════════════════════════════
            #  QIPX CONTROLS
            # ══════════════════════════════════════════════════

            elif key == pygame.K_q and qipx is not None:
                if alt:
                    # Alt+Q → toggle diagnostics overlay
                    qipx.show_diagnostics = not qipx.show_diagnostics
                    r.needs_redraw = True
                elif ctrl:
                    # Ctrl+Q → toggle auto-merge
                    qipx.auto_merge = not qipx.auto_merge
                    status = "ON" if qipx.auto_merge else "OFF"
                    print(f"QIPX auto-merge: {status}")
                elif shift:
                    # Shift+Q → hard reset
                    qipx.hard_reset()
                    print("QIPX: peer table reset")
                    r.needs_redraw = True
                else:
                    # Q → toggle QIPX
                    qipx.toggle()
                    status = "ON" if qipx.enabled else "OFF"
                    print(f"QIPX: {status}  "
                          f"node={qipx.identity.node_id}  "
                          f"ip={qipx.identity.ip}")
                    r.needs_redraw = True

            elif key == pygame.K_n and qipx is not None and qipx.enabled:
                # N → request stream from best peer
                best = qipx.peers.best_scoring_peer()
                if best:
                    qipx.request_stream(best.node_id)
                    print(f"QIPX: requesting stream from {best.node_id}")
                else:
                    print("QIPX: no peers available")

            elif key == pygame.K_j and qipx is not None and qipx.enabled:
                # J → accept pending merge
                qipx.accept_merge()
                print(f"QIPX: merge accepted  "
                      f"reality={qipx.reality_state}")
                r.needs_redraw = True

            elif key == pygame.K_k and qipx is not None and qipx.enabled:
                # K → reject pending merge
                qipx.reject_merge()
                print("QIPX: merge rejected")
                r.needs_redraw = True

            elif key == pygame.K_BACKSPACE and qipx is not None:
                # Backspace → rollback
                qipx.rollback()
                print("QIPX: rollback")
                r.needs_redraw = True

            # ══════════════════════════════════════════════════
            #  CIVILIZATION CONTROLS
            # ══════════════════════════════════════════════════

            elif key == pygame.K_c and civ is not None and entangle is not None:
                if shift:
                    # Shift+C → write crystal snapshot
                    civ._crystal_maybe_save()
                    print("CIV: crystal snapshot written")
                else:
                    # C → toggle civilization overlay
                    if civ.state.enabled:
                        civ.state.show_overlay = not civ.state.show_overlay
                        r.needs_redraw = True
                        print(f"CIV: overlay {'ON' if civ.state.show_overlay else 'OFF'}")
                    else:
                        # Toggle civilization ON/OFF
                        result = civ.toggle()
                        status = "ON" if result else "OFF"
                        print(f"CIV: {status}  "
                              f"archetype={civ.state.local_archetype}  "
                              f"civ_id={civ.state.civ_id}")
                        r.needs_redraw = True

            elif key == pygame.K_v and civ is not None and civ.state.enabled:
                if shift:
                    # Shift+V → branch current local reality
                    branch_id = f"branch-manual-{int(time.time())}"
                    civ.branches[branch_id] = {
                        "branch_id": branch_id,
                        "parent_reality_hash": civ.state.reality_hash,
                        "minority_nodes": [qipx.identity.node_id],
                        "reason": "manual_branch",
                        "fold_w": r.fold_w,
                        "support": 1.0,
                        "created_at": time.time(),
                        "preserved": True,
                    }
                    print(f"CIV: manual branch created  W={r.fold_w}")
                    r.needs_redraw = True
                else:
                    # V → vote for current local reality
                    civ._cast_reality_vote()
                    print(f"CIV: voted for local reality  W={r.fold_w}")
                    r.needs_redraw = True

            elif key == pygame.K_o and civ is not None and civ.state.enabled:
                # O → manual OMEGA rebirth (debug)
                if shift:
                    print("CIV: manual OMEGA rebirth triggered")
                    civ._omega_rebirth("MANUAL_DEBUG")
                    r.needs_redraw = True
                else:
                    print(f"CIV: OMEGA rebirth (use Shift+O to trigger)")
                    print(f"  paradox={civ.state.paradox_pressure:.3f}  "
                          f"(limit={0.80})")
                    print(f"  branches={len(civ.branches)}  "
                          f"laws={len(civ.laws)}  "
                          f"gen={civ.state.generation}")

        # ── scan animation ───────────────────────────────────
        if r.scan_active:
            r.scan_w += scan_speed * scan_dir
            if r.scan_w >= GRID_W:
                scan_dir = -1
            elif r.scan_w <= 10:
                scan_dir = 1
            r.fold_w = int(r.scan_w)
            r.needs_redraw = True

            # Auto-lock on perfect divisors
            if len(r.stream) % r.fold_w == 0:
                r.lock_flash = 10

        # ── QIPX tick ────────────────────────────────────────
        if qipx is not None:
            qipx.tick()

        # ── Entanglement tick ─────────────────────────────────
        if entangle is not None:
            entangle.tick()

        # ── draw ─────────────────────────────────────────────
        if r.needs_redraw or r.scan_active or r.lock_flash > 0:
            r.draw()

        r.clock.tick(60)

    # ── cleanup ──────────────────────────────────────────────
    if civ is not None:
        civ.shutdown()
    if entangle is not None:
        entangle.crystal.flush()
    if qipx is not None:
        qipx.stop()
    pygame.quit()


if __name__ == '__main__':
    main()
```

----------------------------------------

### File: `bs_patterns.py`

**Path:** `./bs_patterns.py`
**Extension:** `.py`
**Size:** 9,319 bytes (9.10 KB)

```py
"""
BS_PATTERNS  –  Procedural demo stream generators.
                Faithfully translates BS_DEMO.HC test patterns.

Each function fills a bytearray of `length` cells (default MAX_CELLS=38400)
folded at `fold_w` columns (default 320).

Patterns:
  diagonal       – 45-degree line across the grid
  checkerboard   – alternating 0x00 / 0xFF
  circle         – circular mask
  vbars          – vertical bars of varying widths
  hbars          – horizontal bars
  gradient       – left-to-right density gradient
  resonance_grid – grid lines at harmonic intervals
  noise          – pseudo-random noise
  face           – simple 8-bit smiley face
  cross          – centred cross / plus
  sierpinski     – Sierpinski-triangle fractal
  waves          – sinusoidal waves
  mandelbrot     – tiny Mandelbrot zoom
  text           – spelling out "BS" in large Braille glyphs
"""

from __future__ import annotations
import math
import random
from bs_engine import MAX_CELLS, GRID_W, CELL_W, CELL_H, bs_mask_from_dots


def _make(length: int, fold_w: int,
          cell_fn) -> bytearray:
    """Utility: apply cell_fn(i, cx, cy) → mask for every cell."""
    buf = bytearray(length)
    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        buf[i] = cell_fn(i, cx, cy) & 0xFF
    return buf


def _circle_mask(cx: int, cy: int,
                 centre_x: float, centre_y: float,
                 radius: float) -> int:
    """Return 0xFF if cell centre is inside circle, else 0x00."""
    if (cx - centre_x) ** 2 + (cy - centre_y) ** 2 <= radius * radius:
        return 0xFF
    return 0x00


# ═══════════════════════════════════════════════════════════════

def diagonal(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Diagonal line from top-left to bottom-right."""
    def fn(i, cx, cy):
        return 0xFF if cx == cy * 2 else 0x00
    return _make(length, fold_w, fn)


def checkerboard(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Alternating full / empty cells."""
    def fn(i, cx, cy):
        return 0xFF if (cx + cy) % 2 == 0 else 0x00
    return _make(length, fold_w, fn)


def circle(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Circle centred in the grid."""
    h = length // fold_w
    cr_x, cr_y, radius = fold_w / 2, h / 2, min(fold_w, h) * 0.4
    def fn(i, cx, cy):
        return _circle_mask(cx, cy, cr_x, cr_y, radius)
    return _make(length, fold_w, fn)


def vbars(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Vertical bars at every 8th column."""
    def fn(i, cx, cy):
        return 0xFF if cx % 8 < 4 else 0x00
    return _make(length, fold_w, fn)


def hbars(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Horizontal bars at every 4th row."""
    def fn(i, cx, cy):
        return 0xFF if cy % 4 < 2 else 0x00
    return _make(length, fold_w, fn)


def gradient(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Left-to-right density gradient  (0 dots → 8 dots)."""
    def fn(i, cx, cy):
        frac = cx / max(fold_w - 1, 1)
        d = int(frac * 8 + 0.5)
        d = max(0, min(8, d))
        if d == 0:
            return 0x00
        if d == 8:
            return 0xFF
        # build mask with exactly d random-ish dots
        dots = set()
        for bit in range(8):
            if bit < d:
                dots.add((bit // CELL_W, bit % CELL_W))
        return bs_mask_from_dots(dots)
    return _make(length, fold_w, fn)


def resonance_grid(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Grid lines at intervals of 10, 20, 40, 80 columns."""
    def fn(i, cx, cy):
        if cx % 80 == 0:
            return 0xFF
        if cy % 20 == 0:
            return 0xAA   # alternating dots
        if cx % 20 == 0:
            return 0x55   # complementary alternating
        return 0x00
    return _make(length, fold_w, fn)


def noise(length: int = MAX_CELLS, fold_w: int = GRID_W,
          seed: int = 42) -> bytearray:
    """Pseudo-random noise at ~50 % fill."""
    rng = random.Random(seed)
    buf = bytearray(length)
    for i in range(length):
        buf[i] = rng.getrandbits(8)
    return buf


def face(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Simple smiley face: circle + eyes + mouth."""
    h = length // fold_w
    cr_x, cr_y = fold_w / 2, h / 2
    R = min(fold_w, h) * 0.35
    buf = bytearray(length)
    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        dx, dy = cx - cr_x, cy - cr_y
        dist = math.sqrt(dx * dx + dy * dy)

        # Outer circle
        if abs(dist - R) < 1.5:
            buf[i] = 0xFF
        # Left eye
        elif (dx + R * 0.3) ** 2 + (dy - R * 0.3) ** 2 < (R * 0.12) ** 2:
            buf[i] = 0xFF
        # Right eye
        elif (dx - R * 0.3) ** 2 + (dy - R * 0.3) ** 2 < (R * 0.12) ** 2:
            buf[i] = 0xFF
        # Smile (arc)
        elif abs(dy + R * 0.15) < 2 and abs(dx) < R * 0.5 and dy > 0:
            buf[i] = 0xFF
        else:
            buf[i] = 0x00
    return buf


def cross(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Centred plus/cross shape."""
    h = length // fold_w
    cr_x, cr_y = fold_w // 2, h // 2
    arm_w, arm_h = max(1, fold_w // 10), max(1, h // 10)

    def fn(i, cx, cy):
        if abs(cx - cr_x) <= arm_w or abs(cy - cr_y) <= arm_h:
            return 0xFF
        return 0x00
    return _make(length, fold_w, fn)


def sierpinski(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Sierpinski triangle via bitwise AND trick."""
    buf = bytearray(length)
    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        # Sierpinski condition
        if (cx & cy) == 0:
            buf[i] = 0xFF
        else:
            buf[i] = 0x00
    return buf


def waves(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Sinusoidal wave bands."""
    buf = bytearray(length)
    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        val = math.sin(cx * 0.15) + math.cos(cy * 0.1)
        if val > 0.5:
            buf[i] = 0xFF
        elif val > 0:
            buf[i] = 0xAA
        elif val > -0.5:
            buf[i] = 0x55
        else:
            buf[i] = 0x00
    return buf


def mandelbrot(length: int = MAX_CELLS, fold_w: int = GRID_W,
               max_iter: int = 24) -> bytearray:
    """Tiny Mandelbrot set rendered into the Braille grid."""
    h = length // fold_w
    x_min, x_max = -2.2, 0.8
    y_min, y_max = -1.2, 1.2
    buf = bytearray(length)

    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        x0 = x_min + (cx / max(fold_w - 1, 1)) * (x_max - x_min)
        y0 = y_min + (cy / max(h - 1, 1)) * (y_max - y_min)
        x, y = 0.0, 0.0
        it = 0
        while x * x + y * y <= 4.0 and it < max_iter:
            x, y = x * x - y * y + x0, 2 * x * y + y0
            it += 1
        # Map iteration count to density
        d = int((it / max_iter) * 8 + 0.5)
        d = max(0, min(8, d))
        if d == 0:
            buf[i] = 0x00
        elif d == 8:
            buf[i] = 0xFF
        else:
            dots = set()
            for bit in range(8):
                if bit < d:
                    dots.add((bit // CELL_W, bit % CELL_W))
            buf[i] = bs_mask_from_dots(dots)
    return buf


def plasma(length: int = MAX_CELLS, fold_w: int = GRID_W) -> bytearray:
    """Classic plasma demo effect using sine interference."""
    buf = bytearray(length)
    for i in range(length):
        cx = i % fold_w
        cy = i // fold_w
        v1 = math.sin(cx * 0.06)
        v2 = math.sin(cy * 0.08)
        v3 = math.sin((cx + cy) * 0.05)
        v4 = math.sin(math.sqrt(cx * cx + cy * cy) * 0.04)
        v = (v1 + v2 + v3 + v4 + 4) / 8.0   # normalise to 0..1
        d = int(v * 8 + 0.5)
        d = max(0, min(8, d))
        if d == 0:
            buf[i] = 0x00
        elif d == 8:
            buf[i] = 0xFF
        else:
            dots = set()
            for bit in range(8):
                if bit < d:
                    dots.add((bit // CELL_W, bit % CELL_W))
            buf[i] = bs_mask_from_dots(dots)
    return buf


# ═══════════════════════════════════════════════════════════════
#  Pattern registry  (name → callable)
# ═══════════════════════════════════════════════════════════════

PATTERNS = {
    "diagonal":       diagonal,
    "checkerboard":   checkerboard,
    "circle":         circle,
    "vbars":          vbars,
    "hbars":          hbars,
    "gradient":       gradient,
    "resonance_grid": resonance_grid,
    "noise":          noise,
    "face":           face,
    "cross":          cross,
    "sierpinski":     sierpinski,
    "waves":          waves,
    "mandelbrot":     mandelbrot,
    "plasma":         plasma,
}
```

----------------------------------------

### File: `bs_pazuzu.py`

**Path:** `./bs_pazuzu.py`
**Extension:** `.py`
**Size:** 13,581 bytes (13.26 KB)

```py
"""
BS_PAZUZU  –  Pazuzu Cognition layer for BrailleStream.
              Part of the QIPX distributed visual cognition mesh.

A lightweight adaptive control loop that gives each node cognition-like
state: perception (stream stats), attention (lock/fold selection),
memory (history), and action (scan/merge/lock/stabilize decisions).

This is NOT literal consciousness.  It is a functional cognitive
architecture metaphor: a self-regulating perceptual agent.

Core concept:
  lambda_dom = proxy instability metric
  target: lambda_min < |lambda_dom| < lambda_max
  inside band  = creatively alive
  below band   = too static  → EXPLORE
  above band   = too chaotic → STABILIZE
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bs_engine import (
    MAX_CELLS, GRID_W, stream_stats, resonance_score, bs_density,
)


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

LAMBDA_MIN = 1e-3      # below = dead/static
LAMBDA_MAX = 1e-1      # above = chaotic/noisy
ENTROPY_TARGET = 5.0   # ideal byte-entropy for visual richness
COHERENCE_HIGH = 0.85  # threshold for LOCK action
COHERENCE_LOW = 0.65   # threshold for parity flip

# Weights for lambda_dom proxy
W_ENTROPY = 0.30
W_SCORE = 0.25
W_PEER = 0.20
W_MERGE = 0.15
W_SCAN = 0.05
W_STREAM = 0.05

HISTORY_LEN = 64       # rolling window for metrics


# ═══════════════════════════════════════════════════════════════
#  State dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class PazuzuMetrics:
    """The five core cognition metrics."""
    novelty: float = 0.0
    entropic_potential: float = 0.5
    elegance: float = 0.5
    coherence: float = 0.5
    criticality_index: float = 0.5


@dataclass
class PazuzuState:
    """Full mind state of a node."""
    lambda_dom: float = 0.01
    coherence: float = 0.5
    novelty: float = 0.0
    entropic_potential: float = 0.5
    elegance: float = 0.5
    criticality_index: float = 0.5
    parity: int = 1           # +1 = exploitation, -1 = exploration
    mood: str = "WATCHING"
    action: str = "QUIET"
    metrics: PazuzuMetrics = field(default_factory=PazuzuMetrics)
    step_count: int = 0
    last_action_time: float = 0.0


@dataclass
class HistoryEntry:
    entropy: float = 0.0
    score: float = 0.0
    unique_masks: int = 0
    fold_w: int = 320
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Pazuzu Cognition Engine
# ═══════════════════════════════════════════════════════════════

class BraillePazuzu:
    """Adaptive criticality controller for a BrailleStream node.

    Each call to step() reads the renderer state, computes a lambda_dom
    instability proxy, updates the five core metrics, and decides on
    an action based on critical-band logic.
    """

    def __init__(self, lambda_min: float = LAMBDA_MIN,
                 lambda_max: float = LAMBDA_MAX):
        self.state = PazuzuState()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # History ring buffer
        self.history: List[HistoryEntry] = []
        self._prev_entropy: Optional[float] = None
        self._prev_score: Optional[float] = None
        self._prev_stream_hash: Optional[str] = None

    # ── Main step ────────────────────────────────────────────

    def step(self, renderer, peers: Dict) -> PazuzuState:
        """Run one cognition cycle.  Updates and returns self.state.

        Parameters
        ----------
        renderer : BrailleStreamRenderer
            The local renderer (reads stream, fold_w, score, etc.)
        peers : dict
            Peer table from QipxNode (node_id → PeerState)
        """
        stats = renderer.stream_stats()
        entropy = stats["entropy"]
        score = renderer.current_score()
        now = time.time()

        # ── Lambda proxy ─────────────────────────────────────
        d_entropy = abs(entropy - (self._prev_entropy or entropy))
        d_score = abs(score - (self._prev_score or score))
        peer_dis = self._peer_disagreement(renderer, peers)
        merge_dis = self._merge_instability()
        scan_vel = 1.0 if renderer.scan_active else 0.0
        stream_mut = self._stream_mutation(renderer)

        lam = (
            W_ENTROPY * min(1.0, d_entropy)
            + W_SCORE * min(1.0, d_score / 10.0)
            + W_PEER * peer_dis
            + W_MERGE * merge_dis
            + W_SCAN * scan_vel
            + W_STREAM * stream_mut
        )
        self.state.lambda_dom = lam

        # ── Coherence ────────────────────────────────────────
        fold_coh = 1.0 / (1.0 + score)
        peer_coh = 1.0 - peer_dis
        self.state.coherence = 0.5 * fold_coh + 0.5 * peer_coh

        # ── Novelty ──────────────────────────────────────────
        self.state.novelty = stream_mut

        # ── Entropic potential ───────────────────────────────
        ep = 1.0 - abs(entropy - ENTROPY_TARGET) / ENTROPY_TARGET
        self.state.entropic_potential = max(0.0, min(1.0, ep))

        # ── Elegance ─────────────────────────────────────────
        self.state.elegance = 1.0 / (1.0 + score)

        # ── Criticality index ────────────────────────────────
        self.state.criticality_index = self._criticality(lam)

        # ── Metrics bundle ───────────────────────────────────
        self.state.metrics = PazuzuMetrics(
            novelty=self.state.novelty,
            entropic_potential=self.state.entropic_potential,
            elegance=self.state.elegance,
            coherence=self.state.coherence,
            criticality_index=self.state.criticality_index,
        )

        # ── Parity gate ──────────────────────────────────────
        if self.state.coherence > COHERENCE_HIGH:
            self.state.parity = +1
        elif self.state.coherence < COHERENCE_LOW:
            self.state.parity = -1
        # else: keep current parity

        # ── Action selection ─────────────────────────────────
        self.state.action = self._choose_action()
        self.state.mood = self._mood()

        # ── Update history ───────────────────────────────────
        self._push_history(entropy, score, stats["unique_masks"],
                           renderer.fold_w, now)

        # ── Remember previous values ─────────────────────────
        self._prev_entropy = entropy
        self._prev_score = score
        import hashlib
        self._prev_stream_hash = hashlib.sha256(
            bytes(renderer.stream)).hexdigest()[:12]

        self.state.step_count += 1
        self.state.last_action_time = now

        return self.state

    # ── Internal metric helpers ──────────────────────────────

    def _peer_disagreement(self, renderer, peers: Dict) -> float:
        """Average normalised projection disagreement with peers."""
        if not peers:
            return 0.0
        local_w = renderer.fold_w
        diffs = []
        for p in peers.values():
            rw = p.fold_w
            if rw > 0:
                diffs.append(abs(local_w - rw) / GRID_W)
        return min(1.0, sum(diffs) / max(1, len(diffs))) if diffs else 0.0

    def _merge_instability(self) -> float:
        """Placeholder for merge-induced instability metric."""
        # In a full implementation this would track recent merge
        # bit-distance spikes.  For v0.1, return 0.
        return 0.0

    def _stream_mutation(self, renderer) -> float:
        """How different the current stream is from last seen."""
        import hashlib
        current_hash = hashlib.sha256(bytes(renderer.stream)).hexdigest()[:12]
        if self._prev_stream_hash is None:
            return 0.0
        if current_hash == self._prev_stream_hash:
            return 0.0
        # Heuristic: if hash changed, assume moderate mutation
        return 0.15

    def _criticality(self, lam: float) -> float:
        """How close lambda is to the critical band (0..1)."""
        abs_lam = abs(lam)
        if self.lambda_min <= abs_lam <= self.lambda_max:
            return 1.0
        if abs_lam < self.lambda_min:
            return abs_lam / self.lambda_min
        return max(0.0, 1.0 - (abs_lam - self.lambda_max))

    def _choose_action(self) -> str:
        """Select action based on lambda + coherence."""
        abs_lam = abs(self.state.lambda_dom)
        if abs_lam < self.lambda_min:
            return "EXPLORE"
        if abs_lam > self.lambda_max:
            return "STABILIZE"
        if self.state.coherence > COHERENCE_HIGH:
            return "LOCK"
        if self.state.entropic_potential > 0.75:
            return "MERGE"
        return "OBSERVE"

    def _mood(self) -> str:
        """Map action to mood string."""
        action = self.state.action
        return {
            "EXPLORE":    "HUNGRY",
            "STABILIZE":  "BOUNDARY",
            "LOCK":       "FOCUSED",
            "MERGE":      "DREAMING",
            "OBSERVE":    "WATCHING",
            "GHOST":      "RECEIVING",
            "BROADCAST":  "SHARING",
            "ROLLBACK":   "RECOVERING",
            "QUIET":      "SILENT",
        }.get(action, "SILENT")

    # ── History ──────────────────────────────────────────────

    def _push_history(self, entropy, score, unique, fold_w, ts):
        self.history.append(HistoryEntry(
            entropy=entropy, score=score,
            unique_masks=unique, fold_w=fold_w,
            timestamp=ts,
        ))
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

    def trend_entropy(self) -> float:
        """Slope of entropy over recent history (negative = falling)."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-10:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(h.entropy for h in recent) / n
        num = sum((i - x_mean) * (h.entropy - y_mean)
                  for i, h in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def trend_score(self) -> float:
        """Slope of resonance score (negative = improving)."""
        if len(self.history) < 3:
            return 0.0
        recent = self.history[-10:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(h.score for h in recent) / n
        num = sum((i - x_mean) * (h.score - y_mean)
                  for i, h in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize state for PAZUZU packet broadcast."""
        return {
            "lambda_dom": round(self.state.lambda_dom, 4),
            "coherence": round(self.state.coherence, 3),
            "novelty": round(self.state.novelty, 3),
            "entropic_potential": round(self.state.entropic_potential, 3),
            "elegance": round(self.state.elegance, 3),
            "criticality_index": round(self.state.criticality_index, 3),
            "parity": self.state.parity,
            "mood": self.state.mood,
            "action": self.state.action,
            "step": self.state.step_count,
        }

    def one_line(self) -> str:
        """Compact string for HUD display."""
        return (f"lambda={self.state.lambda_dom:.4f} "
                f"CI={self.state.criticality_index:.2f} "
                f"mood={self.state.mood} "
                f"action={self.state.action}")
```

----------------------------------------

### File: `bs_qipx.py`

**Path:** `./bs_qipx.py`
**Extension:** `.py`
**Size:** 24,390 bytes (23.82 KB)

```py
"""
BS_QIPX  –  Main QIPX node: LAN discovery, state exchange, stream
            transfer, ghost sync, merge coordination.
            Part of the QIPX distributed visual cognition mesh.

Architecture:
  UDP broadcast on 239.77.77.77:47777 for discovery + state
  Same socket for lightweight packet exchange
  Stream transfer via base64-chunked STATE packets

Keyboard integration (from bs_main.py):
  Q         toggle QIPX ON/OFF
  Shift+Q   hard reset peer table
  Ctrl+Q    toggle auto-merge mode
  Alt+Q     show QIPX diagnostics overlay
  J         accept pending merge (sandbox preview)
  K         reject pending merge
  N         request stream from best-scoring peer
"""

from __future__ import annotations
import socket
import threading
import time
import hashlib
import base64
import queue
import uuid
import math
from typing import Dict, Optional, Callable, List, Tuple

from bs_engine import MAX_CELLS, GRID_W, stream_stats, resonance_score
from bs_identity import NodeIdentity
from bs_qipx_packets import (
    QIPX_MAGIC, QIPX_VERSION, MAX_PACKET_SIZE,
    make_hello, make_state, make_lock, make_get_stream,
    make_stream, make_ghost, make_merge, make_pazuzu,
    encode, decode, decode_stream_payload,
)
from bs_qipx_peer import PeerTable, PeerState
from bs_qipx_merge import (
    MERGE_METHODS, apply_merge, safety_check,
    bit_distance, make_ledger_entry,
)
from bs_pazuzu import BraillePazuzu
from bs_entangle_packets import is_entangle_packet


# ═══════════════════════════════════════════════════════════════
#  Entanglement integration
# ═══════════════════════════════════════════════════════════════

# EntanglementController is wired in by bs_main.py
_entanglement_ctrl = None  # set externally


# ═══════════════════════════════════════════════════════════════
#  Network constants
# ═══════════════════════════════════════════════════════════════

DISCOVERY_PORT = 47777
DATA_PORT = 47778
MULTICAST_GROUP = "239.77.77.77"
BROADCAST_ADDR = "255.255.255.255"

HELLO_INTERVAL = 1.0       # seconds between HELLO broadcasts
STATE_INTERVAL = 0.20      # seconds between STATE broadcasts
PEER_EXPIRE_INTERVAL = 1.0 # seconds between expiry sweeps
STREAM_CHUNK_SIZE = 8192   # raw bytes per STREAM chunk


# ═══════════════════════════════════════════════════════════════
#  Reality states
# ═══════════════════════════════════════════════════════════════

REALITY_LOCAL = "LOCAL"
REALITY_GHOST = "GHOST"
REALITY_SANDBOX = "SANDBOX"
REALITY_MERGED = "MERGED"
REALITY_CONSENSUS = "CONSENSUS"


# ═══════════════════════════════════════════════════════════════
#  QipxNode
# ═══════════════════════════════════════════════════════════════

class QipxNode:
    """Full QIPX networking node for a BrailleStream instance.

    Integrates:
      - UDP discovery (broadcast + multicast)
      - State exchange
      - Stream transfer
      - Ghost overlay coordination
      - Merge engine
      - Pazuzu cognition
    """

    def __init__(self, renderer, name: str | None = None,
                 enabled: bool = False):
        self.renderer = renderer
        self.identity = NodeIdentity(name)
        self.enabled = enabled
        self.auto_merge = False
        self.running = False

        # Network
        self._sock: Optional[socket.socket] = None
        self._rx_thread: Optional[threading.Thread] = None
        self.inbox: queue.Queue = queue.Queue()

        # Timing
        self._last_hello = 0.0
        self._last_state = 0.0
        self._last_expire = 0.0
        self._last_pazuzu = 0.0
        self._started_at = 0.0

        # Peer + Pazuzu
        self.peers = PeerTable()
        self.pazuzu = BraillePazuzu()

        # Reality management
        self.reality_state = REALITY_LOCAL
        self._sandbox_stream: Optional[bytearray] = None
        self._pending_merge: Optional[dict] = None
        self._merge_history: List[dict] = []
        self._rollback_snapshot: Optional[bytearray] = None
        self._rollback_fold_w: int = GRID_W

        # Diagnostics overlay
        self.show_diagnostics = False

        # Callbacks (for renderer integration)
        self.on_peer_count_change: Optional[Callable] = None
        self.on_remote_ghost: Optional[Callable] = None
        self.on_merge_ready: Optional[Callable] = None

        # Entanglement controller (set by bs_main.py)
        self.entanglement = None

    def set_entanglement(self, ctrl):
        """Wire in the EntanglementController."""
        self.entanglement = ctrl

    # ── Lifecycle ────────────────────────────────────────────

    def start(self):
        """Initialise sockets and start receiver thread."""
        self._started_at = time.time()
        self.running = True

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(0.1)
            self._sock.bind(("", DISCOVERY_PORT))

            # Try multicast membership
            try:
                mreq = socket.inet_aton(MULTICAST_GROUP) + socket.inet_aton("0.0.0.0")
                self._sock.setsockopt(socket.IPPROTO_IP,
                                      socket.IP_ADD_MEMBERSHIP, mreq)
            except Exception:
                pass  # multicast optional

        except OSError as e:
            print(f"QIPX: socket bind failed: {e}")
            self.running = False
            return

        self._rx_thread = threading.Thread(target=self._rx_loop,
                                           daemon=True, name="qipx-rx")
        self._rx_thread.start()

    def stop(self):
        """Shut down sockets and threads."""
        self.running = False
        self.enabled = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def toggle(self):
        """Toggle QIPX enabled state."""
        self.enabled = not self.enabled
        if self.enabled:
            self.send_hello(force=True)

    def hard_reset(self):
        """Clear all peers and state."""
        self.peers = PeerTable()
        self.pazuzu = BraillePazuzu()
        self.reality_state = REALITY_LOCAL
        self._sandbox_stream = None
        self._pending_merge = None

    # ── Hash helpers ─────────────────────────────────────────

    def stream_hash(self) -> str:
        """SHA256 of the current renderer stream."""
        return hashlib.sha256(bytes(self.renderer.stream)).hexdigest()

    def stream_hash_short(self) -> str:
        return self.stream_hash()[:12]

    # ── Packet construction ──────────────────────────────────

    def _current_state_dict(self) -> dict:
        """Build a STATE packet dict from the renderer's current state."""
        stats = self.renderer.stream_stats()
        w = self.renderer.fold_w
        h = max(1, -(-len(self.renderer.stream) // w))
        return {
            "type": "STATE",
            "version": QIPX_VERSION,
            "node_id": self.identity.node_id,
            "name": self.identity.name,
            "stream_hash": self.stream_hash()[:12],
            "stream_len": len(self.renderer.stream),
            "fold_w": w,
            "fold_h": h,
            "palette_id": self.renderer.palette_id,
            "render_mode": self.renderer.render_mode,
            "score": round(self.renderer.current_score(), 4),
            "entropy": round(stats.get("entropy", 0.0), 3),
            "avg_density": round(stats.get("avg_density", 0.0), 3),
            "unique_masks": stats.get("unique_masks", 0),
            "ghost_w": self.renderer.ghost_w,
            "show_ghost": self.renderer.show_ghost,
            "scan_active": self.renderer.scan_active,
            "time": time.time(),
        }

    # ── Sending ──────────────────────────────────────────────

    def _send(self, pkt: dict, addr: Tuple[str, int]):
        """Encode and send a packet.  Silently ignores socket errors."""
        if not self._sock:
            return
        try:
            data = encode(pkt)
            if len(data) <= MAX_PACKET_SIZE:
                self._sock.sendto(data, addr)
        except OSError:
            pass

    def send_hello(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_hello) < HELLO_INTERVAL:
            return
        self._last_hello = now
        pkt = make_hello(
            self.identity.node_id, self.identity.name, DATA_PORT)
        self._send(pkt, (BROADCAST_ADDR, DISCOVERY_PORT))

    def publish_state(self):
        """Broadcast current projection state."""
        now = time.time()
        if (now - self._last_state) < STATE_INTERVAL:
            return
        self._last_state = now
        pkt = self._current_state_dict()
        self._send(pkt, (BROADCAST_ADDR, DISCOVERY_PORT))

    def publish_pazuzu(self):
        """Broadcast Pazuzu cognition state."""
        now = time.time()
        if (now - self._last_pazuzu) < 0.5:
            return
        self._last_pazuzu = now
        pkt = make_pazuzu(self.identity.node_id, **self.pazuzu.to_dict())
        self._send(pkt, (BROADCAST_ADDR, DISCOVERY_PORT))

    def send_stream_to(self, target_peer: PeerState):
        """Send the full stream to a specific peer in chunks."""
        stream = self.renderer.stream
        h = self.stream_hash()
        n_chunks = max(1, math.ceil(len(stream) / STREAM_CHUNK_SIZE))
        for i in range(n_chunks):
            chunk = stream[i * STREAM_CHUNK_SIZE:(i + 1) * STREAM_CHUNK_SIZE]
            pkt = make_stream(self.identity.node_id, h[:12],
                              chunk, chunk=i, chunks=n_chunks)
            self._send(pkt, target_peer.addr)

    def request_stream(self, target_node_id: str):
        """Ask a peer for its stream."""
        peer = self.peers.peers.get(target_node_id)
        if peer:
            pkt = make_get_stream(self.identity.node_id,
                                  target_node_id,
                                  peer.stream_hash)
            self._send(pkt, peer.addr)

    # ── Receiving ────────────────────────────────────────────

    def _rx_loop(self):
        """Background thread: receive UDP packets and enqueue them."""
        while self.running:
            try:
                data, addr = self._sock.recvfrom(65535)
                pkt = decode(data)
                if pkt and pkt.get("node_id") != self.identity.node_id:
                    self.inbox.put((pkt, addr))
            except socket.timeout:
                continue
            except OSError:
                break

    def _drain_inbox(self):
        """Process all queued packets.  Called from main thread."""
        while not self.inbox.empty():
            try:
                pkt, addr = self.inbox.get_nowait()
            except queue.Empty:
                break
            self._handle_packet(pkt, addr)

    def _handle_packet(self, pkt: dict, addr: Tuple[str, int]):
        """Route an incoming packet to the appropriate handler."""
        ptype = pkt.get("type", "")
        node_id = pkt.get("node_id", "")

        if not node_id:
            return

        # Forward Entanglement-layer packets to the controller
        if is_entangle_packet(pkt):
            if self.entanglement is not None:
                self.entanglement.handle_entangle_packet(pkt, addr)
            return

        # Forward Civilization-layer packets to the controller
        try:
            from bs_civilization_packets import is_civ_packet
            if is_civ_packet(pkt):
                if self.entanglement is not None and hasattr(self.entanglement, 'civilization'):
                    self.entanglement.civilization.handle_civ_packet(pkt, addr)
                return
        except ImportError:
            pass

        # Rate limit
        peer = self.peers.get_or_create(node_id, addr, pkt.get("name", ""))
        if not peer.check_rate_limit():
            return
        peer.record_packet()
        peer.last_seen = time.time()

        handlers = {
            "HELLO": self._on_hello,
            "STATE": self._on_state,
            "LOCK": self._on_lock,
            "GET_STREAM": self._on_get_stream,
            "STREAM": self._on_stream,
            "GHOST": self._on_ghost,
            "MERGE": self._on_merge,
            "PAZUZU": self._on_pazuzu,
        }
        handler = handlers.get(ptype)
        if handler:
            handler(peer, pkt, addr)

    # ── Packet handlers ──────────────────────────────────────

    def _on_hello(self, peer: PeerState, pkt: dict, addr):
        peer.name = pkt.get("name", peer.name)
        peer.caps = pkt.get("caps", [])
        # Reply with our own HELLO
        hello = make_hello(self.identity.node_id,
                           self.identity.name, DATA_PORT)
        self._send(hello, addr)

    def _on_state(self, peer: PeerState, pkt: dict, addr):
        peer.update_from_state(pkt)

        # If auto-merge and streams differ, request peer stream
        if self.auto_merge and peer.stream_hash:
            if peer.stream_hash != self.stream_hash()[:12]:
                if not peer.stream:
                    self.request_stream(peer.node_id)

    def _on_lock(self, peer: PeerState, pkt: dict, addr):
        """Remote lock → suggest ghost width."""
        self.peers.lock_count += 1
        remote_w = int(pkt.get("fold_w", 0))
        if remote_w > 0 and self.enabled:
            self.renderer.ghost_w = remote_w
            self.renderer.show_ghost = True
            self.renderer.needs_redraw = True

    def _on_get_stream(self, peer: PeerState, pkt: dict, addr):
        """Someone wants our stream.  Send it."""
        target = pkt.get("target")
        if target == self.identity.node_id:
            self.send_stream_to(peer)

    def _on_stream(self, peer: PeerState, pkt: dict, addr):
        """Receive stream chunks from a peer."""
        payload = decode_stream_payload(pkt)
        if payload is None:
            return
        if pkt.get("chunks", 1) <= 1:
            # Single chunk — store directly
            peer.stream = payload
        else:
            # Multi-chunk: accumulate
            if peer.stream is None:
                peer.stream = bytearray()
            peer.stream.extend(payload)

        # If we have the peer stream, try ghost merge
        if peer.stream and self.enabled:
            if self.on_remote_ghost:
                self.on_remote_ghost(peer)

    def _on_ghost(self, peer: PeerState, pkt: dict, addr):
        """Remote ghost request."""
        if pkt.get("target") in ("all", self.identity.node_id):
            ghost_w = int(pkt.get("ghost_w", 0))
            if ghost_w > 0:
                self.renderer.ghost_w = ghost_w
                self.renderer.show_ghost = True
                self.renderer.needs_redraw = True

    def _on_merge(self, peer: PeerState, pkt: dict, addr):
        """Remote merge announcement — store as pending."""
        self.peers.merge_count += 1
        self._pending_merge = pkt

    def _on_pazuzu(self, peer: PeerState, pkt: dict, addr):
        peer.update_from_pazuzu(pkt)

    # ── Merge engine ─────────────────────────────────────────

    def prepare_merge(self, peer: PeerState,
                      method: str = "density",
                      alpha: float = 0.5) -> Optional[bytearray]:
        """Merge local stream with a peer's stream.

        Returns the merged candidate, or None if unsafe.
        """
        if peer.stream is None:
            return None
        candidate = apply_merge(method, self.renderer.stream,
                                peer.stream, alpha=alpha,
                                score_a=self.renderer.current_score(),
                                score_b=peer.score)
        ok, reason = safety_check(candidate, self.renderer.stream)
        if not ok:
            return None
        return candidate

    def accept_merge(self):
        """Commit the sandbox stream as the new reality."""
        if self._sandbox_stream is not None:
            self._save_rollback()
            self.renderer.set_stream(self._sandbox_stream)
            self.reality_state = REALITY_MERGED
            self._sandbox_stream = None
            self._pending_merge = None
            self.renderer.needs_redraw = True

    def reject_merge(self):
        """Discard the sandbox and pending merge."""
        self._sandbox_stream = None
        self._pending_merge = None
        self.reality_state = REALITY_LOCAL

    def sandbox_merge(self, peer: PeerState,
                      method: str = "density",
                      alpha: float = 0.5) -> bool:
        """Prepare a merge and store in sandbox for preview."""
        candidate = self.prepare_merge(peer, method, alpha)
        if candidate:
            self._sandbox_stream = candidate
            self.reality_state = REALITY_SANDBOX
            return True
        return False

    def rollback(self):
        """Restore the last committed stream snapshot."""
        if self._rollback_snapshot is not None:
            self.renderer.set_stream(self._rollback_snapshot)
            self.renderer.fold_w = self._rollback_fold_w
            self.reality_state = REALITY_LOCAL
            self.renderer.needs_redraw = True
            self.pazuzu._prev_stream_hash = None  # force recalc

    def _save_rollback(self):
        """Save current stream + fold state for rollback."""
        self._rollback_snapshot = bytearray(self.renderer.stream)
        self._rollback_fold_w = self.renderer.fold_w

    # ── Consensus ────────────────────────────────────────────

    def consensus_fold(self) -> Optional[int]:
        """Get the swarm's consensus fold width."""
        return self.peers.fold_consensus()

    def consensus_merge(self, method: str = "consensus") -> Optional[bytearray]:
        """Merge all available peer streams using consensus method."""
        streams = [self.renderer.stream]
        weights = [1.0 / (self.renderer.current_score() + 0.001)]
        for p in self.peers.alive().values():
            if p.stream is not None:
                streams.append(p.stream)
                weights.append(1.0 / (p.score + 0.001))
        if len(streams) < 2:
            return None
        fn = MERGE_METHODS.get(method)
        if fn is None:
            return None
        if method in ("consensus", "weighted_cons"):
            result = fn(streams)
        else:
            result = apply_merge(method, streams[0], streams[1],
                                 score_a=1.0 / weights[0],
                                 score_b=1.0 / weights[1])
        ok, _ = safety_check(result, self.renderer.stream)
        return result if ok else None

    # ── Main tick (call from the game loop) ──────────────────

    def tick(self):
        """Process network events, publish state, run Pazuzu.  Called
        every frame from bs_main.py."""
        if not self.enabled:
            return

        now = time.time()

        # Expire dead peers periodically
        if now - self._last_expire > PEER_EXPIRE_INTERVAL:
            evicted = self.peers.expire()
            if evicted > 0 and self.on_peer_count_change:
                self.on_peer_count_change(self.peers.alive_count())
            self._last_expire = now

        # Send periodic broadcasts
        self.send_hello()
        self.publish_state()

        # Drain inbox
        self._drain_inbox()

        # Pazuzu cognition step (every ~100ms)
        if now - self.pazuzu.state.last_action_time > 0.1:
            alive = self.peers.alive()
            self.pazuzu.step(self.renderer, alive)
            self.publish_pazuzu()

            # Auto-merge if Pazuzu says MERGE and peer has stream
            if self.auto_merge and self.pazuzu.state.action == "MERGE":
                best = self.peers.best_scoring_peer()
                if best and best.stream:
                    self.sandbox_merge(best)
                    if self._sandbox_stream:
                        self.accept_merge()

            # Auto-lock if Pazuzu says LOCK
            if self.pazuzu.state.action == "LOCK":
                best = self.peers.best_scoring_peer()
                if best and best.fold_w > 0:
                    self.renderer.ghost_w = best.fold_w
                    self.renderer.show_ghost = True
                    self.renderer.needs_redraw = True

        # Trigger redraw if network caused state changes
        if self._pending_merge and self.on_merge_ready:
            self.on_merge_ready(self._pending_merge)
            self._pending_merge = None

    # ── HUD helpers ──────────────────────────────────────────

    def hud_status(self) -> str:
        """One-line status string for the renderer HUD."""
        if not self.enabled:
            return "QIPX: OFF"
        n = self.peers.alive_count()
        p = self.pazuzu
        mood = p.state.mood
        action = p.state.action
        lam = p.state.lambda_dom
        ci = p.state.criticality_index
        return (f"QIPX: ON peers={n} "
                f"lambda={lam:.4f} CI={ci:.2f} "
                f"mood={mood} action={action} "
                f"reality={self.reality_state}")

    def diagnostics_lines(self) -> List[str]:
        """Multi-line diagnostics text for overlay display."""
        lines = [
            f"QIPX v{QIPX_VERSION}  node={self.identity.node_id}",
            f"name={self.identity.name}  ip={self.identity.ip}",
            f"peers={self.peers.alive_count()}/{len(self.peers.peers)}  "
            f"merges={self.peers.merge_count}  locks={self.peers.lock_count}",
            f"reality={self.reality_state}  auto_merge={self.auto_merge}",
            "",
            self.pazuzu.one_line(),
            f"  parity={self.pazuzu.state.parity:+d}  "
            f"coherence={self.pazuzu.state.coherence:.3f}  "
            f"novelty={self.pazuzu.state.novelty:.3f}",
            "",
        ]
        for nid, peer in self.peers.alive().items():
            lines.append(
                f"  {nid}  W={peer.fold_w} H={peer.fold_h} "
                f"score={peer.score:.3f} mood={peer.mood()}"
            )
        return lines

    def peer_list_compact(self) -> List[str]:
        """Compact peer info for HUD sidebar."""
        lines = []
        for nid, peer in self.peers.alive().items():
            lines.append(
                f"{nid[-4:]} W={peer.fold_w} {peer.mood()}"
            )
        return lines
```

----------------------------------------

### File: `bs_qipx_merge.py`

**Path:** `./bs_qipx_merge.py`
**Extension:** `.py`
**Size:** 12,706 bytes (12.41 KB)

```py
"""
BS_QIPX_MERGE  –  Stream merge functions for reality creation.
                   Part of the QIPX distributed visual cognition mesh.

Twelve merge methods transform two (or more) raw 8-bit Braille streams
into a new derived stream.  Each method preserves a different structural
property:

  xor            – interference / paradox
  or             – accumulate all structure
  and            – keep only mutual agreement
  xnor           – keep sameness
  avg_byte       – numerical average
  weighted_byte  – weighted numerical average
  density        – preserve brightness mass, not exact bits
  polarity       – merge at paradox polarity level
  resonance_win  – block-level winner by local score
  consensus      – majority vote per micro-pixel
  weighted_cons  – confidence-weighted vote

Safety gates:
  mutation_limit, entropy_guard, collapse_guard, saturation_guard
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import math

from bs_engine import MAX_CELLS, bs_density


# ═══════════════════════════════════════════════════════════════
#  Canonical density masks
# ═══════════════════════════════════════════════════════════════
#  Maps density (0..8) to a representative mask with that many
#  active dots.  Uses left-to-right, row-major fill pattern.

_CANONICAL_DENSITY_MASKS: List[int] = [0] * 9
_mask = 0
for _d in range(9):
    _CANONICAL_DENSITY_MASKS[_d] = _mask
    if _d < 8:
        _mask |= 1 << _d
# Result: [0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF]


def canonical_mask(density: int) -> int:
    """Return a representative 8-bit mask with the given dot count."""
    return _CANONICAL_DENSITY_MASKS[max(0, min(8, density))]


# ═══════════════════════════════════════════════════════════════
#  Two-stream merge methods
# ═══════════════════════════════════════════════════════════════

def _ensure_len(a: bytearray, b: bytearray) -> Tuple[bytearray, bytearray]:
    """Pad shorter stream with zeros to match the longer one."""
    length = max(len(a), len(b), 1)
    a2 = bytearray(a[:length])
    b2 = bytearray(b[:length])
    a2 += bytearray(length - len(a2))
    b2 += bytearray(length - len(b2))
    return a2[:MAX_CELLS], b2[:MAX_CELLS]


def merge_xor(a: bytearray, b: bytearray) -> bytearray:
    """XOR merge: keep interference / difference."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(x ^ y for x, y in zip(a2, b2))


def merge_or(a: bytearray, b: bytearray) -> bytearray:
    """OR merge: accumulate all active structure."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(x | y for x, y in zip(a2, b2))


def merge_and(a: bytearray, b: bytearray) -> bytearray:
    """AND merge: keep only mutual agreement."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(x & y for x, y in zip(a2, b2))


def merge_xnor(a: bytearray, b: bytearray) -> bytearray:
    """XNOR merge: keep sameness."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(0xFF ^ (x ^ y) for x, y in zip(a2, b2))


def merge_avg_byte(a: bytearray, b: bytearray) -> bytearray:
    """Average byte merge: round((a+b)/2)."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(round((x + y) / 2) for x, y in zip(a2, b2))


def merge_weighted_byte(a: bytearray, b: bytearray,
                        alpha: float = 0.5) -> bytearray:
    """Weighted byte merge: round(alpha*a + (1-alpha)*b)."""
    a2, b2 = _ensure_len(a, b)
    return bytearray(
        min(255, max(0, round(alpha * x + (1 - alpha) * y)))
        for x, y in zip(a2, b2)
    )


def merge_density(a: bytearray, b: bytearray,
                  alpha: float = 0.5) -> bytearray:
    """Density merge: preserve brightness mass, not exact bit pattern."""
    a2, b2 = _ensure_len(a, b)
    out = bytearray(len(a2))
    for i in range(len(a2)):
        d_a = bs_density(a2[i])
        d_b = bs_density(b2[i])
        d_c = round(alpha * d_a + (1 - alpha) * d_b)
        out[i] = canonical_mask(d_c)
    return out


def merge_polarity(a: bytearray, b: bytearray,
                   alpha: float = 0.5) -> bytearray:
    """Polarity merge at the paradox level (D-4)."""
    a2, b2 = _ensure_len(a, b)
    out = bytearray(len(a2))
    for i in range(len(a2)):
        p_a = bs_density(a2[i]) - 4
        p_b = bs_density(b2[i]) - 4
        p_c = alpha * p_a + (1 - alpha) * p_b
        d_c = max(0, min(8, round(p_c + 4)))
        out[i] = canonical_mask(d_c)
    return out


def merge_resonance_winner(a: bytearray, b: bytearray,
                           score_a: float, score_b: float,
                           block_size: int = 80) -> bytearray:
    """Block-level winner: pick the stream with better local score."""
    a2, b2 = _ensure_len(a, b)
    length = len(a2)
    out = bytearray(length)
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        if score_a <= score_b:
            out[start:end] = a2[start:end]
        else:
            out[start:end] = b2[start:end]
    return out


# ═══════════════════════════════════════════════════════════════
#  Multi-stream merge methods
# ═══════════════════════════════════════════════════════════════

def merge_consensus(streams: List[bytearray]) -> bytearray:
    """Majority vote per micro-pixel across multiple streams."""
    if not streams:
        return bytearray(MAX_CELLS)
    if len(streams) == 1:
        return bytearray(streams[0][:MAX_CELLS])

    length = max(len(s) for s in streams)
    out = bytearray(length)

    for i in range(length):
        for bit in range(8):
            ones = sum(
                1 for s in streams
                if i < len(s) and (s[i] >> bit) & 1
            )
            if ones >= (len(streams) + 1) // 2:
                out[i] |= (1 << bit)

    return out[:MAX_CELLS]


def merge_weighted_consensus(streams: List[bytearray],
                             weights: List[float]) -> bytearray:
    """Confidence-weighted majority vote per micro-pixel."""
    if not streams:
        return bytearray(MAX_CELLS)
    if len(streams) == 1:
        return bytearray(streams[0][:MAX_CELLS])

    total_w = sum(weights[:len(streams)])
    if total_w == 0:
        total_w = 1.0

    length = max(len(s) for s in streams)
    out = bytearray(length)

    for i in range(length):
        for bit in range(8):
            w_sum = 0.0
            w_total = 0.0
            for idx, s in enumerate(streams):
                w = weights[idx] if idx < len(weights) else 1.0
                w_total += w
                if i < len(s) and (s[i] >> bit) & 1:
                    w_sum += w
            if w_total > 0 and w_sum >= 0.5 * w_total:
                out[i] |= (1 << bit)

    return out[:MAX_CELLS]


# ═══════════════════════════════════════════════════════════════
#  Merge method registry
# ═══════════════════════════════════════════════════════════════

MERGE_METHODS = {
    "xor":             merge_xor,
    "or":              merge_or,
    "and":             merge_and,
    "xnor":            merge_xnor,
    "avg_byte":        merge_avg_byte,
    "weighted_byte":   merge_weighted_byte,
    "density":         merge_density,
    "polarity":        merge_polarity,
    "resonance_win":   merge_resonance_winner,
    "consensus":       merge_consensus,
    "weighted_cons":   merge_weighted_consensus,
}


def apply_merge(method: str, a: bytearray, b: bytearray,
                **kwargs) -> bytearray:
    """Dispatch to a named merge method."""
    fn = MERGE_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown merge method: {method}")
    if method in ("consensus", "weighted_cons"):
        return fn([a, b])
    if method == "resonance_winner":
        return fn(a, b,
                  score_a=kwargs.get("score_a", 0.0),
                  score_b=kwargs.get("score_b", 0.0),
                  block_size=kwargs.get("block_size", 80))
    if method in ("weighted_byte", "density", "polarity"):
        return fn(a, b, alpha=kwargs.get("alpha", 0.5))
    return fn(a, b)


# ═══════════════════════════════════════════════════════════════
#  Safety gates
# ═══════════════════════════════════════════════════════════════

def bit_distance(a: bytearray, b: bytearray) -> float:
    """Normalised Hamming bit-distance between two streams."""
    a2, b2 = _ensure_len(a, b)
    total_bits = len(a2) * 8
    if total_bits == 0:
        return 0.0
    diff_bits = sum(bin(x ^ y).count('1') for x, y in zip(a2, b2))
    return diff_bits / total_bits


def density_distance(a: bytearray, b: bytearray) -> float:
    """Normalised density-distance between two streams."""
    a2, b2 = _ensure_len(a, b)
    n = len(a2)
    if n == 0:
        return 0.0
    return sum(abs(bs_density(x) - bs_density(y))
               for x, y in zip(a2, b2)) / (8 * n)


def safety_check(candidate: bytearray, local: bytearray,
                 max_mutation: float = 0.20,
                 max_entropy_delta: float = 2.0,
                 min_unique: int = 2,
                 max_saturation: float = 0.95,
                 max_emptiness: float = 0.95) -> Tuple[bool, str]:
    """Check whether a merge candidate passes all safety gates.

    Returns (accepted, reason).
    """
    n = len(candidate)
    if n == 0:
        return False, "empty candidate"

    # Mutation limit
    dist = bit_distance(candidate, local)
    if dist > max_mutation:
        return False, f"mutation {dist:.3f} > {max_mutation}"

    # Stream length
    if n > MAX_CELLS:
        return False, f"stream too long: {n} > {MAX_CELLS}"

    # Entropy guard
    from bs_engine import stream_stats
    stats_c = stream_stats(candidate)
    stats_l = stream_stats(local)
    if stats_c["entropy"] > stats_l["entropy"] + max_entropy_delta:
        return False, (f"entropy {stats_c['entropy']:.2f} > "
                       f"{stats_l['entropy']:.2f}+{max_entropy_delta}")

    # Collapse guard
    if stats_c["unique_masks"] < min_unique:
        return False, f"unique_masks {stats_c['unique_masks']} < {min_unique}"

    # Saturation guard
    if stats_c["full"] / n > max_saturation:
        return False, f"saturation {stats_c['full']/n:.2f} > {max_saturation}"

    # Emptiness guard
    if stats_c["empty"] / n > max_emptiness:
        return False, f"emptiness {stats_c['empty']/n:.2f} > {max_emptiness}"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════
#  Reality ledger entry
# ═══════════════════════════════════════════════════════════════

def make_ledger_entry(index: int, parent_hash: str,
                      before_hash: str, after_hash: str,
                      method: str, node_ids: List[str],
                      metrics_before: dict, metrics_after: dict,
                      accepted: bool) -> dict:
    """Create a reality ledger entry for auditability."""
    import time
    return {
        "index": index,
        "parent_hash": parent_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "method": method,
        "node_ids": node_ids,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "accepted": accepted,
        "timestamp": time.time(),
    }
```

----------------------------------------

### File: `bs_qipx_packets.py`

**Path:** `./bs_qipx_packets.py`
**Extension:** `.py`
**Size:** 6,813 bytes (6.65 KB)

```py
"""
BS_QIPX_PACKETS  –  QIPX packet definitions, encoding, and decoding.
                     Part of the QIPX distributed visual cognition mesh.

Wire format:  "QIPX " + JSON (newline-delimited, UTF-8)
Binary payloads (STREAM) use base64 chunks after the JSON header.

Packet types:
  HELLO       – discovery announcement
  STATE       – periodic projection state broadcast
  LOCK        – resonance lock announcement
  GET_STREAM  – request a peer's stream
  STREAM      – deliver stream data (base64 chunks)
  GHOST       – remote ghost overlay request
  MERGE       – announce a merge result
  PAZUZU      – broadcast cognition state
"""

from __future__ import annotations
import json
import base64
import time
from typing import Any, Dict, Optional, Tuple

QIPX_MAGIC = b"QIPX "
QIPX_VERSION = "0.1"

# Maximum UDP payload (safe for LAN, under typical MTU)
MAX_PACKET_SIZE = 8192


# ═══════════════════════════════════════════════════════════════
#  Packet construction helpers
# ═══════════════════════════════════════════════════════════════

def _base(pkt_type: str, node_id: str, **extra) -> dict:
    """Return a base packet dict with required fields."""
    d: Dict[str, Any] = {
        "type": pkt_type,
        "version": QIPX_VERSION,
        "node_id": node_id,
        "time": time.time(),
    }
    d.update(extra)
    return d


# ── HELLO ────────────────────────────────────────────────────

def make_hello(node_id: str, name: str, port: int,
               caps: list | None = None) -> dict:
    return _base("HELLO", node_id,
                 name=name,
                 port=port,
                 caps=caps or ["state", "stream", "ghost", "merge", "pazuzu"],
                 screen="640x480",
                 cell="2x4",
                 max_cells=38400)


# ── STATE ────────────────────────────────────────────────────

def make_state(node_id: str, name: str, **kwargs) -> dict:
    return _base("STATE", node_id, name=name, **kwargs)


# ── LOCK ─────────────────────────────────────────────────────

def make_lock(node_id: str, stream_hash: str, fold_w: int, fold_h: int,
              score: float, confidence: float = 0.0,
              reason: str = "resonance_minimum") -> dict:
    return _base("LOCK", node_id,
                 stream_hash=stream_hash,
                 fold_w=fold_w, fold_h=fold_h,
                 score=score, confidence=confidence,
                 reason=reason)


# ── GET_STREAM ───────────────────────────────────────────────

def make_get_stream(node_id: str, target: str,
                    stream_hash: str) -> dict:
    return _base("GET_STREAM", node_id,
                 target=target,
                 stream_hash=stream_hash,
                 encoding="raw")


# ── STREAM ───────────────────────────────────────────────────

def make_stream(node_id: str, stream_hash: str,
                stream: bytearray, chunk: int = 0,
                chunks: int = 1) -> dict:
    return _base("STREAM", node_id,
                 stream_hash=stream_hash,
                 stream_len=len(stream),
                 encoding="b64",
                 chunk=chunk,
                 chunks=chunks,
                 data=base64.b64encode(bytes(stream)).decode("ascii"))


# ── GHOST ────────────────────────────────────────────────────

def make_ghost(node_id: str, target: str,
               stream_hash: str, ghost_w: int,
               alpha: int = 60, polarity: str = "density_split") -> dict:
    return _base("GHOST", node_id,
                 target=target,
                 stream_hash=stream_hash,
                 ghost_w=ghost_w,
                 alpha=alpha,
                 polarity=polarity)


# ── MERGE ────────────────────────────────────────────────────

def make_merge(node_id: str, merge_id: str,
               parents: list, method: str,
               alpha: float = 0.5,
               result_hash: str = "") -> dict:
    return _base("MERGE", node_id,
                 merge_id=merge_id,
                 parents=parents,
                 method=method,
                 alpha=alpha,
                 result_hash=result_hash)


# ── PAZUZU ───────────────────────────────────────────────────

def make_pazuzu(node_id: str, **kwargs) -> dict:
    return _base("PAZUZU", node_id, **kwargs)


# ═══════════════════════════════════════════════════════════════
#  Wire encoding / decoding
# ═══════════════════════════════════════════════════════════════

def encode(pkt: dict) -> bytes:
    """Serialize a packet dict to the QIPX wire format."""
    json_bytes = json.dumps(pkt, separators=(",", ":")).encode("utf-8")
    return QIPX_MAGIC + json_bytes


def decode(data: bytes) -> Optional[dict]:
    """Deserialize raw bytes to a packet dict.  Returns None on failure."""
    if not data or len(data) < len(QIPX_MAGIC):
        return None
    if not data.startswith(QIPX_MAGIC):
        return None
    if len(data) > MAX_PACKET_SIZE:
        return None
    try:
        return json.loads(data[len(QIPX_MAGIC):].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def decode_stream_payload(pkt: dict) -> Optional[bytearray]:
    """Extract a bytearray from a STREAM packet's base64 data field."""
    if pkt.get("type") != "STREAM":
        return None
    b64 = pkt.get("data")
    if not b64:
        return None
    try:
        return bytearray(base64.b64decode(b64))
    except Exception:
        return None
```

----------------------------------------

### File: `bs_qipx_peer.py`

**Path:** `./bs_qipx_peer.py`
**Extension:** `.py`
**Size:** 6,398 bytes (6.25 KB)

```py
"""
BS_QIPX_PEER  –  Peer table, peer state tracking, and trust scoring.
                 Part of the QIPX distributed visual cognition mesh.

Each discovered node is tracked as a PeerState entry.
Peers expire after PEER_TIMEOUT seconds of silence.
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Constants ────────────────────────────────────────────────
PEER_TIMEOUT = 5.0       # seconds before a peer is considered dead
MAX_PEERS = 32           # max tracked peers
RATE_LIMIT_WINDOW = 1.0  # seconds
RATE_LIMIT_MAX = 60      # packets per window per peer


@dataclass
class PeerState:
    """Live state of a remote BrailleStream node."""

    node_id: str
    name: str = ""
    addr: Tuple[str, int] = ("", 0)
    first_seen: float = 0.0
    last_seen: float = 0.0
    caps: List[str] = field(default_factory=list)

    # Projection state (from STATE packets)
    stream_hash: str = ""
    stream_len: int = 0
    fold_w: int = 320
    fold_h: int = 120
    palette_id: int = 1
    render_mode: int = 0
    score: float = 0.0
    entropy: float = 0.0
    avg_density: float = 0.0
    unique_masks: int = 0

    # Ghost state
    ghost_w: int = 0
    show_ghost: bool = False

    # Pazuzu state
    pazuzu: Dict = field(default_factory=dict)

    # Stream cache
    stream: Optional[bytearray] = None

    # Trust / rate tracking
    trust: float = 0.5
    packets_received: int = 0
    last_packet_times: List[float] = field(default_factory=list)

    def update_from_state(self, state: dict):
        """Update fields from an inbound STATE packet dict."""
        self.stream_hash = state.get("stream_hash", self.stream_hash)
        self.stream_len = state.get("stream_len", self.stream_len)
        self.fold_w = int(state.get("fold_w", self.fold_w))
        self.fold_h = int(state.get("fold_h", self.fold_h))
        self.palette_id = int(state.get("palette_id", self.palette_id))
        self.render_mode = int(state.get("render_mode", self.render_mode))
        self.score = float(state.get("score", self.score))
        self.entropy = float(state.get("entropy", self.entropy))
        self.avg_density = float(state.get("avg_density", self.avg_density))
        self.unique_masks = int(state.get("unique_masks", self.unique_masks))
        self.ghost_w = int(state.get("ghost_w", self.ghost_w))
        self.show_ghost = bool(state.get("show_ghost", self.show_ghost))

    def update_from_pazuzu(self, pazuzu: dict):
        """Update Pazuzu cognition state from a PAZUZU packet."""
        self.pazuzu.update(pazuzu)

    def is_alive(self) -> bool:
        return (time.time() - self.last_seen) < PEER_TIMEOUT

    def mood(self) -> str:
        return self.pazuzu.get("mood", "???")

    def action(self) -> str:
        return self.pazuzu.get("action", "???")

    def check_rate_limit(self) -> bool:
        """Return True if the peer is within rate limits."""
        now = time.time()
        # Prune old entries
        self.last_packet_times = [
            t for t in self.last_packet_times
            if now - t < RATE_LIMIT_WINDOW
        ]
        if len(self.last_packet_times) >= RATE_LIMIT_MAX:
            return False
        self.last_packet_times.append(now)
        return True

    def record_packet(self):
        self.packets_received += 1


class PeerTable:
    """Thread-safe-ish peer table with expiry and lookup."""

    def __init__(self, max_peers: int = MAX_PEERS):
        self.peers: Dict[str, PeerState] = {}
        self.max_peers = max_peers
        self.merge_count: int = 0
        self.lock_count: int = 0

    def get_or_create(self, node_id: str,
                      addr: Tuple[str, int] = ("", 0),
                      name: str = "") -> PeerState:
        """Return existing peer or create a new one."""
        peer = self.peers.get(node_id)
        if peer is not None:
            return peer
        # Enforce max peer limit
        if len(self.peers) >= self.max_peers:
            self._evict_oldest()
        peer = PeerState(
            node_id=node_id,
            name=name,
            addr=addr,
            first_seen=time.time(),
            last_seen=time.time(),
        )
        self.peers[node_id] = peer
        return peer

    def touch(self, node_id: str):
        """Update last_seen timestamp."""
        peer = self.peers.get(node_id)
        if peer:
            peer.last_seen = time.time()

    def expire(self) -> int:
        """Remove dead peers.  Returns count of evicted entries."""
        now = time.time()
        dead = [
            nid for nid, p in self.peers.items()
            if (now - p.last_seen) > PEER_TIMEOUT
        ]
        for nid in dead:
            del self.peers[nid]
        return len(dead)

    def alive(self) -> Dict[str, PeerState]:
        """Return only currently-alive peers."""
        return {nid: p for nid, p in self.peers.items() if p.is_alive()}

    def alive_count(self) -> int:
        return len(self.alive())

    def best_scoring_peer(self) -> Optional[PeerState]:
        """Return the alive peer with the lowest (best) resonance score."""
        alive = self.alive()
        if not alive:
            return None
        return min(alive.values(), key=lambda p: p.score)

    def fold_consensus(self) -> Optional[int]:
        """Weighted consensus fold width from all alive peers."""
        alive = self.alive()
        if not alive:
            return None
        total_weight = 0.0
        weighted_sum = 0.0
        for p in alive.values():
            if p.score > 0:
                w = 1.0 / (p.score + 0.001)
                weighted_sum += w * p.fold_w
                total_weight += w
        if total_weight == 0:
            return None
        return int(round(weighted_sum / total_weight))

    def _evict_oldest(self):
        """Remove the peer with the oldest last_seen time."""
        if not self.peers:
            return
        oldest = min(self.peers, key=lambda n: self.peers[n].last_seen)
        del self.peers[oldest]

    def summary(self) -> str:
        """One-line summary string for HUD display."""
        alive = self.alive_count()
        total = len(self.peers)
        return f"peers={alive}/{total}"
```

----------------------------------------

### File: `bs_reality_consensus.py`

**Path:** `./bs_reality_consensus.py`
**Extension:** `.py`
**Size:** 13,531 bytes (13.21 KB)

```py
"""
BS_REALITY_CONSENSUS  -  Voting and majority reality agreement.
                          Part of the BS-TOS-IPX Entanglement Layer.

Implements three consensus levels:
  1. Simple majority    – votes_for / total_votes > 0.5
  2. Strong majority    – votes_for / total_votes >= 0.666
  3. Weighted majority  – trust*confidence*criticality / (score+epsilon)

Consensus only resolves on heartbeat windows to prevent constant switching.

Reality acceptance requires:
  - majority_weight >= theta_majority (default 0.618)
  - lambda in critical band
  - entropy bounded
  - unique_masks >= minimum
  - rollback snapshot exists
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs_engine import GRID_W, stream_stats
from bs_pazuzu import LAMBDA_MIN, LAMBDA_MAX


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

THETA_MAJORITY_DEFAULT = 0.618   # golden ratio as acceptance threshold
THETA_STRONG = 0.666             # strong majority
THETA_SIMPLE = 0.5               # simple majority
MIN_UNIQUE_MASKS = 2             # minimum unique masks to accept
MAX_ENTROPY_DELTA = 2.0          # max entropy deviation from local
EPSILON = 0.001                  # small constant to avoid division by zero

# Consensus subjects that can be voted on
CONSENSUS_SUBJECTS = {
    "fold_w", "palette_id", "render_mode", "ghost_w",
    "scan_active", "merge_method", "lock_w",
}


# ═══════════════════════════════════════════════════════════════
#  Vote data
# ═══════════════════════════════════════════════════════════════

@dataclass
class Vote:
    """A single vote on a consensus subject."""
    vote_id: str
    node_id: str
    subject: str
    value: Any
    stream_hash: str = ""
    weight: float = 0.5
    confidence: float = 0.5
    score: float = 0.0
    timestamp: float = 0.0


@dataclass
class ConsensusResult:
    """The outcome of a consensus round."""
    consensus_id: str
    subject: str
    value: Any
    votes_count: int
    majority: float
    weighted_majority: float
    weighted_confidence: float
    effect: str
    accepted: bool
    reason: str = ""
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Consensus Engine
# ═══════════════════════════════════════════════════════════════

class ConsensusEngine:
    """Manages voting rounds and majority reality computation.

    Votes are collected over time.  On each heartbeat consensus window,
    pending votes are tallied and a consensus result is emitted.
    """

    def __init__(self, theta_majority: float = THETA_MAJORITY_DEFAULT):
        self.theta_majority = theta_majority
        self._votes: Dict[str, List[Vote]] = {}   # subject → list of votes
        self._vote_expiry = 10.0  # seconds before a vote expires
        self._consensus_history: List[ConsensusResult] = []
        self._last_consensus_id: str = ""
        self._vote_counter = 0

    # ── Vote management ───────────────────────────────────────

    def cast_vote(self, node_id: str, subject: str, value: Any,
                  stream_hash: str = "", weight: float = 0.5,
                  confidence: float = 0.5, score: float = 0.0) -> str:
        """Register a vote.  Returns the vote_id."""
        self._vote_counter += 1
        vote_id = f"vote-{subject}-{self._vote_counter:05d}"
        vote = Vote(
            vote_id=vote_id,
            node_id=node_id,
            subject=subject,
            value=value,
            stream_hash=stream_hash,
            weight=weight,
            confidence=confidence,
            score=score,
            timestamp=time.time(),
        )
        if subject not in self._votes:
            self._votes[subject] = []
        self._votes[subject].append(vote)
        return vote_id

    def cast_local_vote(self, renderer, pazuzu_state) -> str:
        """Cast a vote for the local node's current fold width."""
        stats = renderer.stream_stats()
        weight = self._compute_weight(
            trust=0.7,
            confidence=pazuzu_state.coherence,
            criticality=pazuzu_state.criticality_index,
            score=renderer.current_score(),
        )
        return self.cast_vote(
            node_id="LOCAL",
            subject="fold_w",
            value=renderer.fold_w,
            stream_hash="",
            weight=weight,
            confidence=pazuzu_state.coherence,
            score=renderer.current_score(),
        )

    def expire_votes(self):
        """Remove expired votes from all subjects."""
        now = time.time()
        for subject in list(self._votes.keys()):
            self._votes[subject] = [
                v for v in self._votes[subject]
                if now - v.timestamp < self._vote_expiry
            ]
            if not self._votes[subject]:
                del self._votes[subject]

    def clear_votes(self, subject: Optional[str] = None):
        """Clear votes for a specific subject, or all votes."""
        if subject:
            self._votes.pop(subject, None)
        else:
            self._votes.clear()

    # ── Consensus resolution ──────────────────────────────────

    def resolve(self, subject: str = "fold_w",
                lam: float = 0.01,
                local_entropy: float = 0.0,
                local_unique_masks: int = 0,
                has_rollback: bool = False) -> Optional[ConsensusResult]:
        """Tally votes for a subject and produce a consensus result.

        Returns None if no votes exist or consensus cannot be reached.
        """
        votes = self._votes.get(subject, [])
        if not votes:
            return None

        # Tally votes by value
        value_votes: Dict[Any, List[Vote]] = {}
        for v in votes:
            if v.value not in value_votes:
                value_votes[v.value] = []
            value_votes[v.value].append(v)

        # Find the value with highest weighted support
        best_value = None
        best_weight_sum = 0.0
        total_weight = 0.0
        total_votes = len(votes)

        for val, val_votes_list in value_votes.items():
            w_sum = sum(self._compute_weight(
                trust=0.5,
                confidence=v.confidence,
                criticality=0.5,
                score=v.score,
            ) for v in val_votes_list)
            total_weight += w_sum
            if w_sum > best_weight_sum:
                best_weight_sum = w_sum
                best_value = val

        if best_value is None:
            return None

        # Simple majority
        simple_maj = best_weight_sum / total_weight if total_weight > 0 else 0.0

        # Weighted majority
        weighted_maj = best_weight_sum / total_weight if total_weight > 0 else 0.0

        # Weighted confidence (average confidence of supporting votes)
        supporting = value_votes.get(best_value, [])
        weighted_conf = (sum(v.confidence for v in supporting)
                         / len(supporting)) if supporting else 0.0

        # Determine effect
        if weighted_maj >= THETA_STRONG:
            effect = "DIRECT_SWITCH"
        elif weighted_maj >= self.theta_majority:
            effect = "DIRECT_SWITCH"
        elif weighted_maj >= THETA_SIMPLE:
            effect = "SUGGEST"
        else:
            effect = "IGNORE"

        # Reality acceptance check
        accepted = False
        reason = ""

        if weighted_maj < self.theta_majority:
            reason = f"majority {weighted_maj:.3f} < {self.theta_majority}"
        elif not (LAMBDA_MIN <= abs(lam) <= LAMBDA_MAX):
            reason = f"lambda {lam:.4f} outside critical band"
        elif local_unique_masks < MIN_UNIQUE_MASKS:
            reason = f"unique_masks {local_unique_masks} < {MIN_UNIQUE_MASKS}"
        elif not has_rollback:
            reason = "no rollback snapshot"
        else:
            accepted = True
            reason = "accepted"

        consensus_id = f"cons-{subject}-{int(time.time())}"
        self._last_consensus_id = consensus_id

        result = ConsensusResult(
            consensus_id=consensus_id,
            subject=subject,
            value=best_value,
            votes_count=total_votes,
            majority=simple_maj,
            weighted_majority=weighted_maj,
            weighted_confidence=weighted_conf,
            effect=effect,
            accepted=accepted,
            reason=reason,
            timestamp=time.time(),
        )

        self._consensus_history.append(result)
        # Keep only recent history
        if len(self._consensus_history) > 100:
            self._consensus_history = self._consensus_history[-50:]

        return result

    def resolve_all_pending(self, lam: float, local_entropy: float,
                            local_unique_masks: int,
                            has_rollback: bool) -> List[ConsensusResult]:
        """Resolve all subjects that have votes."""
        results = []
        for subject in list(self._votes.keys()):
            result = self.resolve(
                subject, lam, local_entropy,
                local_unique_masks, has_rollback)
            if result:
                results.append(result)
                # Clear resolved votes
                self.clear_votes(subject)
        return results

    # ── Fold consensus (simplified, no voting) ────────────────

    def fold_consensus_value(self, local_fold_w: int,
                             peer_folds: Dict[str, int],
                             peer_trusts: Optional[Dict[str, float]] = None,
                             peer_scores: Optional[Dict[str, float]] = None
                             ) -> Tuple[int, float]:
        """Compute a consensus fold width from local + peer folds.

        W* = round(sum(weight_i * W_i) / sum(weight_i))

        Returns (consensus_fold, majority_ratio).
        """
        folds = [local_fold_w]
        weights = [1.0]

        for nid, fw in peer_folds.items():
            if fw > 0:
                folds.append(fw)
                trust = 0.5
                if peer_trusts and nid in peer_trusts:
                    trust = peer_trusts[nid]
                score = 10.0
                if peer_scores and nid in peer_scores:
                    score = peer_scores[nid]
                w = trust / (score + EPSILON)
                weights.append(w)

        if not folds:
            return local_fold_w, 1.0

        total_w = sum(weights)
        weighted_sum = sum(f * w for f, w in zip(folds, weights))

        if total_w == 0:
            return local_fold_w, 1.0

        consensus_w = round(weighted_sum / total_w)
        consensus_w = max(1, min(GRID_W, consensus_w))

        # Majority ratio: how many nodes agree with the consensus value
        agree = sum(1 for f in folds if abs(f - consensus_w) <= 2)
        majority = agree / len(folds)

        return consensus_w, majority

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_weight(trust: float, confidence: float,
                        criticality: float, score: float) -> float:
        """weight_i = trust * confidence * criticality / (score + epsilon)"""
        return (trust * confidence * criticality) / (score + EPSILON)

    def last_result(self) -> Optional[ConsensusResult]:
        """Return the most recent consensus result."""
        return self._consensus_history[-1] if self._consensus_history else None

    def vote_count(self, subject: Optional[str] = None) -> int:
        """Count pending votes, optionally for a specific subject."""
        if subject:
            return len(self._votes.get(subject, []))
        return sum(len(v) for v in self._votes.values())

    def subjects(self) -> List[str]:
        """Return list of subjects with pending votes."""
        return list(self._votes.keys())

    def one_line(self) -> str:
        """Compact HUD string."""
        last = self.last_result()
        if last:
            return (f"subject={last.subject} value={last.value} "
                    f"maj={last.weighted_majority:.2f} "
                    f"{'ACCEPT' if last.accepted else 'REJECT'}")
        votes = self.vote_count()
        subjects = self.subjects()
        if votes > 0:
            return f"pending: {votes} votes on {subjects}"
        return "idle"
```

----------------------------------------

### File: `bs_renderer.py`

**Path:** `./bs_renderer.py`
**Extension:** `.py`
**Size:** 23,426 bytes (22.88 KB)

```py
"""
BS_RENDERER  –  Pygame 640×480 16-color renderer.
                Faithfully translates BS_RENDER.HC + BS_PALETTE.HC.

Render modes:
  0  Native Pixel Mode    – 320×120 cells, exact 640×480
  1  Fold Scan Mode       – arbitrary W, clipped to screen
  2  Scaled Preview Mode  – any W/H scaled into 640×480
  3  Density Map Mode     – one cell → one coloured block
  4  Resonance Analyzer   – lists best W values on screen
  5  Paradox Overlay      – dual-colour polarity rendering

Palette modes:
  0  Binary Holy Light    – OFF=black, ON=white
  1  Density Fire         – 9-step warm gradient
  2  Paradox Dualism      – entropy/tension/negentropy
  3  Terminal Retina      – greyscale + accent colours
  4  Amiga Retro          – classic 16-colour Amiga palette feel
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Any

import pygame

from bs_engine import (
    SCREEN_W, SCREEN_H, CELL_W, CELL_H, GRID_W, GRID_H, MAX_CELLS,
    bs_dot, bs_density, resonance_score, find_resonance,
    stream_to_screen, reverse_parse,
    TARGET_CELL_RATIO,
)

# QIPX HUD support (imported lazily to avoid hard dependency)
_qipx_node = None  # set externally by bs_main.py

# Entanglement HUD support (set externally by bs_main.py)
_entangle_ctrl = None

# Civilization HUD support (set externally by bs_main.py)
_civ_ctrl = None


# ═══════════════════════════════════════════════════════════════
#  TempleOS 16-Colour Palettes
# ═══════════════════════════════════════════════════════════════
#  Each palette maps density (0..8) → (R, G, B).
#  Density 0 = empty cell (background), 8 = fully filled.

BLACK   = (0,   0,   0)
WHITE   = (255, 255, 255)
DKGRAY  = (64,  64,  64)
GRAY    = (128, 128, 128)
LTGRAY  = (192, 192, 192)
BLUE    = (0,   0,  170)
LTBLUE  = (85,  85,  255)
CYAN    = (0,   170, 170)
LTCYAN  = (85,  255, 255)
GREEN   = (0,   170, 0)
LTGREEN = (85,  255, 85)
YELLOW  = (170, 170, 0)
BROWN   = (170, 85,  0)
LTRED   = (170, 85,  85)
RED     = (170, 0,   0)
MAGENTA = (170, 0,   170)
LTMAGENTA = (255, 85, 255)


def _build_palette(colors_by_density: dict) -> List[Tuple[int, int, int]]:
    """Ensure every density 0..8 has an entry; fill gaps with BLACK."""
    return [colors_by_density.get(d, BLACK) for d in range(9)]


PALETTES = {
    # Mode 0 – Binary Holy Light: OFF=black, ON=white
    0: _build_palette({
        0: BLACK,
        1: WHITE, 2: WHITE, 3: WHITE, 4: WHITE,
        5: WHITE, 6: WHITE, 7: WHITE, 8: WHITE,
    }),

    # Mode 1 – Density Fire
    1: _build_palette({
        0: BLACK,
        1: (0,   0,   100),   # dark blue
        2: BLUE,
        3: CYAN,
        4: GREEN,
        5: YELLOW,
        6: BROWN,
        7: LTRED,
        8: WHITE,
    }),

    # Mode 2 – Paradox Dualism
    #   low  density  → entropy (dark, blue)
    #   mid  density  → tension (green, yellow)
    #   high density  → negentropy (red, white)
    2: _build_palette({
        0: BLACK,
        1: (20,  20,  120),
        2: (40,  40,  180),
        3: (0,   140, 60),
        4: (80,  180, 0),
        5: YELLOW,
        6: (200, 100, 0),
        7: (220, 40,  40),
        8: WHITE,
    }),

    # Mode 3 – Terminal Retina
    3: _build_palette({
        0: BLACK,
        1: (30,  30,  30),
        2: (60,  60,  60),
        3: GRAY,
        4: LTGRAY,
        5: BLUE,
        6: CYAN,
        7: (200, 50,  50),
        8: WHITE,
    }),

    # Mode 4 – Amiga Retro
    4: _build_palette({
        0: BLACK,
        1: (0,   0,   136),
        2: (0,   136, 0),
        3: (0,   136, 136),
        4: (136, 0,   0),
        5: (136, 0,   136),
        6: (136, 68,  0),
        7: (204, 204, 204),
        8: WHITE,
    }),
}

PALETTE_NAMES = {
    0: "Binary Holy Light",
    1: "Density Fire",
    2: "Paradox Dualism",
    3: "Terminal Retina",
    4: "Amiga Retro",
}


# ═══════════════════════════════════════════════════════════════
#  Renderer
# ═══════════════════════════════════════════════════════════════

class BrailleStreamRenderer:
    """640×480 pygame renderer that turns a raw 8-bit Braille stream
    into visible micro-pixel art using TempleOS-style 16-colour palettes."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("BrailleStream – TempleOS Textual Retina")
        self.clock = pygame.time.Clock()
        self.font_hud = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 12)

        # State
        self.stream: bytearray = bytearray(MAX_CELLS)
        self.fold_w: int = GRID_W
        self.palette_id: int = 1       # start with Density Fire
        self.render_mode: int = 0      # start with Native Pixel Mode
        self.show_ghost: bool = False   # multi-width ghosting
        self.ghost_w: int = 0
        self.resonance_lock: bool = False
        self.lock_flash: int = 0
        self.scan_active: bool = False
        self.scan_w: int = 10
        self.running: bool = True
        self.needs_redraw: bool = True

        # Pre-computed resonance cache
        self._res_cache: Optional[List] = None
        self._res_len: int = -1

    # ── helpers ──────────────────────────────────────────────

    @property
    def palette(self) -> List[Tuple[int, int, int]]:
        return PALETTES[self.palette_id]

    def set_stream(self, data: bytearray):
        self.stream = data
        self._res_cache = None
        self._res_len = -1
        self.needs_redraw = True

    def get_resonance(self, top_n: int = 15):
        """Cached resonance computation."""
        if self._res_cache is None or self._res_len != len(self.stream):
            self._res_cache = find_resonance(len(self.stream), top_n=top_n)
            self._res_len = len(self.stream)
        return self._res_cache

    def current_score(self) -> float:
        return resonance_score(len(self.stream), self.fold_w)

    # ── render modes ─────────────────────────────────────────

    def render_native(self, surface: pygame.Surface):
        """MODE 0:  Native Pixel Mode – 320×120 cells, exact 640×480."""
        palette = self.palette
        bg = palette[0]
        surface.fill(bg)
        for i, mask in enumerate(self.stream):
            px, py = stream_to_screen(i, GRID_W)
            if px >= SCREEN_W or py >= SCREEN_H:
                break
            d = bs_density(mask)
            color = palette[d]
            if color == bg:
                continue
            for r in range(CELL_H):
                for c in range(CELL_W):
                    if bs_dot(mask, r, c):
                        surface.set_at((px + c, py + r), color)

    def render_fold(self, surface: pygame.Surface):
        """MODE 1:  Fold Scan Mode – arbitrary W, clipped."""
        palette = self.palette
        bg = palette[0]
        surface.fill(bg)
        w = self.fold_w
        for i, mask in enumerate(self.stream):
            px, py = stream_to_screen(i, w)
            if px >= SCREEN_W or py >= SCREEN_H:
                break
            d = bs_density(mask)
            color = palette[d]
            if color == bg:
                continue
            for r in range(CELL_H):
                for c in range(CELL_W):
                    if bs_dot(mask, r, c):
                        sx, sy = px + c, py + r
                        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
                            surface.set_at((sx, sy), color)

    def render_scaled(self, surface: pygame.Surface):
        """MODE 2:  Scaled Preview – any W/H scaled into 640×480."""
        w = self.fold_w
        h = max(1, math.ceil(len(self.stream) / w))
        # cell grid dimensions
        cw = SCREEN_W / max(w, 1)
        ch = SCREEN_H / max(h, 1)
        palette = self.palette
        bg = palette[0]
        surface.fill(bg)

        for i, mask in enumerate(self.stream):
            cx_f = (i % w) * cw
            cy_f = (i // w) * ch
            d = bs_density(mask)
            color = palette[d]
            if color == bg:
                continue
            # draw a small filled rect for each dot
            pw = max(1, int(cw / CELL_W))
            ph = max(1, int(ch / CELL_H))
            for r in range(CELL_H):
                for c in range(CELL_W):
                    if bs_dot(mask, r, c):
                        rx = int(cx_f + c * pw)
                        ry = int(cy_f + r * ph)
                        rw = max(1, int((c + 1) * pw) - int(c * pw))
                        rh = max(1, int((r + 1) * ph) - int(r * ph))
                        pygame.draw.rect(surface, color,
                                         (rx, ry, rw, rh))

    def render_density_map(self, surface: pygame.Surface):
        """MODE 3:  Density Map – one cell → one coloured block."""
        w = self.fold_w
        palette = self.palette
        bg = palette[0]
        surface.fill(bg)
        h = max(1, math.ceil(len(self.stream) / w))
        cw = SCREEN_W / max(w, 1)
        ch = SCREEN_H / max(h, 1)

        for i, mask in enumerate(self.stream):
            cx_f = (i % w) * cw
            cy_f = (i // w) * ch
            d = bs_density(mask)
            color = palette[d]
            if color == bg:
                continue
            rect = pygame.Rect(int(cx_f), int(cy_f),
                               max(1, int(cw)), max(1, int(ch)))
            pygame.draw.rect(surface, color, rect)

    def render_resonance_analyzer(self, surface: pygame.Surface):
        """MODE 4:  Resonance Analyzer – text overlay of best widths."""
        surface.fill((0, 0, 20))
        top = self.get_resonance(20)

        y = 20
        title = self.font_title.render(
            "RESONANCE ANALYZER", True, (85, 255, 255))
        surface.blit(title, (SCREEN_W // 2 - title.get_width() // 2, y))
        y += 30

        header = self.font_hud.render(
            "  Score     W      H      W/H      Visual", True, LTGRAY)
        surface.blit(header, (20, y))
        y += 20
        pygame.draw.line(surface, DKGRAY, (20, y), (SCREEN_W - 20, y))
        y += 8

        for score, w, h in top:
            ratio = w / max(h, 1)
            marker = " <<<" if w == self.fold_w else ""
            bar_len = int(min(score * 80, SCREEN_W - 200))
            line = f"  {score:6.3f}   {w:4d}  {h:4d}   {ratio:6.3f}"
            col = (0, 255, 0) if w == self.fold_w else (180, 180, 180)
            txt = self.font_small.render(line + marker, True, col)
            surface.blit(txt, (20, y))
            # mini bar
            pygame.draw.rect(surface, (60, 120, 200),
                             (320, y + 2, bar_len, 10))
            y += 18
            if y > SCREEN_H - 60:
                break

        # Footer
        footer = self.font_small.render(
            f"Stream len: {len(self.stream)}  |  "
            f"Target W/H ratio: {TARGET_CELL_RATIO:.3f}  |  "
            f"Current fold: {self.fold_w}",
            True, YELLOW)
        surface.blit(footer, (20, SCREEN_H - 30))

    def render_paradox(self, surface: pygame.Surface):
        """MODE 5:  Paradox Overlay – density split into dual polarities."""
        # blue = dissipation / entropy   (low density)
        # red  = storage / structure     (high density)
        # white = paradox closure        (density = 4)
        surface.fill(BLACK)
        w = self.fold_w
        for i, mask in enumerate(self.stream):
            px, py = stream_to_screen(i, w)
            if px >= SCREEN_W or py >= SCREEN_H:
                break
            d = bs_density(mask)
            for r in range(CELL_H):
                for c in range(CELL_W):
                    if bs_dot(mask, r, c):
                        sx, sy = px + c, py + r
                        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
                            if d < 4:
                                intensity = int(80 + (d / 4) * 175)
                                color = (0, 0, intensity)          # blue
                            elif d == 4:
                                color = WHITE                      # paradox
                            else:
                                intensity = int(80 + ((d - 4) / 4) * 175)
                                color = (intensity, 0, 0)          # red
                            surface.set_at((sx, sy), color)

    # ── ghost overlay ────────────────────────────────────────

    def render_ghost(self, surface: pygame.Surface):
        """Render a second fold width as a translucent overlay."""
        if not self.show_ghost or self.ghost_w < 1:
            return
        ghost = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        w = self.ghost_w
        for i, mask in enumerate(self.stream):
            px, py = stream_to_screen(i, w)
            if px >= SCREEN_W or py >= SCREEN_H:
                break
            d = bs_density(mask)
            alpha = 60
            color = (100, 100, 255, alpha) if d < 4 else (255, 100, 100, alpha)
            for r in range(CELL_H):
                for c in range(CELL_W):
                    if bs_dot(mask, r, c):
                        sx, sy = px + c, py + r
                        if 0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H:
                            ghost.set_at((sx, sy), color)
        surface.blit(ghost, (0, 0))

    # ── HUD ──────────────────────────────────────────────────

    def draw_hud(self, surface: pygame.Surface):
        """Draw the heads-up display overlay, including QIPX status."""
        w = self.fold_w
        h = max(1, math.ceil(len(self.stream) / w))
        score = self.current_score()
        stats = self.stream_stats()

        lines = [
            f"W:{w}  H:{h}  SCORE:{score:.3f}  "
            f"PALETTE:{PALETTE_NAMES[self.palette_id]}  "
            f"MODE:{self.render_mode}",
            f"LEN:{len(self.stream)}  ENTROPY:{stats['entropy']:.2f}  "
            f"AVG_D:{stats['avg_density']:.2f}  "
            f"UNIQUE:{stats['unique_masks']}",
        ]

        # QIPX status line (if QIPX node is attached)
        if _qipx_node is not None:
            qipx_line = _qipx_node.hud_status()
            lines.append(qipx_line)

        # Entanglement status line (if Entanglement controller is attached)
        if _entangle_ctrl is not None:
            ent_line = _entangle_ctrl.hud_status()
            lines.append(ent_line)

        # Civilization status line
        if _civ_ctrl is not None and _civ_ctrl.state.enabled:
            net_offset += 1
            civ_line = _civ_ctrl.hud_status()
            lines.append(civ_line)

        # Determine HUD line offsets
        net_offset = 0  # index offset for network lines
        if _qipx_node is not None:
            net_offset += 1
        if _entangle_ctrl is not None:
            net_offset += 1
        if _civ_ctrl is not None and _civ_ctrl.state.enabled:
            net_offset += 1

        y = 4
        for i, line in enumerate(lines):
            # Network lines (QIPX, Entanglement, Civ) get distinct colours
            if i >= 2 and i < 2 + net_offset:
                if i == 2 and _qipx_node is not None:
                    color = (0, 200, 200)   # cyan for QIPX
                elif _entangle_ctrl is not None and i == 2 + (1 if _qipx_node is not None else 0):
                    color = (200, 120, 255)  # violet for Entanglement
                elif _civ_ctrl is not None and _civ_ctrl.state.enabled:
                    color = (255, 200, 100)  # gold for Civilization
                else:
                    color = (200, 120, 255)  # violet for Entanglement
            else:
                color = (0, 200, 0)         # green for local lines
            txt = self.font_small.render(line, True, color)
            bg = pygame.Surface((txt.get_width() + 8, txt.get_height() + 2),
                                pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            surface.blit(bg, (4, y))
            surface.blit(txt, (8, y))
            y += 16

        # Resonance lock flash
        if self.lock_flash > 0:
            lock_txt = self.font_title.render(
                f"LOCK  W={w}  H={h}", True, (255, 255, 0))
            lx = SCREEN_W // 2 - lock_txt.get_width() // 2
            ly = SCREEN_H // 2 - lock_txt.get_height() // 2
            bg2 = pygame.Surface(
                (lock_txt.get_width() + 20, lock_txt.get_height() + 12),
                pygame.SRCALPHA)
            bg2.fill((0, 0, 0, 180))
            surface.blit(bg2, (lx - 10, ly - 6))
            surface.blit(lock_txt, (lx, ly))
            self.lock_flash -= 1

        # QIPX diagnostics overlay (Alt+Q)
        if _qipx_node is not None and _qipx_node.show_diagnostics:
            diag_lines = _qipx_node.diagnostics_lines()
            # Append entanglement diagnostics if available
            if _entangle_ctrl is not None:
                diag_lines.append("")
                ent_diag = _entangle_ctrl.diagnostics_lines()
                diag_lines.extend(ent_diag)
            # Append civilization diagnostics if available
            if _civ_ctrl is not None and _civ_ctrl.state.enabled:
                diag_lines.append("")
                civ_diag = _civ_ctrl.diagnostics_lines()
                diag_lines.extend(civ_diag)
            dy = 80
            for dl in diag_lines:
                dt = self.font_small.render(dl, True, (85, 255, 255))
                dbg = pygame.Surface(
                    (dt.get_width() + 8, dt.get_height() + 2),
                    pygame.SRCALPHA)
                dbg.fill((0, 0, 0, 180))
                surface.blit(dbg, (4, dy))
                surface.blit(dt, (8, dy))
                dy += 14
                if dy > SCREEN_H - 60:
                    break

        # QIPX peer sidebar (right side)
        if _qipx_node is not None and _qipx_node.enabled:
            peer_lines = _qipx_node.peer_list_compact()
            if peer_lines:
                py = 60
                header = self.font_small.render("QIPX PEERS", True, (0, 200, 200))
                hbg = pygame.Surface(
                    (header.get_width() + 8, header.get_height() + 2),
                    pygame.SRCALPHA)
                hbg.fill((0, 0, 0, 160))
                surface.blit(hbg, (SCREEN_W - 160, py))
                surface.blit(header, (SCREEN_W - 156, py))
                py += 16
                for pl in peer_lines[:12]:
                    pt = self.font_small.render(pl, True, (150, 150, 150))
                    pbg = pygame.Surface(
                        (pt.get_width() + 8, pt.get_height() + 2),
                        pygame.SRCALPHA)
                    pbg.fill((0, 0, 0, 120))
                    surface.blit(pbg, (SCREEN_W - 160, py))
                    surface.blit(pt, (SCREEN_W - 156, py))
                    py += 14

        # Civilization overlay (C key)
        if _civ_ctrl is not None and _civ_ctrl.state.show_overlay:
            civ_diag = _civ_ctrl.diagnostics_lines()
            dy = 100
            for dl in civ_diag:
                dt = self.font_small.render(dl, True, (255, 200, 100))
                dbg = pygame.Surface(
                    (dt.get_width() + 8, dt.get_height() + 2),
                    pygame.SRCALPHA)
                dbg.fill((0, 0, 0, 180))
                surface.blit(dbg, (4, dy))
                surface.blit(dt, (8, dy))
                dy += 14
                if dy > SCREEN_H - 60:
                    break

            # Civilization peer panel (left side)
            civ_peer_lines = _civ_ctrl.hud_peer_panel(max_lines=6)
            cpy = 100
            for cl in civ_peer_lines:
                ct = self.font_small.render(cl, True, (200, 180, 100))
                cbg = pygame.Surface(
                    (ct.get_width() + 8, ct.get_height() + 2),
                    pygame.SRCALPHA)
                cbg.fill((0, 0, 0, 160))
                surface.blit(cbg, (SCREEN_W - 250, cpy))
                surface.blit(ct, (SCREEN_W - 246, cpy))
                cpy += 14

            # Reality panel (bottom-left)
            reality_lines = _civ_ctrl.hud_reality_panel()
            rpy = SCREEN_H - 80 - len(reality_lines) * 14
            for rl in reality_lines:
                rt = self.font_small.render(rl, True, (200, 200, 100))
                rbg = pygame.Surface(
                    (rt.get_width() + 8, rt.get_height() + 2),
                    pygame.SRCALPHA)
                rbg.fill((0, 0, 0, 160))
                surface.blit(rbg, (4, rpy))
                surface.blit(rt, (8, rpy))
                rpy += 14

        # Key help at bottom
        help_lines = [
            "[A/D] width  [P] palette  [M] mode  [G] ghost  [SPACE] scan  "
            "[R] lock  [Q] qipx  [E] entangle  [C] civ  [V] vote  [ESC] quit",
        ]
        for hl in help_lines:
            ht = self.font_small.render(hl, True, (120, 120, 120))
            hbg = pygame.Surface((ht.get_width() + 8, ht.get_height() + 2),
                                 pygame.SRCALPHA)
            hbg.fill((0, 0, 0, 140))
            surface.blit(hbg, (4, SCREEN_H - 20))
            surface.blit(ht, (8, SCREEN_H - 18))

    # ── master draw ──────────────────────────────────────────

    def draw(self):
        dispatch = {
            0: self.render_native,
            1: self.render_fold,
            2: self.render_scaled,
            3: self.render_density_map,
            4: self.render_resonance_analyzer,
            5: self.render_paradox,
        }
        renderer = dispatch.get(self.render_mode, self.render_native)
        renderer(self.screen)

        if self.show_ghost and self.render_mode in (0, 1):
            self.render_ghost(self.screen)

        self.draw_hud(self.screen)
        pygame.display.flip()
        self.needs_redraw = False

    def stream_stats(self) -> dict:
        """Lazy stats cache."""
        if not hasattr(self, '_stats') or self._stats_len != len(self.stream):
            import bs_engine
            self._stats = bs_engine.stream_stats(self.stream)
            self._stats_len = len(self.stream)
        return self._stats

    def invalidate_stats(self):
        self._stats_len = -1

    def quit(self):
        self.running = False


# Late import to avoid circular – only used in render_scaled
import math
```

----------------------------------------

### File: `bs_routes.py`

**Path:** `./bs_routes.py`
**Extension:** `.py`
**Size:** 9,677 bytes (9.45 KB)

```py
"""
BS_ROUTES  -  Routing table and TTL forwarding for Entanglement.
              Part of the BS-TOS-IPX Entanglement Layer.

Entanglement is a routing layer.  It routes:
  state, votes, locks, heartbeat, merge candidates,
  influence commands, crystal updates, rollback alerts.

Routing score:
  RouteScore(peer) =
      0.35 * Trust
    + 0.25 * HeartbeatCoherence
    + 0.20 * LowLatency
    + 0.20 * Recentness

TTL: each routed packet has a max hops counter.
  On forward: ttl -= 1; if ttl <= 0: drop.

Default max TTL: 3 hops.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs_qipx_peer import PEER_TIMEOUT


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_TTL = 3
MAX_TTL = 10
ROUTE_EXPIRE = 30.0  # seconds before a route expires

# Routing score weights
W_ROUTE_TRUST = 0.35
W_ROUTE_HEARTBEAT = 0.25
W_ROUTE_LATENCY = 0.20
W_ROUTE_RECENT = 0.20

# Estimated latency for peers on same LAN
LAN_LATENCY_MS = 5.0


# ═══════════════════════════════════════════════════════════════
#  Route Entry
# ═══════════════════════════════════════════════════════════════

@dataclass
class RouteEntry:
    """A route to a remote node."""
    node_id: str
    addr: Tuple[str, int]
    last_seen: float = 0.0
    hops: int = 1
    trust: float = 0.5
    latency_ms: float = LAN_LATENCY_MS
    heartbeat_coherence: float = 0.5
    role: str = "retina_node"
    packets_forwarded: int = 0


# ═══════════════════════════════════════════════════════════════
#  Routing Table
# ═══════════════════════════════════════════════════════════════

class RoutingTable:
    """Manages routing entries and computes best routes for packets."""

    def __init__(self, max_routes: int = 64):
        self._routes: Dict[str, RouteEntry] = {}
        self.max_routes = max_routes
        self._total_forwarded = 0

    # ── Route management ──────────────────────────────────────

    def update_route(self, node_id: str, addr: Tuple[str, int],
                     trust: float = 0.5, hops: int = 1,
                     latency_ms: float = LAN_LATENCY_MS,
                     heartbeat_coherence: float = 0.5,
                     role: str = "retina_node"):
        """Update or create a route entry."""
        entry = self._routes.get(node_id)
        if entry is None:
            if len(self._routes) >= self.max_routes:
                self._evict_worst()
            entry = RouteEntry(
                node_id=node_id,
                addr=addr,
            )
            self._routes[node_id] = entry
        entry.last_seen = time.time()
        entry.trust = trust
        entry.hops = hops
        entry.latency_ms = latency_ms
        entry.heartbeat_coherence = heartbeat_coherence
        entry.role = role

    def remove_route(self, node_id: str):
        self._routes.pop(node_id, None)

    def get_route(self, node_id: str) -> Optional[RouteEntry]:
        return self._routes.get(node_id)

    def expire(self) -> int:
        """Remove expired routes.  Returns count of evicted."""
        now = time.time()
        dead = [
            nid for nid, r in self._routes.items()
            if now - r.last_seen > ROUTE_EXPIRE
        ]
        for nid in dead:
            del self._routes[nid]
        return len(dead)

    def sync_from_peers(self, peers: Dict):
        """Sync routes from QIPX peer table.

        Parameters
        ----------
        peers : dict
            QipxNode.peers dict (node_id → PeerState)
        """
        for nid, peer in peers.items():
            if peer.is_alive():
                self.update_route(
                    node_id=nid,
                    addr=peer.addr,
                    trust=peer.trust,
                    hops=1,
                    role="retina_node",
                )
        # Remove routes for dead peers
        alive_ids = set(peers.keys())
        for nid in list(self._routes.keys()):
            if nid not in alive_ids and nid != "BROADCAST":
                self.remove_route(nid)

    # ── Route scoring ─────────────────────────────────────────

    def route_score(self, node_id: str) -> float:
        """Compute routing score for a peer.

        RouteScore = 0.35*Trust + 0.25*HeartbeatCoherence
                   + 0.20*LowLatency + 0.20*Recentness
        """
        entry = self._routes.get(node_id)
        if entry is None:
            return 0.0

        # Trust component (already 0..1)
        trust_score = entry.trust

        # Heartbeat coherence (already 0..1)
        hb_score = entry.heartbeat_coherence

        # Low latency: inverse, lower latency = higher score
        lat_score = 1.0 / (1.0 + entry.latency_ms / 50.0)

        # Recentness: how recently the peer was seen
        age = time.time() - entry.last_seen
        recent_score = max(0.0, 1.0 - age / ROUTE_EXPIRE)

        return (W_ROUTE_TRUST * trust_score
                + W_ROUTE_HEARTBEAT * hb_score
                + W_ROUTE_LATENCY * lat_score
                + W_ROUTE_RECENT * recent_score)

    def best_route(self) -> Optional[RouteEntry]:
        """Return the route with the highest score."""
        if not self._routes:
            return None
        best_id = max(self._routes, key=self.route_score)
        return self._routes[best_id]

    def best_route_for_consensus(self) -> Optional[RouteEntry]:
        """Route packets to the peer most likely to improve majority agreement.

        Heuristic: highest trust + heartbeat coherence.
        """
        if not self._routes:
            return None
        def consensus_score(nid):
            r = self._routes[nid]
            return 0.6 * r.trust + 0.4 * r.heartbeat_coherence
        best_id = max(self._routes, key=consensus_score)
        return self._routes[best_id]

    # ── TTL forwarding ────────────────────────────────────────

    def can_forward(self, pkt: dict) -> bool:
        """Check if a packet can be forwarded (has remaining TTL)."""
        ttl = pkt.get("ttl", DEFAULT_TTL)
        return ttl > 0

    def prepare_forward(self, pkt: dict) -> Optional[dict]:
        """Decrement TTL and prepare packet for forwarding.

        Returns the modified packet, or None if TTL expired.
        """
        ttl = pkt.get("ttl", DEFAULT_TTL)
        if ttl <= 0:
            return None
        forwarded = dict(pkt)
        forwarded["ttl"] = ttl - 1
        forwarded["hops"] = pkt.get("hops", 0) + 1
        return forwarded

    def forward_packet(self, pkt: dict, target_id: str) -> bool:
        """Forward a packet to a specific route.

        Returns True if the packet was forwarded (caller must actually send).
        """
        route = self.get_route(target_id)
        if route is None:
            return False
        prepared = self.prepare_forward(pkt)
        if prepared is None:
            return False
        route.packets_forwarded += 1
        self._total_forwarded += 1
        return True

    def forward_to_all(self, pkt: dict, exclude: str = "") -> List[Tuple[str, dict]]:
        """Forward a packet to all alive routes.

        Returns list of (node_id, prepared_packet) tuples.
        """
        results = []
        for nid, route in self._routes.items():
            if nid == exclude:
                continue
            prepared = self.prepare_forward(pkt)
            if prepared is None:
                continue
            route.packets_forwarded += 1
            self._total_forwarded += 1
            results.append((nid, prepared))
        return results

    # ── Internal ──────────────────────────────────────────────

    def _evict_worst(self):
        """Remove the route with the lowest score."""
        if not self._routes:
            return
        worst_id = min(self._routes, key=self.route_score)
        del self._routes[worst_id]

    # ── Queries ───────────────────────────────────────────────

    def count(self) -> int:
        return len(self._routes)

    def alive_count(self) -> int:
        now = time.time()
        return sum(1 for r in self._routes.values()
                   if now - r.last_seen < ROUTE_EXPIRE)

    def all_routes(self) -> Dict[str, RouteEntry]:
        return dict(self._routes)

    def one_line(self) -> str:
        return (f"routes={self.count()} "
                f"forwarded={self._total_forwarded}")
```

----------------------------------------

### File: `requirements.txt`

**Path:** `./requirements.txt`
**Extension:** `.txt`
**Size:** 37 bytes (0.04 KB)

```txt
pygame>=2.5
Pillow>=10.0
numpy>=1.24
```

----------------------------------------

## Directory: `logs`

