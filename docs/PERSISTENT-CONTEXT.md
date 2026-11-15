# Conversations on the SSD (S25, 2026-11-10)

A prompt token costs about 3 ms of GPU to compute and 19 KB to keep. The drive reads 6 GB/s. So the fast way to get
a context back is to read it, and the engine now does: `--kv-store DIR` keeps conversations on the SSD, and a
restart, an OOM kill or a switch between conversations costs a read instead of a prefill.

## Measured (console, 3.18 pack; `scripts/console/session_probe.py`, `session_e2e.py`)

| case | before | now |
|---|---|---|
| 30,018-token prompt, engine killed with SIGKILL, restarted, same prompt | 103 s (prefilled again) | **0.31 s** to read 30,009 tokens back, 2.5 s to the end of a 32-token answer; the answer is token-identical (exact routing) |
| 120,018-token prompt, same test (serving flags) | 418 s | **0.96 s** to read 120,009 tokens back (2.4 GB), 3.1 s to the end of the answer |
| a prompt killed 60% of the way through its prefill | starts from zero | resumes from the last periodic save: killed at 72,223 of 120,020 tokens, the restart continued from 65,536 (0.56 s to read) and finished the rest in 208 s instead of 418 |
| back to an earlier conversation after another one used the engine | prefilled again (two snapshots in memory, state only) | read back: 0.33 s at 30k, 1.3 s at 120k |
| a request that branches from the live conversation (edit, regenerate) | prefilled from the last in-memory snapshot | only the 118 MB state is read (0.035 s); the rows still in memory are kept |
| HTTP chat, 12k-token document, whole unit killed with SIGKILL between turns 2 and 3 | 37-39 s | **3.8-4.8 s** for turn 3 (`resume=disk`, 12,267-12,351 tokens reused; most of it is the cold expert pool after the restart) |
| cost of saving after a request | - | 3-50 ms typical (5 MB of new rows; 118 MB when a new resume point is written), after the reply has been sent |
| cost of the periodic save during a long prefill | - | 0.13 s every 8,192 tokens (S30; was 16,384) |

## What is stored

One directory per conversation under the store:

| file | content | size |
|---|---|---|
| `kv` | every per-position cache of the 12 attention layers and the MTP layer: K and V (int8), the indexer's raw keys and pooled block keys; rows `[0, n)` in blocks of 4,096 positions | 19,136 B per token (2.3 GB at 120k); S30b 15,808; S32 (`--kv k4v4`, fp16 pooled keys) 7,696 B (0.92 GB at 120k) |
| `rs.<pos>` | the recurrent state at position `pos`: GatedDeltaNet state and convolution tails of 36 layers, the PLE convolution tail | 118 MB each, at most 4 per conversation |
| `meta` | the tokens the rows were computed from, the resume positions, a format fingerprint (pack, KV format, row sizes) | 4 B per token |

`DIR/turns/` holds the HTTP front end's last 8 finished turns (prompt text, prompt ids, generated ids). Without them a
restarted front end would re-tokenise the assistant's earlier answers from text, the ids would differ from the ones
the engine generated, and the stored conversation would only match up to the first answer.

## How a request uses it

1. The engine picks its resume point in memory as before (live state, else an in-memory snapshot).
2. `SessionStore::best()` looks for the longest stored resume point whose tokens are a prefix of the request, in any
   conversation. If it is at least 256 tokens further than what memory offers, the rows are read into the KV buffers
   (rows the engine already holds for the same tokens are skipped), the recurrent state is read, and the engine
   continues from there. The reply reports `resume=disk` and `load_s`.
3. After the reply is sent, the rows written since the last save go to the drive, plus any new in-memory snapshot
   (the front end asks for one at the end of the system/tools block and one at the end of the prompt) and the end
   state if it is 256 tokens past the last resume point. During a long prefill the same happens every 8,192 tokens.
   S30: the resume points kept are the first (system block), one per 8,192 tokens of the conversation, and the
   newest two (at most 48, 118 MB each). Until S30 only the first and the 3 newest were kept: an agent client that
   pruned an old tool result 65k tokens back found no point between the system block and the prune, re-prefilled
   about 160k tokens (11 minutes) and timed out at 300 s, three times in one hour of served use.
4. A request continues a stored conversation when it shares more than the system/tools block with it and at least
   half of either side; otherwise it starts a new one. Least recently used conversations are deleted above
   `--kv-store-gb` (default 16) or 16 conversations.

Crash safety: `meta` only ever describes rows that are already on the device (O_DIRECT, `fdatasync`, then
`meta.tmp` renamed). Before rows that `meta` describes are overwritten (a conversation that was edited and
continues differently), `meta` is cut back first. A failed load drops that conversation and resets the engine; a
failed save (full drive) leaves the conversation in memory only. Neither stops the server.

## Flags

| flag | default | meaning |
|---|---|---|
| `--kv-store DIR` | off | enable; the served configuration uses `/home/steam/ps5lm/sessions` |
| `--kv-store-gb N` | 16 | size cap |
| `--kv-store-every T` | 8192 | periodic save during a prefill (0 = only after the request) |
| `--kv-store-gap T` | 8192 | spacing of the resume points kept along a conversation (S30) |
| `--kv-store-min T` | 256 | load from the drive only if it saves at least this many tokens |
| `--ctx-save DIR`, `--ctx-load DIR` | - | CLI test: store the prompt's state / resume a prompt from it (`smoke.sh x --ctx-load DIR` must print PASS) |

## Limits

- The state is bound to the pack and the KV format. A conversation stored under another pack is kept for the size
  accounting and never loaded.
- With cache-aware routing the engine is not deterministic run to run, so a resumed conversation continues from
  exactly the stored state but a fresh prefill of the same prompt would not have produced bit-identical rows. With
  exact routing the resumed answer is token-identical (checked).
- Each request writes 5-240 MB. At 1,000 requests a day that is up to 240 GB per day on a 512 GB drive rated for
  about 300 TB written: years, but not unlimited. `--kv-store-every 0` and a larger `live_min` reduce it.
- Reading is 2.3 GB/s end to end (drive to bounce buffer to staging buffer to the GPU copy), a third of what the
  drive can deliver: a 120k-token context takes about a second. Reading straight into pages the GPU can see would
  halve that. Not built.
- A turn that the client cut off (not `stop` or `length`) is not remembered by the front end; the next request then
  resumes from the previous prompt's end.
