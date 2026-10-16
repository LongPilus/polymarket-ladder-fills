"""Fetch the on-chain fill events of Polymarket's daily Bitcoin ladder trades from Polygon
and push them, chunk by chunk, to a Hugging Face dataset.

Cloud version of scripts/fetch_strip_fills.py in the (private) paper repository; same
chunks, same output. The transaction list (one Polygon transaction per taker fill, in
trade-time order) is read from the dataset:

    <PREFIX>/tx_list.parquet          column `tx`; chunk k = rows 20,000*k to 20,000*(k+1)

and every chunk is written to it as

    <PREFIX>/logs/part_<k>.parquet    tx, block, logIndex, address, topic0-topic3, data:
                                      the receipt's OrderFilled, OrdersMatched (both
                                      exchange versions) and FeeRefunded events, as returned
    <PREFIX>/failed/part_<k>.txt      transactions no endpoint answered

Chunks already in the dataset are skipped, so a rerun resumes. Each shard takes the chunks
with k % shards == shard; GitHub runners have separate IP addresses, so the free
endpoints' per-address limits apply to each shard separately.

    HF_TOKEN=... python fetch_fills.py --shard 0 --shards 10
    python fetch_fills.py --chunks 54 --no-upload      # test: write to ./out only
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = os.environ.get("HF_REPO", "SmazakVMense/disertacka-data")
PREFIX = "raw/polymarket_btc_strips/fills_onchain_2026-10-15"
CHUNK = 20_000
WORKERS = 32
STOP_AFTER = 5.2 * 3600  # GitHub ends a job at 6 hours; start no chunk after this
RPCS = ["https://polygon.drpc.org", "https://polygon.gateway.tenderly.co",
        "https://rpc-mainnet.matic.quiknode.pro"]
TOPICS = {
    "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6",  # OrderFilled v1
    "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee",  # OrderFilled v2
    "0x63bf4d16b7fa898ef4c4b2b6d90fd201e9c56313b65638af6088d149d2ce956c",  # OrdersMatched v1
    "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c",  # OrdersMatched v2
    "0xb608d2bf25d8b4b744ba23ce2ea9802ea955e216c064a62f42152fbf98958d24",  # FeeRefunded
}
COLS = ["tx", "block", "logIndex", "address", "topic0", "topic1", "topic2", "topic3", "data"]


def receipt(i: int, tx: str, rounds: int = 3):
    """Receipt of `tx`, trying each endpoint in turn, `rounds` times; None if all fail."""
    for a in range(rounds * len(RPCS)):
        try:
            r = requests.post(RPCS[(i + a) % len(RPCS)], timeout=30, json={
                "jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionReceipt",
                "params": [tx]}).json().get("result")
            if r:
                return r
        except (requests.RequestException, ValueError):
            pass
        if (a + 1) % len(RPCS) == 0:
            time.sleep(2 * (a + 1) // len(RPCS))
    return None


def rows(r) -> list[list]:
    out = []
    for lg in r["logs"]:
        tp = lg["topics"] + [None] * (4 - len(lg["topics"]))
        if tp[0] in TOPICS:
            out.append([r["transactionHash"], int(r["blockNumber"], 16), int(lg["logIndex"], 16),
                        lg["address"].lower(), *tp[:4], lg["data"]])
    return out


def fetch(txs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    with ThreadPoolExecutor(WORKERS) as ex:
        got = list(ex.map(receipt, range(len(txs)), txs))
    data = [row for r in got if r for row in rows(r)]
    failed = [tx for tx, r in zip(txs, got, strict=True) if not r]
    return pd.DataFrame(data, columns=COLS), failed


def push(api: HfApi, k: int, files: dict[str, Path]) -> None:
    ops = [CommitOperationAdd(f"{PREFIX}/{p}", str(f)) for p, f in files.items()]
    for a in range(8):  # parallel shards commit to one repo: retry on conflicts
        try:
            api.create_commit(REPO, ops, repo_type="dataset",
                              commit_message=f"Add on-chain fill events, chunk {k}")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  commit of chunk {k} failed ({type(e).__name__}: {e}); retry", flush=True)
            time.sleep(15 * (a + 1))
    raise RuntimeError(f"could not push chunk {k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--chunks", default="", help="comma-separated chunk numbers only")
    ap.add_argument("--no-upload", action="store_true", help="write to ./out, push nothing")
    a = ap.parse_args()
    t_start = time.time()
    api = HfApi()
    txs = pd.read_parquet(hf_hub_download(REPO, f"{PREFIX}/tx_list.parquet",
                                          repo_type="dataset")).tx.tolist()
    n = (len(txs) + CHUNK - 1) // CHUNK
    if a.chunks:
        todo = [int(k) for k in a.chunks.split(",")]
    else:
        done = {int(f.rsplit("_", 1)[1].split(".")[0])
                for f in api.list_repo_files(REPO, repo_type="dataset")
                if f.startswith(f"{PREFIX}/logs/part_") and "retry" not in f}
        todo = [k for k in range(n) if k % a.shards == a.shard and k not in done]
    print(f"{len(txs):,} transactions, {n} chunks; shard {a.shard}/{a.shards}: "
          f"{len(todo)} to do", flush=True)
    out = Path("out")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "failed").mkdir(parents=True, exist_ok=True)
    for k in todo:
        if time.time() - t_start > STOP_AFTER:
            print("time limit reached; rerun to continue", flush=True)
            break
        t0 = time.time()
        part = txs[k * CHUNK:(k + 1) * CHUNK]
        df, failed = fetch(part)
        files = {f"logs/part_{k:04d}.parquet": out / "logs" / f"part_{k:04d}.parquet",
                 f"failed/part_{k:04d}.txt": out / "failed" / f"part_{k:04d}.txt"}
        df.to_parquet(files[f"logs/part_{k:04d}.parquet"], index=False)
        files[f"failed/part_{k:04d}.txt"].write_text("\n".join(failed))
        if not a.no_upload:
            push(api, k, files)
        el = time.time() - t0
        print(f"chunk {k}: {len(part) - len(failed):,} of {len(part):,} receipts, "
              f"{len(df):,} events, {el:.0f}s ({len(part) / el:.0f} tx/s)", flush=True)


if __name__ == "__main__":
    main()
