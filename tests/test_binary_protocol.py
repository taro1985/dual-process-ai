"""
Tests & Benchmark for Binary-Native IPC and Bitwise Latent Protocol
"""
import pytest
import time
import json
from dpai.binary_protocol import (
    BinaryActionPacket,
    BitwiseLatentMatcher,
    ACTION_GET_STATUS,
    ACTION_HABIT_REFLEX,
    FLAG_HABIT_HIT,
)


def test_binary_packet_pack_unpack():
    """Verify 16-byte fixed-width packet serialization and zero loss."""
    packet = BinaryActionPacket(
        action_code=ACTION_GET_STATUS,
        confidence=0.9542,
        flags=FLAG_HABIT_HIT,
        latent_hash=0xFEEDFACECAFEBEEF,
    )

    raw_bytes = packet.pack()
    assert len(raw_bytes) == 16  # Exactly 16 bytes

    unpacked = BinaryActionPacket.unpack(raw_bytes)
    assert unpacked.action_code == ACTION_GET_STATUS
    assert unpacked.action_name == "get_status"
    assert round(unpacked.confidence, 3) == 0.954
    assert unpacked.is_habituated is True
    assert unpacked.latent_hash == 0xFEEDFACECAFEBEEF


def test_bitwise_latent_simhash_matching():
    """Verify 64-bit bitwise Hamming distance matching."""
    matcher = BitwiseLatentMatcher()

    matcher.register_habit(
        query="PostgreSQLの起動コマンドを教えて",
        response="sudo systemctl start postgresql",
    )

    # Exact match
    res1 = matcher.match_fast("PostgreSQLの起動コマンドを教えて")
    assert res1 is not None
    assert res1["response"] == "sudo systemctl start postgresql"
    assert res1["hamming_distance"] == 0  # 0 bit difference!
    assert res1["similarity_score"] == 1.0

    # Very similar query (minor word variation)
    res2 = matcher.match_fast("PostgreSQLの起動コマンド")
    assert res2 is not None
    assert res2["response"] == "sudo systemctl start postgresql"
    assert res2["hamming_distance"] <= 14  # close hamming distance

    # Negation safety guard: "教えないで" must NEVER trigger positive habit reflex
    res_neg = matcher.match_fast("PostgreSQLの起動コマンドを教えないで")
    assert res_neg is None  # Blocked!

    # Side-effect mutation guard: mutating action without allow_mutation must NEVER reflex
    matcher.register_habit(
        query="Dockerコンテナを再起動して",
        response="docker restart vaio-core",
        action="docker_restart",
        is_read_only=False,
    )
    res_side_effect = matcher.match_fast("Dockerコンテナを再起動して", allow_mutation=False)
    assert res_side_effect is None  # Blocked from reflexive auto-execution!


def test_benchmark_3way_dataclass_vs_binary_vs_json():
    """
    Empirical 3-way benchmark:
    1. Direct Python Object (Dataclass slots=True) - True intra-process baseline
    2. 16-Byte Binary Packet (C-struct pack/unpack) - Inter-process IPC baseline
    3. JSON String Serialization (dumps/loads) - Traditional text API baseline
    """
    from dataclasses import dataclass

    @dataclass(slots=True)
    class DecisionPayload:
        action_code: int
        confidence: float
        flags: int
        latent_hash: int

    iterations = 20000

    # 1. Intra-Process Baseline: Dataclass Direct Passing (Zero Serialization)
    t0 = time.perf_counter()
    for _ in range(iterations):
        obj = DecisionPayload(ACTION_GET_STATUS, 0.95, FLAG_HABIT_HIT, 0xFEEDFACE)
        _ = obj.action_code
    time_dataclass_ms = (time.perf_counter() - t0) * 1000

    # 2. 16-Byte Machine Binary Packet (pack/unpack)
    t1 = time.perf_counter()
    for _ in range(iterations):
        pkt = BinaryActionPacket(
            action_code=ACTION_GET_STATUS,
            confidence=0.95,
            flags=FLAG_HABIT_HIT,
            latent_hash=0xFEEDFACE,
        )
        raw_bytes = pkt.pack()
        unpacked = BinaryActionPacket.unpack(raw_bytes)
        _ = unpacked.action_code
    time_binary_ms = (time.perf_counter() - t1) * 1000

    # 3. Traditional Human-Readable JSON Text Serialization
    t2 = time.perf_counter()
    for _ in range(iterations):
        payload = {
            "source": "Jev Classifier",
            "action": "get_status",
            "confidence": 0.95,
            "flags": 2,
            "hash": 0xFEEDFACE,
        }
        text_data = json.dumps(payload)
        parsed = json.loads(text_data)
        _ = parsed["action"]
    time_json_ms = (time.perf_counter() - t2) * 1000

    print(f"\n📊 3-Way Benchmark Results ({iterations:,} iterations):")
    print(f"   1. Python Dataclass Direct  : {time_dataclass_ms:.2f} ms ({time_dataclass_ms/iterations*1000:.3f} µs/op) [True intra-process baseline]")
    print(f"   2. 16-Byte Binary Packet    : {time_binary_ms:.2f} ms ({time_binary_ms/iterations*1000:.3f} µs/op) [IPC/C-struct baseline]")
    print(f"   3. Human Text/JSON API      : {time_json_ms:.2f} ms ({time_json_ms/iterations*1000:.3f} µs/op) [Web/REST baseline]")

    # Dataclass is strictly faster than serialization
    assert time_dataclass_ms < time_binary_ms
    # 16-byte fixed binary packet payload is significantly smaller than text JSON (~7x smaller)
    assert len(raw_bytes) == 16
    assert len(raw_bytes) < len(text_data)
    # Both IPC binary packet and JSON are in single-digit microseconds (< 50µs/op)
    assert (time_binary_ms / iterations * 1000) < 50.0


def test_router_binary_fast_path():
    """Verify DualProcessRouter binary packet routing and habit integration."""
    from dpai.router import DualProcessRouter

    router = DualProcessRouter(threshold=0.80)

    # 1. Classify directly to 16-byte packet
    pkt = router.classify_packet("ステータス教えて")
    assert isinstance(pkt, BinaryActionPacket)
    assert pkt.action_name == "get_status"
    assert pkt.confidence >= 0.90

    # 2. Process binary packet directly (Pack -> Process -> Unpack)
    raw_in = pkt.pack()
    raw_out = router.process_binary(raw_in)
    assert len(raw_out) == 16

    out_pkt = BinaryActionPacket.unpack(raw_out)
    assert out_pkt.action_name == "get_status"
    assert out_pkt.confidence == pkt.confidence

    # 3. Habituation with binary matcher
    router.habituate(
        query="PostgreSQLの死活確認",
        response="pg_isready -h localhost",
    )
    # Binary packet should reflect habit reflex
    habit_pkt = router.classify_packet("PostgreSQLの死活確認")
    assert habit_pkt.is_habituated is True
    assert habit_pkt.action_name == "habituated_response"

