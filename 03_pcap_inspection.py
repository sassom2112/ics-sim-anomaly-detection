"""
PCAP Inspection — Raw Modbus/TCP Packet Analysis

Reads the first N packets from the ICSSim traffic capture and decodes
Modbus/TCP MBAP headers so you can see the industrial protocol traffic
that the network-layer Dataset.csv was derived from.

This is an exploratory tool; it does not modify any files.
"""

import os
import kagglehub
from scapy.all import rdpcap, IP, TCP, UDP

dataset_dir = kagglehub.dataset_download("alirezadehlaghi/icssim")
pcap_path = os.path.join(dataset_dir, "traffic.pcap")

PACKET_LIMIT = 1000

print(f"Loading first {PACKET_LIMIT:,} packets from {pcap_path} …")
packets = rdpcap(pcap_path, count=PACKET_LIMIT)
print(f"Loaded {len(packets):,} packets.\n")

# ─── Protocol summary ─────────────────────────────────────────────────────────
proto_counts = {"TCP": 0, "UDP": 0, "other": 0}
for pkt in packets:
    if pkt.haslayer(TCP):
        proto_counts["TCP"] += 1
    elif pkt.haslayer(UDP):
        proto_counts["UDP"] += 1
    else:
        proto_counts["other"] += 1

print("Protocol distribution:")
for proto, count in proto_counts.items():
    print(f"  {proto:6s}: {count:,}")

# ─── Decode first 5 packets with Modbus/TCP payloads ─────────────────────────
print("\n── First 5 packets with payload ──")
shown = 0
for i, pkt in enumerate(packets):
    if shown >= 5:
        break
    if not (pkt.haslayer(IP) and pkt.haslayer(TCP)):
        continue
    if not pkt[TCP].payload:
        continue

    src  = f"{pkt[IP].src}:{pkt[TCP].sport}"
    dst  = f"{pkt[IP].dst}:{pkt[TCP].dport}"
    raw  = bytes(pkt[TCP].payload)
    hex_ = raw[:16].hex()

    # Modbus/TCP MBAP header: [TransID 2B][ProtoID 2B][Length 2B][UnitID 1B]
    mbap = f"{hex_[0:4]} {hex_[4:8]} {hex_[8:12]} {hex_[12:14]}"
    pdu  = " ".join(hex_[j:j+4] for j in range(14, len(hex_), 4))

    print(f"\n[{i}] {src} → {dst}")
    print(f"     MBAP: {mbap}  |  PDU: {pdu}")
    shown += 1
