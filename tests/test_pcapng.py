from scapy.all import PcapReader, PcapNgReader
import tempfile, os

with tempfile.NamedTemporaryFile(delete=False, suffix=".pcapng") as tmp:
    tmp_path = tmp.name

try:
    with open(tmp_path, "wb") as f:
        # Write a dummy pcapng header
        f.write(b'\x0a\x0d\x0d\x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    
    try:
        with PcapReader(tmp_path) as reader:
            pass
    except Exception as e:
        print(f"PcapReader error: {e}")
        
finally:
    os.remove(tmp_path)
