#!/usr/bin/env python3
"""Audit external HDF5 headers/attributes without importing Torch or reading bags."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
from prepare_cptac_brca import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("preserve prior audit")
    with args.manifest.open() as handle:rows=list(csv.DictReader(handle))
    records={}
    for row in rows:
        for column,mag,px in (("feature__conch_v1_5x",5,512),("feature__conch_v1_10x",10,256)):
            path=Path(row[column])
            attributes=subprocess.check_output(["h5dump","-a","/features/encoder","-a","/coords/patch_size",
                "-a","/coords/target_magnification",str(path)],text=True,timeout=30)
            def attr(name):
                match=re.search(r'ATTRIBUTE "'+name+r'".*?\(0\):\s*("[^"]*"|[^\s}]+)',attributes,re.S)
                if not match:raise ValueError(f"missing attribute {name}: {path}")
                return match.group(1).strip('"')
            if attr("encoder")!="conch_v1" or int(attr("patch_size"))!=px or float(attr("target_magnification"))!=mag:
                raise ValueError(f"incompatible feature attributes: {path}")
            header=subprocess.check_output(["h5dump","-H","-d","features",str(path)],text=True,timeout=30)
            shape=re.search(r'DATASPACE\s+SIMPLE\s*\{\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',header)
            if not shape or int(shape[1])<1 or int(shape[2])!=512:raise ValueError(f"invalid feature shape: {path}")
            stat=path.stat()
            records[str(path)]={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"patches":int(shape[1]),
                "width":512,"encoder":"conch_v1","magnification":mag,"patch_size":px}
    report={"status":"passed","manifest_sha256":sha(args.manifest),"files":records,
            "historical_extraction_checkpoint_hash_recorded":False,
            "provenance_boundary":"CONCH-v1 identity, width and geometry verified from HDF5 attributes; extraction checkpoint digest absent",
            "payload_finiteness":"checked at inference, not by this header-only audit","auditor_sha256":sha(__file__)}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","files":len(records),"slides":len(rows)}))


if __name__=="__main__":main()
