#!/usr/bin/env python3
"""Freeze public PDC histology + primary-tumor sample labels against TCIA WSIs.

No predicted labels, substring patient guesses, mixed histologies, normal
specimens, or unidentified specimens are admitted. Input downloads are kept
verbatim with query, version-catalog and content hashes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

ENDPOINT="https://pdc.cancer.gov/graphql"
CLINICAL_QUERY='{clinicalPerStudy(pdc_study_id:"PDC000120") {case_id case_submitter_id morphology primary_diagnosis disease_type primary_site}}'
BIOSPECIMEN_QUERY='{biospecimenPerStudy(pdc_study_id:"PDC000120") {case_submitter_id sample_submitter_id sample_type}}'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def histology(row):
    """Map explicit diagnoses only; morphology must agree or be unreported."""
    label={"Infiltrating duct carcinoma, NOS":"IDC",
           "Infiltrating lobular carcinoma, NOS":"ILC","Invasive lobular carcinoma":"ILC"}.get(row["primary_diagnosis"])
    if label is None:return None,"mixed_other_or_unreported_histology"
    expected="8500/3" if label=="IDC" else "8520/3"
    if row["morphology"] not in (expected,"Not Reported"):
        return None,"conflicting_morphology_or_noninvasive_code"
    return label,"eligible"


def specimen_matches(slide_id,sample_id):
    """Allow exact sample IDs and TCIA's explicitly disclosed _D<number> suffix."""
    return bool(sample_id and re.fullmatch(re.escape(sample_id)+r"(?:_D\d+)*",slide_id))


def read_api(path,key):
    value=json.loads(Path(path).read_text())
    if value.get("errors"):raise ValueError(f"API errors in {path}")
    return value["data"][key]


def write_csv(path,rows,fields):
    with path.open("x") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clinical",type=Path,required=True);p.add_argument("--biospecimen",type=Path,required=True)
    p.add_argument("--catalog",type=Path,required=True);p.add_argument("--tcia-manifest",type=Path,required=True)
    p.add_argument("--feature-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("use a fresh version; do not overwrite a frozen cohort")
    clinical=read_api(args.clinical,"clinicalPerStudy");biospecimens=read_api(args.biospecimen,"biospecimenPerStudy")
    catalog=read_api(args.catalog,"studyCatalog")
    versions=next(s["versions"] for s in catalog if s["pdc_study_id"]=="PDC000120")
    version=next(v for v in versions if v["is_latest_version"]=="yes")
    if version["study_id"]!="b91a0a02-f3a0-11ea-b1fd-0aad30af8a83":raise ValueError("review changed PDC study version")
    cases={r["case_submitter_id"]:r for r in clinical}
    if len(cases)!=len(clinical) or None in cases:raise ValueError("duplicate/missing clinical case IDs")
    by_case=defaultdict(list)
    for b in biospecimens:by_case[b["case_submitter_id"]].append(b)
    with args.tcia_manifest.open() as h:manifest=list(csv.DictReader(h))
    low=args.feature_root/"5x_512px_0px_overlap/features_conch_v1"
    high=args.feature_root/"10x_256px_0px_overlap/features_conch_v1"
    kept=[];audit=[];used_paths=set()
    for r in manifest:
        case=r["patient_id"];label=None;reason="no_clinical_annotation";sample=""
        if case in cases:
            label,reason=histology(cases[case])
        if label:
            matches=[b for b in by_case[case] if specimen_matches(r["slide_id"],b["sample_submitter_id"])]
            if not matches:reason="specimen_type_not_identified";label=None
            elif {b["sample_type"] for b in matches}!={"Primary Tumor"}:reason="non_tumor_or_conflicting_specimen";label=None
            elif len({b['sample_submitter_id'] for b in matches})!=1:raise ValueError("ambiguous PDC sample identity")
            else:sample=matches[0]["sample_submitter_id"]
        if label:
            aliases={r["slide_id"],case+"-"+r["slide_id"],Path(r["wsiimage_url"]).stem}
            matches=[a for a in aliases if (low/(a+".h5")).is_file() and (high/(a+".h5")).is_file()]
            if len(matches)>1:raise ValueError("multiple feature files for one TCIA slide; review explicitly")
            if not matches:reason="missing_dual_scale_features";label=None
            else:
                stem=matches[0]
                if stem in used_paths:raise ValueError("duplicate feature identity in TCIA manifest")
                used_paths.add(stem)
                kept.append({"slide_id":stem,"case_id":case,"label":label,"label_id":0 if label=="IDC" else 1,
                    "tcia_slide_id":r["slide_id"],"pdc_sample_submitter_id":sample,"sample_type":"Primary Tumor",
                    "feature__conch_v1_5x":str(low/(stem+".h5")),"feature__conch_v1_10x":str(high/(stem+".h5"))})
        audit.append({"case_id":case,"tcia_slide_id":r["slide_id"],"label":label or "", "reason":reason})
    if {r["label"] for r in kept}!={"IDC","ILC"}:raise ValueError("both histologies are required")
    args.output.mkdir(parents=True)
    inputs={"pdc_clinical.json":args.clinical,"pdc_biospecimen.json":args.biospecimen,
            "pdc_catalog.json":args.catalog,"tcia_manifest.csv":args.tcia_manifest}
    for name,path in inputs.items():shutil.copy2(path,args.output/name)
    kept.sort(key=lambda r:(r["case_id"],r["slide_id"]))
    write_csv(args.output/"manifest.csv",kept,list(kept[0]))
    write_csv(args.output/"slide_audit.csv",audit,["case_id","tcia_slide_id","label","reason"])
    clinical_audit=[{"case_id":r["case_submitter_id"],"primary_diagnosis":r["primary_diagnosis"],
                     "morphology":r["morphology"],"label":histology(r)[0] or "", "reason":histology(r)[1]} for r in clinical]
    write_csv(args.output/"clinical_labels.csv",clinical_audit,list(clinical_audit[0]))
    metadata={"created_at":datetime.datetime.now().astimezone().isoformat(),"endpoint":ENDPOINT,
              "clinical_query":CLINICAL_QUERY,"biospecimen_query":BIOSPECIMEN_QUERY,"study":version,
              "api_version_policy":"pdc_study_id endpoint served latest; catalog v2 recorded and raw responses content-pinned",
              "label_policy":"strict named invasive histology; exclude mixed/other/conflicting morphology",
              "sample_policy":"PDC Primary Tumor only, exact case and sample ID or _D<number> suffix match",
              "universe":"eligible PDC-annotated primary tumor slides in local TCIA manifest with both CONCH scales",
              "slides":len(kept),"patients":len({r["case_id"] for r in kept}),
              "slide_class_counts":dict(Counter(r["label"] for r in kept)),
              "patient_class_counts":dict(Counter(dict((r["case_id"],r["label"]) for r in kept).values())),
              "exclusions":dict(Counter(r["reason"] for r in audit if not r["label"])),
              "builder_sha256":sha(__file__),"source_sha256":{name:sha(args.output/name) for name in inputs},
              "artifact_sha256":{name:sha(args.output/name) for name in ("manifest.csv","slide_audit.csv","clinical_labels.csv")}}
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":main()
