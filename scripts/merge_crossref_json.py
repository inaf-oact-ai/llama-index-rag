#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def extract_record(obj):
	"""
	Accepts:
	- raw Crossref work record
	- Crossref API response: {"status": "ok", "message": {...}}
	- wrapped local record: {"message": {...}}
	"""
	if isinstance(obj, dict) and isinstance(obj.get("message"), dict):
		return obj["message"]

	return obj


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--json_dir",
		required=True,
		help="Directory containing one Crossref JSON file per paper",
	)
	parser.add_argument(
		"--output",
		required=True,
		help="Output aggregate JSON file",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		default=False,
		help="Write indented JSON",
	)
	args = parser.parse_args()

	json_dir = Path(args.json_dir)
	output = Path(args.output)

	records = []
	seen_dois = set()

	for path in sorted(json_dir.glob("*.json")):
		with path.open("r", encoding="utf-8") as f:
			obj = json.load(f)

		record = extract_record(obj)

		if not isinstance(record, dict):
			print(f"[WARN] Skipping non-object JSON: {path}")
			continue

		doi = record.get("DOI") or record.get("doi")
		doi_key = doi.lower().strip() if isinstance(doi, str) else None

		if doi_key and doi_key in seen_dois:
			print(f"[WARN] Duplicate DOI skipped: {doi} ({path})")
			continue

		if doi_key:
			seen_dois.add(doi_key)

		records.append(record)

	output.parent.mkdir(parents=True, exist_ok=True)

	with output.open("w", encoding="utf-8") as f:
		if args.pretty:
			json.dump(records, f, ensure_ascii=False, indent=2)
		else:
			json.dump(records, f, ensure_ascii=False)

	print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
	main()
