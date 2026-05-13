import json
import sys
from pathlib import Path

def load_compounds(txt_file):
    """Load compounds from input.txt (one per line)."""
    with open(txt_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

def load_json(json_file):
    """Load JSON list of compound dicts."""
    with open(json_file, "r") as f:
        return json.load(f)

def filter_compounds(data, compound_set):
    """Filter JSON entries based on compound names."""
    return [item for item in data if item.get("compound") in compound_set]

def main(input_txt, input_json):
    compounds = load_compounds(input_txt)
    data = load_json(input_json)

    filtered = filter_compounds(data, compounds)

    n = len(filtered)

    # build output filename
    json_path = Path(input_json)
    output_file = f"{json_path.stem}_selected_{n}.json"

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved {n} compounds → {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter.py input.txt input.json")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
