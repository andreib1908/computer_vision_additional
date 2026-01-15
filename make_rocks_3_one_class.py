from pathlib import Path

ROOT = Path("datasets/rocks-3")
SPLITS = ["train", "valid", "test"]


def convert_label_file(p: Path):
    if not p.exists():
        return
    text = p.read_text().strip()
    if not text:
        return  # empty is fine

    new_lines = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue  # skip broken lines
        parts[0] = "0"  # force single class
        new_lines.append(" ".join(parts))

    p.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def main():
    for split in SPLITS:
        labels_dir = ROOT / split / "labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

        for txt in labels_dir.rglob("*.txt"):
            convert_label_file(txt)

    print("Done: datasets/rocks-3 labels are now single-class (class 0).")
    print("Train with: datasets/rocks-3/data_rock.yaml")


if __name__ == "__main__":
    main()
