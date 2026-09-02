from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"

def remove_pt_files(root: Path = DATASET_DIR) -> int:
    removed = 0
    for pt_file in root.rglob("*.pt"):
        if pt_file.is_file():
            pt_file.unlink()
            print(f"removed {pt_file}")
            removed += 1
    return removed


if __name__ == "__main__":
    count = remove_pt_files()
    print(f"removed {count} .pt file(s)")
