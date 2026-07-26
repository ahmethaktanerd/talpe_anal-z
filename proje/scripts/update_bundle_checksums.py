"""Model paketindeki tüm teslim dosyaları için SHA-256 kayıtlarını yeniler."""

import json

from scripts.data_utils import file_sha256
from scripts.project_config import BUNDLE_DIR


def main() -> None:
    if not BUNDLE_DIR.is_dir():
        raise FileNotFoundError(f"Model paket dizini bulunamadı: {BUNDLE_DIR}")

    checksums = {
        path.name: file_sha256(path)
        for path in sorted(BUNDLE_DIR.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    if not checksums:
        raise RuntimeError("Model paketinde checksum üretilecek dosya bulunamadı.")

    output_path = BUNDLE_DIR / "checksums.json"
    output_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{len(checksums)} dosyanın checksum kaydı güncellendi: {output_path}")


if __name__ == "__main__":
    main()
