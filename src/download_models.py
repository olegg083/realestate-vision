"""Скачивание обученной модели, FAISS-индекса и метрик из GitHub Release."""
import argparse
import urllib.error
import urllib.request

from src.config import GITHUB_REPO, MODELS_DIR, MODELS_RELEASE_TAG, RELEASE_ARTIFACTS


def download(url: str, target) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    tmp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=MODELS_RELEASE_TAG)
    parser.add_argument("--force", action="store_true", help="Перекачать уже существующие файлы")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{args.tag}"

    for path in RELEASE_ARTIFACTS:
        if path.exists() and not args.force:
            print(f"{path.name:24s} уже есть, пропускаю")
            continue
        print(f"{path.name:24s} скачиваю...", end=" ", flush=True)
        try:
            download(f"{base_url}/{path.name}", path)
        except urllib.error.HTTPError as e:
            raise SystemExit(
                f"\nОшибка {e.code} для {path.name}. Проверьте, что релиз {args.tag} существует, "
                f"содержит этот файл и репозиторий {GITHUB_REPO} публичный."
            ) from e
        print(f"{path.stat().st_size / 1024 / 1024:.1f} МБ")

    print(f"Готово: {MODELS_DIR}")


if __name__ == "__main__":
    main()
