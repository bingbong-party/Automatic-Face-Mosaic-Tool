# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
from pathlib import Path

# -- 설정 --
MAX_FILE_SIZE  = 2 * 1024 * 1024   # 목표 용량 (2MB)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def imread_unicode(path):
    stream = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image, quality=None):
    ext = Path(path).suffix.lower()
    params = []
    if quality is not None:
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    result, buf = cv2.imencode(ext, image, params)
    if result:
        buf.tofile(str(path))
        return True
    return False


def resize_under_limit(image, ext, max_bytes):
    """이미지를 인코딩했을 때 max_bytes 이하가 되도록 품질/해상도를 줄인다."""
    quality = 95
    while True:
        params = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        result, buf = cv2.imencode(ext, image, params)
        if not result or buf.nbytes <= max_bytes:
            return image, (quality if params else None)

        if params and quality > 50:
            quality -= 10
            continue

        h, w = image.shape[:2]
        if min(h, w) <= 200:
            return image, (quality if params else None)
        image = cv2.resize(image, (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_AREA)


def process_image(image_path, output_path):
    image = imread_unicode(image_path)
    if image is None:
        return False, False

    ext = image_path.suffix.lower()
    resized = False
    quality = None
    if image_path.stat().st_size > MAX_FILE_SIZE:
        image, quality = resize_under_limit(image, ext, MAX_FILE_SIZE)
        resized = True

    imwrite_unicode(output_path, image, quality)
    return True, resized


def main():
    print("=" * 50)
    print("  이미지 용량 리사이저 (2MB 이하)")
    print("=" * 50)

    base_dir   = Path.cwd()
    input_dir  = base_dir / "images"
    output_dir = base_dir / "resized_output"

    print(f"\n실행 위치: {base_dir}")
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}\n")

    if not input_dir.exists():
        print(f"[오류] 'images' 폴더가 없습니다.")
        input("\nEnter 키를 눌러 종료...")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)

    images = [f for f in input_dir.iterdir()
              if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print("[알림] images 폴더에 지원되는 이미지 파일이 없습니다.")
        input("\nEnter 키를 눌러 종료...")
        sys.exit(0)

    print(f"총 {len(images)}장 처리 시작...\n")

    success_count = 0
    resized_count  = 0

    for i, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        ok, resized = process_image(img_path, out_path)
        status = "리사이즈됨" if resized else "유지 (이미 2MB 이하)"
        if not ok:
            status = "읽기 실패"
        print(f"  [{i:>3}/{len(images)}] {img_path.name:<35} -> {status}")
        if ok:
            success_count += 1
            if resized:
                resized_count += 1

    print("\n" + "=" * 50)
    print(f"  완료! {success_count}/{len(images)}장 처리")
    print(f"  {resized_count}장 리사이즈됨")
    print(f"  저장 위치: {output_dir}")
    print("=" * 50)
    input("\nEnter 키를 눌러 종료...")


if __name__ == "__main__":
    main()
