# -*- coding: utf-8 -*-
import cv2
import numpy as np
import urllib.request
import sys
from pathlib import Path
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

# -- 설정 --
BLUR_STRENGTH  = 99
FACE_PADDING   = 0.05
CONFIDENCE     = 0.8   # 낮출수록 더 많이 감지 (오탐 증가 가능)
NMS_THRESHOLD  = 0.3
MAX_FILE_SIZE  = 4 * 1024 * 1024   # 처리 전 리사이징 목표 용량 (4MB)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic", ".heif"}

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL      = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def download_model(model_path):
    if model_path.exists():
        return
    print("  얼굴 감지 모델 다운로드 중... (최초 1회)")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("  모델 다운로드 완료!\n")


def imread_unicode(path):
    if Path(path).suffix.lower() in (".heic", ".heif"):
        pil_img = Image.open(str(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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


def apply_blur(image, x1, y1, x2, y2):
    face_region = image[y1:y2, x1:x2]
    if face_region.size == 0:
        return image

    strength = BLUR_STRENGTH if BLUR_STRENGTH % 2 == 1 else BLUR_STRENGTH + 1
    blurred = cv2.GaussianBlur(face_region, (strength, strength), 0)

    fh, fw = face_region.shape[:2]
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.ellipse(mask, (fw // 2, fh // 2), (fw // 2, fh // 2), 0, 0, 360, 255, -1)
    mask3 = cv2.merge([mask, mask, mask])

    image[y1:y2, x1:x2] = np.where(mask3 == 255, blurred, face_region)
    return image


def process_image(image_path, output_path, detector):
    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        return False, 0

    ext = image_path.suffix.lower()
    # HEIC는 OpenCV로 인코딩 불가 → JPEG로 저장
    if ext in (".heic", ".heif"):
        ext = ".jpg"
        output_path = output_path.with_suffix(".jpg")

    quality = None
    # HEIC→JPEG 변환 시 파일 크기가 커질 수 있으므로 인코딩 후 크기 기준으로 확인
    needs_resize = image_path.stat().st_size > MAX_FILE_SIZE
    if not needs_resize and ext == ".jpg":
        _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        needs_resize = buf.nbytes > MAX_FILE_SIZE
    if needs_resize:
        image_bgr, quality = resize_under_limit(image_bgr, ext, MAX_FILE_SIZE)

    h, w = image_bgr.shape[:2]

    # 고해상도 이미지는 리사이즈해서 감지 후 원본에 적용
    scale = 1.0
    detect_img = image_bgr
    if max(h, w) > 1920:
        scale = 1920 / max(h, w)
        detect_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    dh, dw = detect_img.shape[:2]
    detector.setInputSize((dw, dh))
    _, faces = detector.detect(detect_img)

    face_count = 0
    if faces is not None:
        for face in faces:
            score = face[14]
            if score < CONFIDENCE:
                continue

            # 감지 좌표를 원본 해상도로 역산
            ox, oy, bw, bh = face[0:4] / scale

            pad_x = int(bw * FACE_PADDING)
            pad_y = int(bh * FACE_PADDING)
            x1 = max(0, int(ox) - pad_x)
            y1 = max(0, int(oy) - pad_y)
            x2 = min(w, int(ox + bw) + pad_x)
            y2 = min(h, int(oy + bh) + pad_y)

            image_bgr = apply_blur(image_bgr, x1, y1, x2, y2)
            face_count += 1

    imwrite_unicode(output_path, image_bgr, quality)
    return True, face_count


def main():
    print("=" * 50)
    print("  얼굴 자동 가우시안 블러 처리기")
    print("=" * 50)

    base_dir   = Path.cwd()
    input_dir  = base_dir / "images"
    output_dir = base_dir / "blurred_output"
    model_path = base_dir / MODEL_FILENAME

    print(f"\n실행 위치: {base_dir}")
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}\n")

    if not input_dir.exists():
        print(f"[오류] 'images' 폴더가 없습니다.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    download_model(model_path)

    images = [f for f in input_dir.iterdir()
              if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print("[알림] images 폴더에 지원되는 이미지 파일이 없습니다.")
        sys.exit(0)

    print(f"총 {len(images)}장 처리 시작...\n")

    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320),
        score_threshold=CONFIDENCE,
        nms_threshold=NMS_THRESHOLD,
    )

    success_count = 0
    total_faces   = 0

    for i, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        ok, faces = process_image(img_path, out_path, detector)
        status = f"얼굴 {faces}개 블러" if faces > 0 else "얼굴 없음"
        if not ok:
            status = "읽기 실패"
        print(f"  [{i:>3}/{len(images)}] {img_path.name:<35} -> {status}")
        if ok:
            success_count += 1
            total_faces += faces

    print("\n" + "=" * 50)
    print(f"  완료! {success_count}/{len(images)}장 처리")
    print(f"  총 {total_faces}개 얼굴 블러 처리")
    print(f"  저장 위치: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
