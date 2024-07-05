import cv2
import numpy as np

def apply_spyhole_effect(frame, center=None, radius_ratio=0.5):
    height, width = frame.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    radius = int(min(width, height) * radius_ratio)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    distance_from_center = np.sqrt((np.indices((height, width))[0] - center[1]) ** 2 + (np.indices((height, width))[1] - center[0]) ** 2)
    gradient = np.clip((radius - distance_from_center) / radius, 0, 1)
    gradient = (gradient * 255).astype(np.uint8)
    gradient = cv2.merge([gradient] * 3)

    blurred = cv2.GaussianBlur(frame, (21, 21), 0)
    mask = cv2.merge([mask] * 3)
    masked_frame = np.where(mask == 255, frame, gradient)

    blended = cv2.addWeighted(masked_frame, 1, blurred, 0.5, 0)
    return blended

def adjust_hsl(frame, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0):
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hls_frame = np.float32(hls_frame)

    hls_frame[:, :, 0] = (hls_frame[:, :, 0] + hue_shift) % 180
    hls_frame[:, :, 1] = np.clip(hls_frame[:, :, 1] * lightness_scale, 0, 255)
    hls_frame[:, :, 2] = np.clip(hls_frame[:, :, 2] * saturation_scale, 0, 255)

    hls_frame = np.uint8(hls_frame)
    return cv2.cvtColor(hls_frame, cv2.COLOR_HLS2BGR)

def adjust_exposure(frame, exposure_scale=1.0):
    frame = np.float32(frame)
    frame = np.clip(frame * exposure_scale, 0, 255)
    return np.uint8(frame)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = adjust_hsl(frame, hue_shift=10, saturation_scale=1.5, lightness_scale=1.2)
        frame = adjust_exposure(frame, exposure_scale=1.2)
        spyhole_frame = apply_spyhole_effect(frame)

        cv2.imshow('Spyhole Effect', spyhole_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()