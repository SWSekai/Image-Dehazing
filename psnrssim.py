import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def compute_metrics(original_dir, processed_dir):
    filenames = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    for filename in filenames:
        orig_path = os.path.join(original_dir, filename)
        proc_path = os.path.join(processed_dir, filename)

        if not os.path.exists(proc_path):
            print(f"[Warning] Processed image not found for: {filename}")
            continue

        # Read images
        img1 = cv2.imread(orig_path)
        img2 = cv2.imread(proc_path)

        if img1 is None or img2 is None:
            print(f"[Error] Failed to read images: {filename}")
            continue

        if img1.shape != img2.shape:
            print(f"[Warning] Image size mismatch: {filename}")
            continue

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        psnr = compare_psnr(img1, img2, data_range=255)
        ssim = compare_ssim(img1, img2, channel_axis=2)

        print(f"{filename} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

        psnr_total += psnr
        ssim_total += ssim
        count += 1

    if count > 0:
        print("\nAverage PSNR:", psnr_total / count)
        print("Average SSIM:", ssim_total / count)
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    compute_metrics('./gt/', './input_image/')