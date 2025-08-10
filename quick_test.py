import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from EliteNET import UNet
import yaml


def load_and_preprocess(img_path):
    """Load and transform the image to tensor"""
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # shape: [1, H, W]
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    img_tensor = transform(raw)  # shape: [1, 192, 128]
    return img_tensor.unsqueeze(0), raw  # add batch dim, return raw too


def postprocess_and_save(output_tensor, orig_raw, out_dir, img_name):
    """Save output and calculate metrics"""
    os.makedirs(out_dir, exist_ok=True)

    denoised = output_tensor.squeeze().cpu().numpy()  # shape: [H, W]
    
    orig_resized = cv2.resize(orig_raw, (128, 156), interpolation=cv2.INTER_LINEAR)
    # Pad 18 pixels on top and 18 on bottom to make height = 192
    orig_padded = cv2.copyMakeBorder(orig_resized, 18, 18, 0, 0, cv2.BORDER_REFLECT)


    denoised_img = (denoised * 255).astype(np.uint8)
    orig_img = orig_padded.astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, f"{img_name}_SR.jpg"), denoised_img)
    cv2.imwrite(os.path.join(out_dir, f"{img_name}_LR.jpg"), orig_img)

    psnr = peak_signal_noise_ratio(orig_img, denoised_img, data_range=255)
    ssim = structural_similarity(orig_img, denoised_img)

    with open(os.path.join(out_dir, f"{img_name}_metrics.txt"), "w") as f:
        f.write(f"PSNR: {psnr:.2f}\nSSIM: {ssim:.4f}\n")

    print(f"✅ Saved denoised output at {out_dir}")
    print(f"📈 PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model loading
    model = UNet(in_c=1, n_classes=1, layers=[4, 8, 16]).to(device)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Preprocess
    img_tensor, orig_raw = load_and_preprocess(args.img_path)
    img_tensor = img_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)

    # Save result
    img_name = os.path.splitext(os.path.basename(args.img_path))[0]
    postprocess_and_save(output, orig_raw, args.out_dir, img_name)



def parse_tuple_or_float(value):
    """
    Parse an argument that can be either:
    - a single float → returns (val, val)
    - two floats separated by space or comma → returns (val1, val2)
    """
    # Split by comma or space
    parts = value.replace(",", " ").split()
    nums = [float(p) for p in parts]

    if len(nums) == 1:
        return (nums[0], nums[0])
    elif len(nums) == 2:
        return tuple(nums)
    else:
        raise argparse.ArgumentTypeError(
            f"Expected 1 or 2 float values, got {len(nums)}: '{value}'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to single LR image")
    parser.add_argument("--ckpt",type=str, default="models/best.pth", help="Path of the Checkpoint")
    parser.add_argument("--scale", type=parse_tuple_or_float, required=True,
                        help="Scale factor as single float or two floats (e.g., 2.3 or 2.3,3.1)")
    parser.add_argument("--hr_size", type=parse_tuple_or_float, required=False,
                        help="HR size as single float or two floats (e.g., 256 or 256,512)")
    parser.add_argument("--out_dir", type=str, default="results_single", help="Where to save the result")
    args = parser.parse_args()
    main(args)
