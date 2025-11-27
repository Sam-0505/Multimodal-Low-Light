import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm

# PSNR implementation
def compare_psnr(img1, img2, data_range=None):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    if data_range is None:
        data_range = img1.max() - img1.min()
        if data_range == 0:
            data_range = 255.0
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

# SSIM implementation
def compare_ssim(img1, img2, data_range=None):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    if data_range is None:
        data_range = img1.max() - img1.min()
        if data_range == 0:
            data_range = 255.0
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Ensure images are 2D (grayscale)
    if len(img1.shape) == 3:
        # Convert to grayscale if needed
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

import argparse
parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--test_dir', type=str, default='test_output', help='training data directory')
parser.add_argument('--test_gt_dir', type=str, default='data/LOL/test15/high', help='training data directory')
args = parser.parse_args()

def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def align_to_four(img):
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col, :]
    return img


def evaluate_raindrop(in_dir, gt_dir):
    inputs = sorted(glob(os.path.join(in_dir, '*.png')) + glob(os.path.join(in_dir, '*.jpg')))
    gts = sorted(glob(os.path.join(gt_dir, '*.png')) + glob(os.path.join(gt_dir, '*.jpg')))
    psnrs = []
    ssims = []
    for input, gt in tqdm(zip(inputs, gts)):
        inputdata = cv2.imread(input)
        gtdata = cv2.imread(gt)
        inputdata = align_to_four(inputdata)
        gtdata = align_to_four(gtdata)
        psnrs.append(calc_psnr(inputdata, gtdata))
        ssims.append(calc_ssim(inputdata, gtdata))

    ave_psnr = np.array(psnrs).mean()
    ave_ssim = np.array(ssims).mean()
    return ave_psnr, ave_ssim


if __name__ == '__main__':
    ave_psnr, ave_ssim = evaluate_raindrop(args.test_dir, args.test_gt_dir)
    print('')
    print('PSNR: ', ave_psnr)
    print('SSIM: ', ave_ssim)
