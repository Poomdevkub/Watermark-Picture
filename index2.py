import numpy as np
import pywt
import cv2

# 1. ฟังก์ชันสำหรับการทำ SVD
def apply_svd(matrix):
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    return U, S, V

def combine_svd(U, S, V):
    return np.dot(U, np.dot(np.diag(S), V))

# 2. ฟังก์ชันการฝังลายน้ำ (DWT + SVD)
def embed_watermark(image, watermark, alpha=0.01):
    # ทำ DWT กับภาพต้นฉบับ
    coeffs_image = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs_image
    
    # ทำ SVD กับ LL ของภาพต้นฉบับ
    U, S, V = apply_svd(LL)
    
    # ทำ DWT และ SVD กับลายน้ำ
    coeffs_watermark = pywt.dwt2(watermark, 'haar')
    LL_w, _ = coeffs_watermark
    Uw, Sw, Vw = apply_svd(LL_w)
    
    # ฝังลายน้ำโดยการปรับค่า S ของภาพด้วยค่า S ของลายน้ำ
    S_marked = S + alpha * Sw
    
    # รวมค่า LL ที่ฝังลายน้ำแล้วกลับมาด้วย SVD
    LL_marked = combine_svd(U, S_marked, V)
    
    # รวมกลับมาเป็นภาพต้นฉบับที่ฝังลายน้ำ
    marked_image = pywt.idwt2((LL_marked, (LH, HL, HH)), 'haar')
    return marked_image

# 3. ฟังก์ชันการสกัดลายน้ำ
def extract_watermark(marked_image, original_image, alpha=0.01):
    # ทำ DWT กับภาพที่มีลายน้ำและภาพต้นฉบับ
    coeffs_marked = pywt.dwt2(marked_image, 'haar')
    LL_marked, _ = coeffs_marked
    
    coeffs_original = pywt.dwt2(original_image, 'haar')
    LL_original, _ = coeffs_original
    
    # ทำ SVD กับ LL ของภาพทั้งสอง
    U_marked, S_marked, V_marked = apply_svd(LL_marked)
    U_original, S_original, V_original = apply_svd(LL_original)
    
    # สกัดค่า S ของลายน้ำ
    Sw_extracted = (S_marked - S_original) / alpha
    
    # สร้างลายน้ำจากค่า S ที่สกัดออกมา
    LL_watermark_extracted = combine_svd(U_marked, Sw_extracted, V_marked)
    
    # รวมกลับมาเป็นลายน้ำที่สกัดออกมา
    extracted_watermark = pywt.idwt2((LL_watermark_extracted, (None, None, None)), 'haar')
    return extracted_watermark

# ตัวอย่างการทดสอบ
# โหลดภาพต้นฉบับและลายน้ำ
image = cv2.imread('image.jpg', 0)  # โหลดภาพในโหมด grayscale
watermark = cv2.imread('watermark.jpg', 0)  # โหลดลายน้ำในโหมด grayscale

# ฝังลายน้ำ
marked_image = embed_watermark(image, watermark, alpha=0.05)
cv2.imwrite('marked_image.jpg', marked_image)

# สกัดลายน้ำ
extracted_watermark = extract_watermark(marked_image, image, alpha=0.05)
cv2.imwrite('extracted_watermark.jpg', extracted_watermark)
