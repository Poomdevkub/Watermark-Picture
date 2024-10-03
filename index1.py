from PIL import Image
import numpy as np
import pywt

# ฟังก์ชันแปลงภาพเป็นอาร์เรย์
def image_to_array(image):
    return np.array(image, dtype=np.float32)

# ฟังก์ชันแปลงอาร์เรย์กลับเป็นภาพ
def array_to_image(array):
    return Image.fromarray(array.astype(np.uint8))

# ฟังก์ชันฝังลายน้ำใน DWT ของภาพสี
def embed_watermark_dwt_rgb(image_array, watermark_array, alpha=0.5):
    channels = []
    
    for i in range(3):  # แยกช่องสี R, G, B
        channel = image_array[:, :, i]
        coeffs2 = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # ขนาดของลายน้ำ
        wm_height, wm_width = watermark_array.shape
        LL[:wm_height, :wm_width] += alpha * watermark_array  # ใช้ alpha เพื่อควบคุมความแรง
        
        # แปลงกลับด้วย Inverse DWT
        watermarked_coeffs = (LL, (LH, HL, HH))
        watermarked_channel = pywt.idwt2(watermarked_coeffs, 'haar')
        
        channels.append(watermarked_channel)
    
    return np.clip(np.stack(channels, axis=-1), 0, 255)  # ใช้ np.clip เพื่อให้ค่าสีไม่เกินช่วง 0-255

# โหลดภาพต้นฉบับและลายน้ำ
original_image = Image.open('01.png')
watermark_image = Image.open('02.png').convert('L')

# แปลงภาพเป็นอาร์เรย์
original_array = image_to_array(original_image)
watermark_array = image_to_array(watermark_image)

# ขนาดของลายน้ำต้องเล็กกว่า LL component
if watermark_array.shape[0] > original_array.shape[0] // 2 or watermark_array.shape[1] > original_array.shape[1] // 2:
    watermark_array = watermark_array[:original_array.shape[0] // 2, :original_array.shape[1] // 2]

# ฝังลายน้ำในช่องสี
watermarked_array = embed_watermark_dwt_rgb(original_array, watermark_array, alpha=0.25)

# สร้างและบันทึกภาพที่มีลายน้ำ
watermarked_image = array_to_image(watermarked_array)
watermarked_image.save('watermarked_image_1.png')
