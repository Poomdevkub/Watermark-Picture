import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

def remove_watermark_with_gaussian(image_path, sigma_value=2):
    # เปิดภาพที่มีลายน้ำ
    watermarked_image = Image.open(image_path)
    
    # แปลงภาพเป็นอาเรย์ NumPy
    watermarked_array = np.array(watermarked_image)

    if watermarked_array.ndim == 3:  # ถ้าภาพมี 3 ช่องสี (RGB)
        channels = []
        for i in range(3):  # สำหรับ R, G, B
            filtered_channel = gaussian_filter(watermarked_array[:, :, i], sigma=sigma_value)
            channels.append(filtered_channel)

        new_image_array = np.stack(channels, axis=-1)
    else:
        new_image_array = gaussian_filter(watermarked_array, sigma=sigma_value)

    new_image = Image.fromarray(np.uint8(new_image_array))
    
    # เพิ่มความคมชัด
    enhancer = ImageEnhance.Sharpness(new_image)
    new_image_sharpened = enhancer.enhance(1.5)  # ปรับค่าที่นี่ตามต้องการ

    # บันทึกภาพที่ไม่มีลายน้ำ
    new_image_sharpened.save('output_image_without_watermark.png')
    new_image_sharpened.show()

# เรียกใช้ฟังก์ชัน
remove_watermark_with_gaussian('watermarked0_image.png', sigma_value=1.0)
