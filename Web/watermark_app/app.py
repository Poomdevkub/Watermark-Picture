from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from PIL import Image, ImageEnhance
import pywt
from numpy.linalg import svd
from scipy.ndimage import gaussian_filter

app = Flask(__name__)

# Ensure the output folder exists
if not os.path.exists('static/outputs'):
    os.makedirs('static/outputs')

@app.route('/')
def page1():
    return render_template('index.html')

#ยังไม่เสร็จ
#---
@app.route('/make-watermark-dwt', methods=['GET', 'POST'])
def make_watermark_dwt_page():
    if request.method == 'POST':
        # รับไฟล์ภาพต้นฉบับและภาพลายน้ำ
        image_file = request.files['image']
        watermark_file = request.files['watermark']
        alpha = float(request.form['alpha'])  # รับค่า alpha จากฟอร์ม

        if image_file and watermark_file:
            # เปิดภาพ
            image = Image.open(image_file).convert('RGB')
            watermark = Image.open(watermark_file).convert('L')  # แปลงเป็นขาวดำสำหรับลายน้ำ

            # แปลงภาพเป็น numpy array
            img_array = np.array(image)
            wm_array = np.array(watermark)

            # แยกช่องสี R, G, B ของภาพต้นฉบับ
            r_channel, g_channel, b_channel = img_array[..., 0], img_array[..., 1], img_array[..., 2]

            # ฟังก์ชันสำหรับการแปลง DWT
            def apply_dwt(channel):
                coeffs = pywt.dwt2(channel, 'haar')
                LL, (LH, HL, HH) = coeffs
                return LL, (LH, HL, HH)

            # นำลายน้ำไปใส่ในช่องสีแดง
            LL_r, (LH_r, HL_r, HH_r) = apply_dwt(r_channel)

            # Resize watermark ให้มีขนาดเท่ากับ LL_r ของช่องสีแดง
            wm_array_resized = np.array(watermark.resize(LL_r.shape[::-1]))  # resize ขนาดลายน้ำให้ตรงกับ LL_r

            # เพิ่มลายน้ำเข้าไปใน LL ของช่องสีแดง
            LL_r_watermarked = LL_r + alpha * wm_array_resized

            # ฟังก์ชันสำหรับการแปลงกลับ IDWT
            def apply_idwt(LL, LH_HL_HH):
                return pywt.idwt2((LL, LH_HL_HH), 'haar')

            # แปลงกลับเป็นภาพที่ใส่ลายน้ำแล้ว
            r_channel_watermarked = apply_idwt(LL_r_watermarked, (LH_r, HL_r, HH_r))

            # รวมช่องสี R, G, B กลับเป็นภาพเดียว
            watermarked_image = np.stack([r_channel_watermarked, g_channel, b_channel], axis=-1)

            # จัดการค่าให้เหมาะสมกับภาพ
            watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

            # บันทึกภาพที่ใส่ลายน้ำแล้ว
            output_filename = 'watermarked_image_dwt.png'
            output_path = os.path.join('static/outputs', output_filename)

            # ตรวจสอบว่าโฟลเดอร์สำหรับเก็บผลลัพธ์มีอยู่หรือไม่ หากไม่มีให้สร้างขึ้น
            if not os.path.exists('static/outputs'):
                os.makedirs('static/outputs')

            Image.fromarray(watermarked_image).save(output_path)

            # แสดงลิงค์ให้ดาวน์โหลดภาพ
            return render_template('Make watermark_DWT.html', download_link=output_filename)

    return render_template('Make watermark_DWT.html')


#---

@app.route('/make-watermark-dwt-svd', methods=['GET', 'POST'])
def make_watermark_dwt_svd_page():
    if request.method == 'POST':
        # Get the uploaded files
        image_file = request.files['image']
        watermark_file = request.files['watermark']

        if image_file and watermark_file:
            # Open the images
            image = Image.open(image_file)
            watermark = Image.open(watermark_file)

            # Apply DWT and SVD watermarking (normal visibility)
            watermarked_image = dwt_svd_watermarking(image, watermark)

            # Apply DWT and SVD watermarking (enhanced visibility)
            enhanced_watermarked_image = dwt_svd_watermarking(image, watermark, enhanced=True)

            # Save the output images to the static/outputs folder
            output_filename = 'watermarked_image_dwt_svd.png'
            enhanced_output_filename = 'watermarked_image_dwt_svd_enhanced.png'

            output_path = os.path.join('static/outputs', output_filename)
            enhanced_output_path = os.path.join('static/outputs', enhanced_output_filename)

            Image.fromarray(watermarked_image).save(output_path)
            Image.fromarray(enhanced_watermarked_image).save(enhanced_output_path)

            # After processing, show download links
            return render_template('Make watermark_DWT+SVD.html', 
                                   download_link=output_filename, 
                                   enhanced_download_link=enhanced_output_filename)

    return render_template('Make watermark_DWT+SVD.html')

@app.route('/detect-watermark', methods=['GET', 'POST'])
def detect_watermark_page():
    if request.method == 'POST':
        # รับไฟล์ภาพที่อัปโหลด
        original_image_file = request.files['original_image']
        watermarked_image_file = request.files['watermarked_image']

        if original_image_file and watermarked_image_file:
            # เปิดภาพ
            original_image = Image.open(original_image_file)
            watermarked_image = Image.open(watermarked_image_file)

            # ตรวจจับลายน้ำ
            result = detect_watermark_svd(original_image, watermarked_image)

            # ส่งผลลัพธ์กลับไปยังผู้ใช้
            return render_template('Detect watermark.html', result=result)
        
    return render_template('Detect watermark.html')


@app.route('/delete-watermark', methods=['GET', 'POST'])
def remove_watermark():
    if request.method == 'POST':
        image_file = request.files['image']
        
        if image_file:
            watermarked_image = Image.open(image_file)
            watermarked_array = np.array(watermarked_image)

            # ใช้ค่าซิกม่าคงที่
            sigma_value = 0.1  # ค่านี้ไม่ต้องให้ผู้ใช้เลือก

            # ใช้ Gaussian filter
            if watermarked_array.ndim == 3:
                channels = []
                for i in range(3):
                    filtered_channel = gaussian_filter(watermarked_array[:, :, i], sigma=sigma_value)
                    channels.append(filtered_channel)
                new_image_array = np.stack(channels, axis=-1)
            else:
                new_image_array = gaussian_filter(watermarked_array, sigma=sigma_value)

            new_image = Image.fromarray(np.uint8(new_image_array))

            # ปรับความคมชัด
            enhancer = ImageEnhance.Sharpness(new_image)
            new_image_sharpened = enhancer.enhance(1.5)  # ปรับตามต้องการ

            # ปรับคอนทราสต์
            contrast_enhancer = ImageEnhance.Contrast(new_image_sharpened)
            new_image_contrasted = contrast_enhancer.enhance(1.0)  # ปรับตามต้องการ

            # ปรับความสว่าง
            brightness_enhancer = ImageEnhance.Brightness(new_image_contrasted)
            new_image_brightened = brightness_enhancer.enhance(1.0)  # ปรับตามต้องการ

            # บันทึกภาพที่ไม่มีลายน้ำ
            output_filename = 'output_image_without_watermark.png'
            output_path = os.path.join('static/outputs', output_filename)
            new_image_brightened.save(output_path)

            return render_template('Delete watermark.html', download_link=output_filename)

    return render_template('Delete watermark.html')

@app.route('/group-member')
def page3():
    return render_template('Group member.html')

def dwt_svd_watermarking(image, watermark, enhanced=False):
    # Convert images to RGB
    image = image.convert('RGB')
    watermark = watermark.convert('L')

    # Resize watermark to match the image size
    watermark = watermark.resize(image.size)

    # Convert images to numpy arrays
    img_array = np.array(image)
    wm_array = np.array(watermark)

    # Split the image into RGB channels
    r_channel, g_channel, b_channel = img_array[..., 0], img_array[..., 1], img_array[..., 2]

    # Apply DWT and SVD for each channel
    def apply_dwt_svd(channel):
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs
        U, S, Vt = svd(LL, full_matrices=False)

        # Apply SVD to the resized and grayscale watermark
        U_w, S_w, Vt_w = svd(wm_array, full_matrices=False)

        # Resize the singular values of the watermark to match the channel
        min_len = min(len(S), len(S_w))
        S_w_resized = np.pad(S_w, (0, max(0, len(S) - len(S_w))), mode='constant')[:min_len]

        # Adjust visibility of the watermark if enhanced
        alpha = 0.01 if not enhanced else 0.5  # Higher alpha for enhanced visibility

        # Modify singular values with the resized singular values of the watermark
        S_new = S + alpha * S_w_resized

        # Reconstruct the LL subband with modified singular values
        LL_new = np.dot(U, np.dot(np.diag(S_new), Vt))

        # Apply inverse DWT to get the watermarked channel
        return np.clip(pywt.idwt2((LL_new, (LH, HL, HH)), 'haar'), 0, 255)

    # Process each channel
    r_watermarked = apply_dwt_svd(r_channel)
    g_watermarked = apply_dwt_svd(g_channel)
    b_watermarked = apply_dwt_svd(b_channel)

    # Combine the watermarked channels back into an image
    watermarked_image = np.stack((r_watermarked, g_watermarked, b_watermarked), axis=-1)

    return watermarked_image.astype(np.uint8)

def detect_watermark_svd(original_image, watermarked_image):
    # Convert images to RGB
    original_image = original_image.convert('RGB')
    watermarked_image = watermarked_image.convert('RGB')

    # Convert images to numpy arrays
    original_array = np.array(original_image)
    watermarked_array = np.array(watermarked_image)

    # Split the image into RGB channels
    r_original, g_original, b_original = original_array[..., 0], original_array[..., 1], original_array[..., 2]
    r_watermarked, g_watermarked, b_watermarked = watermarked_array[..., 0], watermarked_array[..., 1], watermarked_array[..., 2]

    # Apply DWT and SVD for watermark detection
    def check_watermark_channel(original_channel, watermarked_channel):
        # Apply DWT
        coeffs_original = pywt.dwt2(original_channel, 'haar')
        coeffs_watermarked = pywt.dwt2(watermarked_channel, 'haar')

        LL_original, _ = coeffs_original
        LL_watermarked, _ = coeffs_watermarked

        # Apply SVD
        _, S_original, _ = svd(LL_original, full_matrices=False)
        _, S_watermarked, _ = svd(LL_watermarked, full_matrices=False)

        # Compare singular values
        return np.allclose(S_original, S_watermarked, atol=0.1)  # Tolerance level for differences

    # Check watermark on each channel
    r_detected = not check_watermark_channel(r_original, r_watermarked)
    g_detected = not check_watermark_channel(g_original, g_watermarked)
    b_detected = not check_watermark_channel(b_original, b_watermarked)

    # If watermark detected in any channel, return True
    return r_detected or g_detected or b_detected


# ฟังก์ชันสำหรับเบลอลายน้ำ
def remove_watermark(image_path, x, y, w, h):
    # โหลดรูปภาพ
    image = cv2.imread(image_path)

    # ส่วนที่ลายน้ำอยู่ (ระบุตำแหน่ง x, y, w, h)
    watermark_area = image[y:y+h, x:x+w]

    # เบลอบริเวณที่มีลายน้ำ
    blurred = cv2.GaussianBlur(watermark_area, (51, 51), 0)

    # นำบริเวณที่เบลอกลับไปใส่ในภาพเดิม
    image[y:y+h, x:x+w] = blurred

    # บันทึกรูปภาพที่ถูกแก้ไข
    output_path = 'static/edited_image.jpg'
    cv2.imwrite(output_path, image)

    return output_path



@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static/outputs', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
