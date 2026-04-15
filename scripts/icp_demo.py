import cv2, numpy as np

def msrcp(img, sigmas=[15,80,250]):
    img_f = img.astype(np.float64) + 1.0
    retinex = np.zeros_like(img_f)
    for s in sigmas:
        blur = cv2.GaussianBlur(img_f, (0,0), s)
        retinex += np.log(img_f) - np.log(blur + 1.0)
    retinex /= len(sigmas)

    # 颜色恢复
    img_sum = np.sum(img_f, axis=2, keepdims=True) + 1e-6
    cr = np.clip(np.log(125.0 * img_f / img_sum + 1e-6), 0, None)

    result = np.exp(retinex + cr)
    # 线性拉伸到0-255
    for c in range(3):
        ch = result[:,:,c]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255
        result[:,:,c] = ch

    return result.astype(np.uint8)

img = cv2.imread('/home/csy/graduation_project/Singapura_40.jpg')
dark = (img * 0.15).astype(np.uint8)
enhanced = msrcp(dark)
cv2.imwrite('/home/csy/graduation_project/dark.jpg', dark)
cv2.imwrite('/home/csy/graduation_project/enhanced.jpg', enhanced)
print('已保存到 dark.jpg 和 enhanced.jpg')
cv2.waitKey(0)


### 你需要向答辩委员解释的完整链路

# 暗/过曝图像
#     ↓ MSRCP（增强对比度，恢复几何边缘可见性）
# 增强图像
#     ↓ 深度相机 / 已有点云数据
# 点云（含高斯噪声模拟光照干扰）
#     ↓ Open3D ICP
# 变换矩阵 T_estimated
#     ↓ 与 T_manual 比较互逆性偏差
# 量化结论：MSRCP预处理使ICP在σ=X噪声下偏差从A降到B
