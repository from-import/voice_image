import numpy as np
import librosa
import soundfile as sf
from PIL import Image

# 加载音频文件
y, sr = librosa.load('test.wav')

# 计算STFT
D = librosa.stft(y)

# 提取振幅和相位
mag, phase = librosa.magphase(D)

# 计算相位的角度
phase_angle = np.angle(phase)

# 映射到图像像素范围
mag_img = np.interp(mag, (mag.min(), mag.max()), (0, 255))
phase_img = np.interp(phase_angle, (-np.pi, np.pi), (0, 255))

# 创建一个两通道的图像，一个通道为振幅，另一个通道为相位角度
image = np.stack([mag_img, phase_img], axis=2).astype(np.uint8)

# 保存图像
Image.fromarray(image).save('spectrum.png')

# 读取图像
image = Image.open('spectrum.png')
image_data = np.array(image)

# 分离振幅和相位通道，并将它们反向映射到原始范围
mag_img = image_data[:, :, 0]
phase_img = image_data[:, :, 1]
mag_restored = np.interp(mag_img, (0, 255), (mag.min(), mag.max()))
phase_angle_restored = np.interp(phase_img, (0, 255), (-np.pi, np.pi))

# 用恢复的振幅和相位角度创建复数频谱
D_restored = mag_restored * np.exp(1j * phase_angle_restored)

# 进行逆STFT
y_restored = librosa.istft(D_restored)

# 保存为音频文件
sf.write('audio_restored.wav', y_restored, sr)
