import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

s = hs.load("2.bcf")
print(s)
image = s[0]
maps = s[2]

# 配置信号类型
maps.set_signal_type('EDS_SEM')

# 数据重采样
maps_rebin = maps.rebin([256, 192, 512])
image_rebin = image.rebin([256, 192])
maps_rebin.plot(navigator=image_rebin)
maps_rebin.change_dtype('float32')
maps_rebin.decomposition(False, algorithm="SVD", output_dimension=20)
ax = maps_rebin.plot_explained_variance_ratio()
ax.figure.savefig('PAC Scree Plot.png')
plt.close()
ax = maps_rebin.plot_explained_variance_ratio(threshold=5)
ax.figure.savefig('PAC Scree Plot_threshold.png')
plt.close()

# 以上与PCA-Denoised相同；在进行NMF处理前，需要采用PCA处理，选择NMF的output_dimension
maps_rebin.decomposition(False, algorithm="NMF", output_dimension=3)

spe = maps_rebin.get_decomposition_factors()
img = maps_rebin.get_decomposition_loadings()
for i in range(len(spe.split())):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle('Components_NMF_%d' % i, fontsize=20)

    ax1 = fig.add_subplot(121)
    ax1.imshow(img.split()[i], cmap='plasma')
    ax1.axis('off')

    ax2 = fig.add_subplot(122)
    y = spe.split()[i].data
    x = np.arange(512)
    k = 512 / 20
    xr = x / k
    ax2.plot(xr, y, 'r')
    ax2.set_xlim(-0.5, 20)
    ax2.set_xlabel('Energy/keV')

    plt.savefig('Components_NMF_%d' % i)
    plt.close()
