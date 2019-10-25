"""
Creat on Wednesday, October 23, 2019 9:49:55 AM
By: wangliangju@gmail.com

"""
#%%
import veinnet as vn
import matplotlib.pyplot as plt
import wljbox.smalltools as sts
import networkx as nx

fpath = r'F:\dataProcessed\leafspec\soybean\2019-03-25\leafspecB-20190804\ndvitiff\soybean3_Harosoy_WS-N150_16_3_2019-3-23-23-42-28_2019-03-25-16-01-24.tiff'
#%%
segshow = True

segimg = vn.SegImage(fpath)
segimg.imread(flag=-1)

segimg.togray()
segimg.blur(blursize=5)
segimg.seg(blocksize=51, threshBackgrd=1, inversed=True)
if segshow:
    plt.imshow(segimg.imggray)
    plt.show()
    plt.imshow(segimg.imgbinary)
    plt.show()

segimg.denoise(minimum_feature_size = 71, smoothing=True)
if segshow:
    plt.imshow(segimg.imgbinary)
    plt.show()

print('SEG>>>>  Done')
#%%
netshow = True
vnet = vn.VeinNet(segimg.imgbinary, debug=True, verbose=True)

vnet.getDistnaceMap(issave=True)

if netshow:
    plt.imshow(vnet.distanceMap)
    plt.show()

vnet.getContours()
print('NET>>>>  contours Done')
#%%
vnet.mesh()

vnet.triangulate()
vnet.classifyTriangle()
vnet.graphy()
vnet.removeRedundantNode()

print('NET>>>>  net Done')
#%%
gragh = nx.read_gpickle('./veinnet_graph_r0_p5.gpickle')
vpara = vn.VeinPara(vnet.graph)
paras = vpara.allParas()
print(paras)
