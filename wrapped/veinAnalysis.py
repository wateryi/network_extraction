# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:52:53 2019

@author: wangl
"""

#%%
import veinnet as vn
import matplotlib.pyplot as plt
import wljbox.smalltools as sts
from wljbox.files import Files
import networkx as nx
import os

#%%
def leafveinAnalysis(fpath, dest,
                     imreadFlag=-1, blursize=5, segBlocksize=51,
                     threshBackgrd=1, imgInversed=True,
                     denoiseSize=71, smoothing=True,
                     netDebug=True, netVerbose=True,
                     isShow=True, isSave=True):
    
    imagename = os.path.split(fpath)[1].split('.')[0]
    log = 'LVA>>>>>' + imagename
    segimg = vn.SegImage(fpath)
    segimg.imread(flag=imreadFlag)
    
    segimg.togray()
    segimg.blur(blursize=blursize)
    segimg.seg(blocksize=segBlocksize, threshBackgrd=threshBackgrd,
               inversed=imgInversed)
    if isShow:
        plt.imshow(segimg.imggray)
        plt.show()
        plt.imshow(segimg.imgbinary)
        plt.show()
    
    segimg.denoise(minimum_feature_size = denoiseSize, smoothing=smoothing)
    if isShow:
        plt.imshow(segimg.imgbinary)
        plt.show()
    
    print(log + 'SEG>>>>  Done')
    
    vnet = vn.VeinNet(segimg.imgbinary,
                      imagename=imagename, dest=dest,
                      debug=netDebug, verbose=netVerbose)

    vnet.getDistnaceMap(issave=True)
    
    if isShow:
        plt.imshow(vnet.distanceMap)
        plt.show()
    
    vnet.getContours()
    print(log + 'NET>>>>  contours Done')
    
    vnet.mesh()
    vnet.triangulate()
    vnet.classifyTriangle()
    vnet.graphy()
    vnet.removeRedundantNode()    
    print(log + 'NET>>>>  net Done')
    
    #gragh = nx.read_gpickle('./veinnet_graph_r0_p5.gpickle')
    vpara = vn.VeinPara(vnet.graph)
    paras = vpara.allParas()
    
    print(log + 'Analysis Done')
    return paras

#%%
if __name__=='__main__':
    fpath = r'F:\dataProcessed\leafspec\soybean\2019-03-25\leafspecB-20190804\ndvitiff'
    files = Files(fpath, ext=['*.tiff'])
    imgpaths = files.filesWithPath
    dest = r'F:\dataProcessed\leafspec\soybean\2019-03-25\leafspecB-20190804\veinAnalysis_2'
    outParas = {}
    imgpaths = imgpaths[0:2]
    for ps in imgpaths:
        paras = leafveinAnalysis(ps, dest,
                                 imreadFlag=-1, blursize=5, segBlocksize=51,
                                 threshBackgrd=1, imgInversed=True,
                                 denoiseSize=71, smoothing=True,
                                 netDebug=True, netVerbose=True,
                                 isShow=False, isSave=True)
        for k, v in paras.items():
            outKeys = list(outParas.keys())
            if k in outKeys:
                outParas[k].append(v)
            else:
                outParas[k] = [v]
                
    try:
        sts.Write.saveDict2csv(dest+'/veinResult2.csv', outParas)
        sts.Write.saveDict2csv(dest+'/veinResult2.json', outParas)
    except:
        print('save error')
                
            
        