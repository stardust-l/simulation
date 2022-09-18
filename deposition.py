#created by lgs, M07 iop
from time import time
import numpy as np
import scipy,random,os

#模拟1024*1024网格，周期性边界条件
#1.随机生成给定概率分布的速度dP/dOmega=cos(theta)/pi,横向位置随机分布，纵向位置最大高度加1
#2.当溅射原子碰到薄膜表面时，溅射原子沉积，此点高度加1

#将默认目录改为当前文件夹
current_path=os.path.abspath(__file__)
current_fold = os.path.dirname(current_path)
os.chdir(current_fold)    
def deposition_one(h_list,**par):
    #沉积一个原子的过程
    ##随机生成溅射原子位置，以及速度方向
    #print(par)
    x0,y0,z0=random.randint(1,par['N1']),random.randint(1,par['N2']),np.max(h_list)+1#随机位置
    vtheta,vphi=np.arcsin(random.random()),2*np.pi*random.random()#随机速度方向
    #print(x0,y0,z0,h_list)
    vx,vy,vz=np.sin(vtheta)*np.cos(vphi),np.sin(vtheta)*np.sin(vphi),np.cos(vtheta)#转为直角坐标系
    k_xy=vy/vx
    ##判断原子哪时候接触到薄膜表面，采用布雷森汉姆直线演算法判断溅射原子到哪个格点了
    if(abs(k_xy)<1):
        #要求斜率绝对值小于1，所以两种情况分开讨论
        error=0
        x,y,z=x0,y0,z0
        x_old,y_old=x0,y0
        while(True):
            #碰到表面之前粒子运动不停止
            #更新到下一个格点，根据溅射原子高度判断是否到达表面
            error+=abs(k_xy)
            x=((x+np.sign(vx))+par['N1'])%par['N1']#周期性边界条件
            z=z-vz/abs(vx)
            if(error>=1):
                #若误差大于1，则y坐标更新
                y=((y+np.sign(vy))+par['N2'])%par['N2']#周期性边界条件
                error-=1
            
            if(h_list[int(x)-1][int(y)-1]>z):
                #如果小于当前点表面高度，则判断被上一个点吸附
                h_list[int(x_old)-1][int(y_old)-1]+=1
                break
            x_old,y_old=x,y
    else:
        #同上
        error=0
        x,y,z=x0,y0,z0
        x_old,y_old=x0,y0
        while(True):
            #碰到表面之前粒子运动不停止
            #更新到下一个格点，根据溅射原子高度判断是否到达表面
            error+=abs(1/k_xy)
            y=((y+np.sign(vy))+par['N2'])%par['N2']#周期性边界条件

            z=z-vz/abs(vy)
            if(error>=1):
                #若误差大于1，则y坐标更新
                x=((x+np.sign(vx))+par['N1'])%par['N1']#周期性边界条件
                error-=1
            
            if(h_list[int(x)-1][int(y)-1]>z):
                #如果小于当前点表面高度，则判断被上一个点吸附
                h_list[int(x_old)-1][int(y_old)-1]+=1
                break
            x_old,y_old=x,y


if __name__=='__main__':
    #初始化参数
    N1,N2=1024,1024#晶格在x,y方向的宽度
    h_list=np.array([[0 for i in range(N1)] for j in range(N2)])#生成每个晶格位置的高度
    N_particle=100000003#粒子数目 
    par={'N1':N1,'N2':N2}#晶格参数
    t0=time()
    t=1
    for i in range(N_particle):
        deposition_one(h_list,**par)
        if(i==t):
            print('第%g个,平均数%.3g,方差%.3g,'%(i,np.mean(h_list),np.var(h_list)))
            t*=10
            np.savetxt('h-N1_%g-N2_%g-N_p_%g.txt'%(N1,N2,N_particle),h_list)
            print('运行时间为%.1fs'%(time()-t0))
    '''数据保存'''
    np.savetxt('h-N1_%g-N2_%g-N_p_%g.txt'%(N1,N2,N_particle),h_list)
    
    '''三、绘图'''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    im1=ax.imshow(h_list,cmap=plt.cm.hot,origin="lower",interpolation='bilinear',aspect='auto')
    plt.colorbar(im1)
    plt.show()