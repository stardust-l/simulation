from numpy.core.function_base import linspace
from scipy.optimize import fsolve,fmin,minimize
import numpy as np
import matplotlib.pyplot as plt
import os

class Magnetism_system(object):
    def __init__(self) -> None:
        #将当前目录转化为默认路径
        current_path=os.path.abspath(__file__)
        current_fold = os.path.dirname(current_path)
        os.chdir(current_fold)

        theta0_common_init=[0.3,np.pi/2+0.3,0.1,0.1]#初始theta1,theta2
        self.theta0_common=[theta0_common_init[:]]#用于存储系统所有的状态
        self.data=[]#用于存储数据
        self.error=1*np.pi/180#偏离一定角度
        self.par_2D=dict(
            A1=1/3,
            K2=4.0/3, #Co垂直各向异性
            M_Co=1,
            eta_TIG=0.95,
            eta_Co=0.95,
            M_TIG_Co=0.05#仅计算时使用
            )

    def E_all_2D(self,theta,H_var):
        #系统能量项，包含各向异性能，交换能，测量磁场引起的塞曼能
        #两层模型
        #参数设置
        M1=1
        K2=self.par_2D['K2']
        A1=self.par_2D['A1']
        eta_TIG=self.par_2D['eta_TIG']
        eta_Co=self.par_2D['eta_Co']
        M2=self.par_2D['M_Co']
        
        theta1,theta2,phi1,phi2=theta.tolist()
        H,theta_H,phi_H=H_var
        #print(H_var)
        while(True and (theta1<0 or theta1>np.pi)):
            if(theta1<0):
                theta1=-theta1
                phi1=phi1+np.pi
            elif(theta1>=np.pi):
                theta1=2*np.pi-theta1
                phi1=phi1+np.pi
        #print(theta1)
        while(True and (theta2<0 or theta2>np.pi)):
            if(theta2<0):
                theta2=-theta2
                phi2=phi2+np.pi
            elif(theta2>np.pi):
                theta2=2*np.pi-theta2
                phi2=phi2+np.pi
        E_k_TIG=np.power(np.sin(theta1),2)*(eta_TIG+(1-eta_TIG)*np.power(np.sin(phi1),2))
        #-1.0/96*((21+4*np.cos(2*theta1))+7*np.cos(4*theta1)-4*np.sqrt(2)*np.cos(2*theta1-3*phi1)+2*np.sqrt(2)*np.cos(4*theta1-3*phi1)+4*np.sqrt(2)*np.cos(2*theta1+3*phi1)-2*np.sqrt(2)*np.cos(4*theta1+3*phi1))
        E_k_Co=-K2*np.power(np.sin(theta2),2)*(eta_Co+(1-eta_Co)*np.power(np.sin(phi2),2))
        E_k=E_k_TIG+E_k_Co
        E_ex=A1*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)+np.cos(theta1)*np.cos(theta2))
        E_zeeman=-H*M1*(np.sin(theta1)*np.sin(theta_H)*np.cos(phi1-phi_H)+np.cos(theta1)*np.cos(theta_H))-H*M2*(np.sin(theta2)*np.sin(theta_H)*np.cos(phi2-phi_H)+np.cos(theta2)*np.cos(theta_H))
        E=E_k+E_ex+E_zeeman
        return E
    def get_Mz_2D(self,H,theta0=None,**parm):
        #获取每个外场下的稳定态下的垂直方向的磁性

        #res=fsolve(f,theta0,args=(H,))
        if(type(theta0)!=type(None)):
            self.theta0_common[0]=theta0#确定初始态
        #bnd=((0,np.pi),(0,np.pi),(0,2*np.pi),(0,np.pi)) 
        #theta=minimize(self.E_all_2D,x0=self.theta0_common[-1],args=(H,),bounds=bnd).x #,xtol=1e-10,ftol=1e-10 ,bounds=bnd
        theta=fmin(self.E_all_2D,x0=self.theta0_common[-1],args=(H,),disp=0,xtol=1e-10,ftol=1e-10)
        
        if(False):
            if(abs(np.sum(f(theta,H)**2))>1e-2):
                print('H为',H,'解为',theta,'解有问题',f(theta,H))
        theta1,theta2,phi1,phi2=theta.tolist()
        while(True and (theta1<0 or theta1>np.pi)):
            if(theta1<0):
                theta1=-theta1
                phi1=phi1+np.pi
            elif(theta1>=np.pi):
                theta1=2*np.pi-theta1
                phi1=phi1+np.pi
        #print(theta1)
        while(True and (theta2<0 or theta2>np.pi)):
            if(theta2<0):
                theta2=-theta2
                phi2=phi2+np.pi
            elif(theta2>np.pi):
                theta2=2*np.pi-theta2
                phi2=phi2+np.pi
        #phi1,phi2=phi1%(np.pi*2),phi2%(np.pi*2)
        self.theta0_common.append([theta1,theta2,phi1,phi2])
        Mz=parm['M_TIG_Co']*np.cos(theta1)+parm['M_Co']*np.cos(theta2) #np.cos(theta1)+
        return Mz
    def get_Mz_2D_list(self,H_list,theta0=None,**parm):
        #获取一系列垂直方向的磁性

        #res=fsolve(f,theta0,args=(H,))
        data=[]
        for H_i in H_list:
            x=H_i[0]#计算磁场大小
            y=self.get_Mz_2D(H_i[1:],theta0,**parm)#计算稳态下磁矩的角度
            data.append([x,y,*self.theta0_common[-1]])
        self.theta0_common=[self.theta0_common[-1]]#将角度历史重置
        return np.array(data).T
    def plot_Mz_2D(self,f,x_start,x_end,n_point=200,**par):
        x1=np.linspace(x_start,x_end,n_point)
        x2=np.linspace(x_end,x_start,n_point)
        "AHE"
        if(True):
            #AHE的设置，theta角由0变为np.pi，phi始终为np.pi
            x_AHE=[]
            
            for xi in x1:
                if(xi>=0):
                    theta_H=self.error
                else:
                    theta_H=np.pi+self.error
                x_AHE.append([xi,abs(xi),theta_H,self.error])
            data_down_to_up_AHE=self.get_Mz_2D_list(x_AHE,[0.1,np.pi/2+0.1,0,0],**par)
            plt.plot(*data_down_to_up_AHE[[0,1]],'ko-',label='AHE')
            if(False):
                plt.plot(*data_down_to_up_AHE[[0,2]],'g--',label='theta_YIG')
                plt.plot(*data_down_to_up_AHE[[0,3]],'b--',label='theta_Co')
                plt.plot(*data_down_to_up_AHE[[0,4]],'g:',label='phi_YIG')
                plt.plot(*data_down_to_up_AHE[[0,5]],'b:',label='phi_Co')
            plt.legend()

        if(True):
            #PHE的设置，theta角始终为np.pi/2，phi由np.pi变为0
            "PHE"
            x_PHE=[]
            for xi in x1:
                if(xi>=0):
                    theta_H=np.pi/2-self.error
                    phi_H=self.error
                else:
                    theta_H=np.pi/2+self.error
                    phi_H=np.pi+self.error
                #x_PHE.append([xi,abs(xi),np.pi/2+self.error,phi_H])
                x_PHE.append([xi,abs(xi),theta_H,phi_H])
            data_down_to_up_PHE=self.get_Mz_2D_list(x_PHE,[0.2,np.pi/2+0.1,0,0],**par)
            data_up_to_down_PHE=self.get_Mz_2D_list(x_PHE[::-1],None,**par)
            #self.par_list_2D[2]=(0.1,np.pi/2+0.1,0,0)
            plt.plot(*data_down_to_up_PHE[[0,1]],'ro-',label='PHE')
            #plt.plot(*data_up_to_down_PHE[[0,1]],'go-')
            if(True):
                plt.plot(*data_down_to_up_PHE[[0,2]],'g--',label='theta_YIG')
                plt.plot(*data_down_to_up_PHE[[0,3]],'b--',label='theta_Co')
                plt.plot(*data_down_to_up_PHE[[0,4]],'g:',label='phi_YIG')
                plt.plot(*data_down_to_up_PHE[[0,5]],'b:',label='phi_Co')
            plt.legend()

        #data_down_to_up_PHE=self.get_Mz_2D_list(x_PHE[::-1],[0.2,np.pi/2+0.1,0,0],**par)
        #plt.plot(*data_down_to_up_PHE[[0,1]],'ko--',label='PHE')
        s=''

        #将参数添加到图像中
        for key,value in par.items():
            if(type(value)!=type((0,1))):
                s+=key+':'+'{:.3g}\n'.format(value)
            else:
                s+=key+':'+'{:.3g}-{:.3g}\n'.format(*value)
        ax=plt.gca()
        plt.text(0.7,0.5,s=s,transform=ax.transAxes)
        #plt.plot(x2,[f(xi,[0.1+np.pi,np.pi*3/2+0.1]) for xi in x2],'ko-')

        #将参数保存到文件中
        s=''
        for key,value in par.items():
            if(type(value)!=type((0,1))):
                s+=key+'&'+'{:.3g}\n'.format(value)
            else:
                s+=key+'&'+'{:.3g}-{:.3g}\n'.format(*value)
        #plt.savefig('out/'+s.replace('\n','+')+'.png')
        plt.show()

system=Magnetism_system()
system.plot_Mz_2D(system.get_Mz_2D,-15,15,**system.par_2D)
#angle_history=np.array(system.theta0_common[::5])*180/np.pi%360
#np.savetxt('out/angle_list.txt',angle_history,fmt='%.2f')