import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wrf_module as wrf
import metpy.calc as mpcalc 
from metpy.cbook import get_test_data
from metpy.plots import SkewT , Hodograph
from metpy.units import units

#Parametros para controlar tamaño de letra y tamaño de figura
labelsize_colorbar= 9.0
dpi=300

def plot_momentum_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound=None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)


    for it in range( my_data['nt'] ) :
        
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/momentum_equation_w_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/momentum_equation_w_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          t=my_data['t'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          bt=my_data['dwdt_bt'][:,:,sw,it]
          bqv=my_data['dwdt_bqv'][:,:,sw,it]
          bqc=my_data['dwdt_bqc'][:,:,sw,it]
          bp =my_data['dwdt_bp'][:,:,sw,it]
          dwdt =my_data['dwdt_loc'][:,:,sw,it]
          pz =my_data['dwdt_pz'][:,:,sw,it]
          adv =my_data['dwdt_adv'][:,:,sw,it]
          ppert = my_data['p'][:,:,sw,it] - my_data['p0'][:,:,sw,it]
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          t=my_data['t'][:,sw,:,it]
          qv=my_data['qv'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          bt=my_data['dwdt_bt'][:,sw,:,it]
          bqv=my_data['dwdt_bqv'][:,sw,:,it]
          bqc=my_data['dwdt_bqc'][:,sw,:,it]
          bp =my_data['dwdt_bp'][:,sw,:,it]
          dwdt =my_data['dwdt_loc'][:,sw,:,it]
          pz =my_data['dwdt_pz'][:,sw,:,it]
          adv =my_data['dwdt_adv'][:,sw,:,it]
          ppert = my_data['p'][:,sw,:,it] - my_data['p0'][:,sw,:,it]
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          t=my_data['t'][1,:,:,it]
          qv=my_data['qv'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          bt=my_data['dwdt_bt'][1,:,:,it]
          bqv=my_data['dwdt_bqv'][1,:,:,it]
          bqc=my_data['dwdt_bqc'][1,:,:,it]
          bp =my_data['dwdt_bp'][1,:,:,it]
          dwdt =my_data['dwdt_loc'][1,:,:,it]
          pz =my_data['dwdt_pz'][1,:,:,it]
          adv =my_data['dwdt_adv'][1,:,:,it]
          ppert = my_data['p'][1,:,:,it] - my_data['p0'][1,:,:,it]
          
      
       ncols=4
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex = True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #Ploteo la reflectividad y la velocidad vertical.
       ax = axs[0,0]
       clevs1=np.arange(0,70,1) * scale_factor 
       my_map = cmap_discretize('gist_ncar',clevs1.size)
       p1=ax.contourf( x , y , ref , clevs1 , cmap=my_map)
       clevs2=np.array([1.0,5.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0,70.0]) * scale_factor
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='solid' )
       clevs2=np.array([-30.0,-20.0,-15.0,-10.0,-5.0,-1.0]) * scale_factor 
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='dashed' )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title('Reflectividad (sh, dBZ) y W (cont., $ms^{-1}$)')
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(ref)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)
       ax.grid()
       #ax.set_xticks([])

       #Fijo los niveles para las componentes del empuje.
       clevs1=np.arange(-0.8,0.8+0.05,0.05)
       my_map = cmap_discretize('bwr',clevs1.size)

       #Ploteo el aporte de T al empuje y la temperatura
       ax = axs[0,1]
       p1=ax.contourf( x , y , bt , clevs1 , cmap=my_map)
       clevs2=np.arange(100,320,5)
       p2=ax.contour( x , y , t , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )

       ax.set_title('$B_T$ (sh., $ms^{-2}$) y T (cont., K)')
       ax.grid()
       #ax.set_xticks([])
       #ax.set_yticks([])

       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(bt)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[0,2]
       p1=ax.contourf( x , y , bqv , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y , qv * 1000 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )

       ax.set_title('$B_{q_v}$ (sh., $ms^{-2}$) y $q_v$ (cont., K)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()       

       #Ploteo el aporte de qc al empuje
       ax = axs[0,3]
       p1=ax.contourf( x , y , bqc , clevs1 , cmap=my_map)
       clevs2=np.arange(0.1,1.0,0.1) * scale_factor 
       p2=ax.contour( x , y , bt + bp + bqv + bqc , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-1.0,0.0,0.1) * scale_factor
       p2=ax.contour( x , y , bt + bp + bqv + bqc , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )

       ax.set_title('$B_{q_c}$ (sh., $ms^{-2}$) y B (cont., $ms^{-2}$) ')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()       

       #Ploteo el aporte de P al empuje
       ax = axs[1,0]
       p1=ax.contourf( x , y , bp , clevs1 , cmap=my_map) 
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )

       ax.set_title('$B_P$ (sh., $ms^{-2}$) y viento(U,W) (vectores)')
       ax.grid()

       #Ploteo la componente vertical de la fuerza de presion y las perturbaciones de presion.
       ax = axs[1,1]
       p1=ax.contourf( x , y , pz , clevs1 , cmap=my_map)
       clevs2=np.arange(25,500,50) * scale_factor
       p2=ax.contour( x , y , ppert , clevs2 , colors='k',linestyles='solid' )
       clevs3=np.arange(-500,0,50) * scale_factor
       p3=ax.contour( x , y , ppert , clevs3 , colors='k',linestyles='dashed' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$-{\frac{1}{{\rho}_0}}{{\nabla}_z}{P}^{\prime}$ (sh., $ms^{-2}$) y ${P}^{\prime}$ (cont., Pa)')
       #ax.set_yticks([])
       ax.grid()
       
       #Ploteo el termino advectivo.
       ax = axs[1,2]
       p1=ax.contourf( x , y , adv , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_title(r'$-{V}{\nabla}{w}$ (sh., ms{-2})')
       #ax.set_yticks([])
       #Ploteo la aceleracion local y el residuo.
       ax = axs[1,3]
       p1=ax.contourf( x , y , dwdt , clevs1 , cmap=my_map)
       clevs2=np.arange(0.05,0.5,0.2) * scale_factor
       p2=ax.contour( x , y , dwdt - adv - pz - bt -bqv -bqc -bp , clevs2 , colors='k' , linestyles='solid' )
       clevs2=np.arange(-0.5,-0.05,0.2) * scale_factor
       p3=ax.contour( x , y , dwdt - adv - pz - bt -bqv -bqc -bp , clevs2 , colors='k' , linestyles='dashed' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       
       ax.set_title(r'${\frac{{\partial}w}{{\partial}t}}}$ (sh., $ms^{-2}$), Residuo (cont. $ms^-2$)')
       #ax.set_yticks([])
       ax.grid()

       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return


def plot_momentum_equation_2_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :
        
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/momentum_equation_w_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/momentum_equation_w_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          t=my_data['t'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          bt=my_data['dwdt_bt'][:,:,sw,it]
          bqv=my_data['dwdt_bqv'][:,:,sw,it]
          bqc=my_data['dwdt_bqc'][:,:,sw,it]
          bp =my_data['dwdt_bp'][:,:,sw,it]
          dwdt =my_data['dwdt_loc'][:,:,sw,it]
          pz =my_data['dwdt_pz'][:,:,sw,it]
          adv =my_data['dwdt_adv'][:,:,sw,it]
          ppert = my_data['p'][:,:,sw,it] - my_data['p0'][:,:,sw,it]
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          t=my_data['t'][:,sw,:,it]
          qv=my_data['qv'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          bt=my_data['dwdt_bt'][:,sw,:,it]
          bqv=my_data['dwdt_bqv'][:,sw,:,it]
          bqc=my_data['dwdt_bqc'][:,sw,:,it]
          bp =my_data['dwdt_bp'][:,sw,:,it]
          dwdt =my_data['dwdt_loc'][:,sw,:,it]
          pz =my_data['dwdt_pz'][:,sw,:,it]
          adv =my_data['dwdt_adv'][:,sw,:,it]
          ppert = my_data['p'][:,sw,:,it] - my_data['p0'][:,sw,:,it]
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          t=my_data['t'][1,:,:,it]
          qv=my_data['qv'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          bt=my_data['dwdt_bt'][1,:,:,it]
          bqv=my_data['dwdt_bqv'][1,:,:,it]
          bqc=my_data['dwdt_bqc'][1,:,:,it]
          bp =my_data['dwdt_bp'][1,:,:,it]
          dwdt =my_data['dwdt_loc'][1,:,:,it]
          pz =my_data['dwdt_pz'][1,:,:,it]
          adv =my_data['dwdt_adv'][1,:,:,it]
          ppert = my_data['p'][1,:,:,it] - my_data['p0'][1,:,:,it]
      
       ncols=4
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex = True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #Ploteo la reflectividad y la velocidad vertical.
       ax = axs[0,0]
       clevs1=np.arange(0,70,1) * scale_factor 
       my_map = cmap_discretize('gist_ncar',clevs1.size)
       p1=ax.contourf( x , y , ref , clevs1 , cmap=my_map)
       clevs2=np.array([1.0,5.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0,70.0]) * scale_factor
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='solid' )
       clevs2=np.array([-30.0,-20.0,-15.0,-10.0,-5.0,-1.0]) * scale_factor 
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='dashed' )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title('Reflectividad (sh, dBZ) y W (cont., $ms^{-1}$)')
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(ref)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #ax.set_xticks([])

       #Fijo los niveles para las componentes del empuje.
       clevs1=np.arange(-0.8,0.8+0.05,0.05)
       my_map = cmap_discretize('bwr',clevs1.size)

       #Ploteo el aporte de T al empuje y la temperatura
       ax = axs[0,1]
       p1=ax.contourf( x , y , bt , clevs1 , cmap=my_map)
       clevs2=np.arange(100,320,5)
       p2=ax.contour( x , y , t , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title('$B_T$ (sh., $ms^{-2}$) y T (cont., K)')
       ax.grid()
       #ax.set_xticks([])
       #ax.set_yticks([])

       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(bt)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[0,2]
       p1=ax.contourf( x , y , bqv , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y , qv * 1000 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title('$B_{q_v}$ (sh., $ms^{-2}$) y $q_v$ (cont., K)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()       

       #Ploteo el aporte de qc al empuje
       ax = axs[0,3]
       p1=ax.contourf( x , y , bqc , clevs1 , cmap=my_map)
       clevs2=np.arange(0.1,1.0,0.1) * scale_factor 
       p2=ax.contour( x , y , bt + bp + bqv + bqc , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-1.0,0.0,0.1) * scale_factor
       p2=ax.contour( x , y , bt + bp + bqv + bqc , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title('$B_{q_c}$ (sh., $ms^{-2}$) y Empuje total (cont., $ms^{-2}$) ')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()       

       #Ploteo el aporte de P al empuje
       ax = axs[1,0]
       p1=ax.contourf( x , y , bp , clevs1 , cmap=my_map) 
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title('$B_P$ (sh., $ms^{-2}$) y viento(U,W) (vectores)')
       ax.grid()

       #Ploteo la componente vertical de la fuerza de presion y las perturbaciones de presion.
       ax = axs[1,1]
       p1=ax.contourf( x , y , pz , clevs1 , cmap=my_map)
       clevs2=np.arange(25,500,50) * scale_factor
       p2=ax.contour( x , y , ppert , clevs2 , colors='k',linestyles='solid' )
       clevs3=np.arange(-500,0,50) * scale_factor
       p3=ax.contour( x , y , ppert , clevs3 , colors='k',linestyles='dashed' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$-{\frac{1}{{\rho}_0}}{{\nabla}_z}{P}^{\prime}$ (sh., $ms^{-2}$) y ${P}^{\prime}$ (cont., Pa)')
       #ax.set_yticks([])
       ax.grid()

       #Ploteo el residuo.
       ax = axs[1,2]
       p1=ax.contourf( x , y , dwdt - adv - pz - bt -bqv -bqc -bp , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'Residuo (sh. $ms^-2$)')
       #ax.set_yticks([])
       ax.grid()

       
       #Ploteo el residuo
       ax = axs[1,3]
       p1=ax.contourf( x , y , dwdt - adv , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'${\frac{{d}w}{{d}t}}}$ (sh., $ms^{-2}$)')
       #ax.set_yticks([])

       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return

def plot_ref_vertvel( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None) :
    #Ploteo un cross section de la reflectividad y la velocidad vertical

    for it in range( my_data['nt'] ) :
        
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/ref_vertvel_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/ref_vertvel_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          t=my_data['t'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          t=my_data['t'][:,sw,:,it]
          qv=my_data['qv'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          t=my_data['t'][1,:,:,it]
          qv=my_data['qv'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
      
       ncols=4
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]

       #Ploteo la reflectividad y la velocidad vertical.
       fig,ax=plt.subplots(1,1,figsize=[5,5])
       clevs1=np.arange(0,70,1) * scale_factor 
       my_map = cmap_discretize('gist_ncar',clevs1.size)
       p1=ax.contourf( x , y , ref , clevs1 , cmap=my_map)
       clevs2=np.array([1.0,5.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0,70.0]) * scale_factor
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='solid' )
       clevs2=np.array([-30.0,-20.0,-15.0,-10.0,-5.0,-1.0]) * scale_factor 
       p2=ax.contour( x , y , w , clevs2 , colors='k',linestyles='dashed' )
       #Plot the wind
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)

       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title('Reflectividad (sh, dBZ) & W (cont., $ms^{-1}$)')
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(ref)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)


       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return

def plot_horwind_theta( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None) :
    #Ploteo un cross section de la reflectividad y la velocidad vertical

    for it in range( my_data['nt'] ) :
        
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/horwind_theta_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/horwind_theta_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1)) / 1000.0
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          th=my_data['theta'][:,:,sw,it] - my_data['theta'][:,:,sw,0]
          w=my_data['w'][:,:,sw,it] - my_data['w'][:,:,sw,0]
          hwind=my_data['v'][:,:,sw,it] - my_data['v'][:,:,sw,0]

       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1)) / 1000.0
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          th=my_data['theta'][:,sw,:,it] - my_data['theta'][:,sw,:,0]
          w=my_data['w'][:,sw,:,it] - my_data['w'][:,sw,:,0]
          hwind=my_data['u'][:,sw,:,it] - my_data['u'][:,sw,:,0]
 
      
       ncols=4
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]

       #Plot horizontal wind anomaly.
       fig,ax=plt.subplots(1,1,figsize=[5,5])
       clevs1=np.arange(-25,25,1) * scale_factor 
       my_map = cmap_discretize('bwr',clevs1.size)
       p1=ax.contourf( x , y , hwind , clevs1 , cmap=my_map)
       clevs2=np.array([1.0,2.5,5.0,7.5,10.0,12.5,15.0]) * scale_factor
       p2=ax.contour( x , y , th , clevs2 , colors='k',linestyles='solid' )
       clevs2=np.array([-15.0,-12.5,-10.0,-7.5,-5.0,-2.5,-1.0]) * scale_factor 
       p2=ax.contour( x , y , th , clevs2 , colors='k',linestyles='dashed' )
       #Plot the wind
       skipx=5
       skipz=1
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)

       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title('Horizontal wind anom. (sh, $ms^{-1}$) & Theta anom. (cont., K)')
       cbar_ax = fig.add_axes([0.36, 0.03, 0.5, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(hwind)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)


       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return





def plot_termo_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/termo_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/termo_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'


       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          theta=my_data['theta'][:,:,sw,it]
          theta_pert=my_data['theta'][:,:,sw,it]-my_data['theta0'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          dthetadt_loc =my_data['dthetadt_loc'][:,:,sw,it]
          dthetadt_adv =my_data['dthetadt_adv'][:,:,sw,it]
          h_diabatic =my_data['h_diabatic'][:,:,sw,it]                   
          
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          theta=my_data['theta'][:,sw,:,it]
          theta_pert=my_data['theta'][:,sw,:,it]-my_data['theta0'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          dthetadt_loc =my_data['dthetadt_loc'][:,sw,:,it]
          dthetadt_adv =my_data['dthetadt_adv'][:,sw,:,it]
          h_diabatic =my_data['h_diabatic'][:,sw,:,it]

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          theta=my_data['theta'][1,:,:,it]
          theta_pert=my_data['theta'][1,:,:,it]-my_data['theta0'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          dthetadt_loc =my_data['dthetadt_loc'][1,:,:,it]
          dthetadt_adv =my_data['dthetadt_adv'][1,:,:,it]
          h_diabatic =my_data['h_diabatic'][1,:,:,it]
          
    
       ncols=2
       nrows=2
       fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
 
       #Ploteo theta y su perturbacion
       ax = axs[0,0]
       clevs1=np.arange(-15,15.5,0.5) * scale_factor
       theta_pert[theta_pert > np.max(clevs1)]=np.max(clevs1)
       theta_pert[theta_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , theta_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\theta}^{\prime}$ (sh., $K$) y $\theta$ (cont., $K$)')   
       #ax.set_xticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(theta_pert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Defino la escala de colores para la tasa de cambio de theta con el tiempo.
       clevs1=np.arange(-0.15,0.16,0.01)  * scale_factor

       #Ploteo la tasa de cambio local de theta con el tiempo
       ax = axs[0,1]
       dthetadt_loc[dthetadt_loc > np.max(clevs1)]=np.max(clevs1)
       dthetadt_loc[dthetadt_loc < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('bwr',clevs1.size)
       p1=ax.contourf( x , y , dthetadt_loc , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{\partial{\theta}}{{\partial}t}}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dthetadt_loc)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[1,0]
       my_map = cmap_discretize('bwr',clevs1.size)
       dthetadt_adv[dthetadt_adv > np.max(clevs1)]=np.max(clevs1)
       dthetadt_adv[dthetadt_adv < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dthetadt_adv , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-V{\nabla}{\theta}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$) y viento (U,W) (vectores)')
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)

       #Ploteo el aporte del calor latente. 
       ax = axs[1,1]
       my_map = cmap_discretize('bwr',clevs1.size)
       h_diabatic[h_diabatic > np.max(clevs1)]=np.max(clevs1)
       h_diabatic[h_diabatic < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , h_diabatic , clevs1 , cmap=my_map)
       clevs2=np.array([0.01,0.05,0.1,0.2]) * scale_factor
       p2=ax.contour( x , y , dthetadt_loc - dthetadt_adv - h_diabatic , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.array([-0.2,-0.1,-0.05,-0.01]) * scale_factor
       p2=ax.contour( x , y , dthetadt_loc - dthetadt_adv - h_diabatic , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$\dot{Q}_{lat}$ (sh., $Ks^{-1}$) y ${\frac{d{\theta}}{dt}}_{res}$ (cont., $K$)')
       #ax.set_yticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return


def plot_termo_equation_2_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/termo_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/termo_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          theta=my_data['theta'][:,:,sw,it]
          theta_pert=my_data['theta'][:,:,sw,it]-my_data['theta0'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          dthetadt_loc =my_data['dthetadt_loc'][:,:,sw,it]
          dthetadt_adv =my_data['dthetadt_adv'][:,:,sw,it]
          h_diabatic =my_data['h_diabatic'][:,:,sw,it]                   
          
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          theta=my_data['theta'][:,sw,:,it]
          theta_pert=my_data['theta'][:,sw,:,it]-my_data['theta0'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          dthetadt_loc =my_data['dthetadt_loc'][:,sw,:,it]
          dthetadt_adv =my_data['dthetadt_adv'][:,sw,:,it]
          h_diabatic =my_data['h_diabatic'][:,sw,:,it]

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          theta=my_data['theta'][1,:,:,it]
          theta_pert=my_data['theta'][1,:,:,it]-my_data['theta0'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          dthetadt_loc =my_data['dthetadt_loc'][1,:,:,it]
          dthetadt_adv =my_data['dthetadt_adv'][1,:,:,it]
          h_diabatic =my_data['h_diabatic'][1,:,:,it]
          
    
       ncols=2
       nrows=2
       fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
 
       #Ploteo theta y su perturbacion
       ax = axs[0,0]
       clevs1=np.arange(-15,15.5,0.5) * scale_factor
       theta_pert[theta_pert > np.max(clevs1)]=np.max(clevs1)
       theta_pert[theta_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , theta_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y  , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y  , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'${\theta}^{\prime}$ (sh., $K$) y $\theta$ (cont., $K$)')   
       #ax.set_xticks([])

       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(theta_pert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Defino la escala de colores para la tasa de cambio de theta con el tiempo.
       clevs1=np.arange(-0.15,0.16,0.01)  * scale_factor

       #Ploteo la tasa de cambio local de theta con el tiempo
       ax = axs[0,1]
       dthetadt_tot = dthetadt_loc - dthetadt_adv
       dthetadt_tot[dthetadt_tot > np.max(clevs1)]=np.max(clevs1)
       dthetadt_tot[dthetadt_tot < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('bwr',clevs1.size)
       p1=ax.contourf( x , y  , dthetadt_tot , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{d{\theta}}{dt}}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$) y viento (U,W) (vectores)')
       ax.grid()
       #ax.set_xticks([])
       #ax.set_yticks([])
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dthetadt_loc)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[1,0]
       my_map = cmap_discretize('bwr',clevs1.size)
       h_diabatic[h_diabatic > np.max(clevs1)]=np.max(clevs1)
       h_diabatic[h_diabatic < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , h_diabatic , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.grid()
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$\dot{Q}_{lat}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$)')

       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)

       #Ploteo el aporte del calor latente. 
       ax = axs[1,1]
       res = dthetadt_loc - dthetadt_adv - h_diabatic
       my_map = cmap_discretize('bwr',clevs1.size)
       res[res > np.max(clevs1)]=np.max(clevs1)
       res[res < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y  , res , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( x , y , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.grid()
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{d{\theta}}{dt}}_{res}$ (sh., $K$) y $\theta$ (cont., $K$)')
       #ax.set_yticks([])
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()

def plot_termo_equation_hov_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , h_ini=None , h_end=None  ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    fig_name = plot_path + '/termo_equation_hov_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '.png'
    if my_data['slice_type'] == 'vx' :
       dist=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
       dist=np.tile(dist,(my_data['nz'],1))
       sw=my_data['slice_width']
       z=np.mean( my_data['z'][:,h_ini:h_end,sw,:] , 1)
       theta=np.mean( my_data['theta'][:,h_ini:h_end,sw,:] , 1)
       theta_pert=np.mean( my_data['theta'][:,h_ini:h_end,sw,:]-my_data['theta0'][:,h_ini:h_end,sw,:] , 1)
       ref=np.mean( my_data['ref'][:,:,sw,:] , 1)
       dthetadt_loc =np.mean( my_data['dthetadt_loc'][:,h_ini:h_end,sw,:] , 1)
       dthetadt_adv =np.mean( my_data['dthetadt_adv'][:,h_ini:h_end,sw,:] , 1)
       h_diabatic =np.mean( my_data['h_diabatic'][:,h_ini:h_end,sw,:] , 1)          

          
    if my_data['slice_type'] == 'vy' :
       dist=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
       dist=np.tile(dist,(my_data['nz'],1))
       sw=my_data['slice_width']
       z=np.mean( my_data['z'][:,sw,h_ini:h_end,:] , 1)
       theta=np.mean( my_data['theta'][:,sw,h_ini:h_end,:] , 1)
       theta_pert=np.mean( my_data['theta'][:,sw,h_ini:h_end,:]-my_data['theta0'][:,sw,h_ini:h_end,:] , 1)
       ref=np.mean( my_data['ref'][:,sw,h_ini:h_end,:] , 1)
       dthetadt_loc=np.mean( my_data['dthetadt_loc'][:,sw,h_ini:h_end,:] , 1)
       dthetadt_adv=np.mean( my_data['dthetadt_adv'][:,sw,h_ini:h_end,:] , 1)
       h_diabatic=np.mean( my_data['h_diabatic'][:,sw,h_ini:h_end,:] , 1)
  
    ncols=2
    nrows=2
    fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
    fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
    if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' and ybound is None  :
       ybound=[0,20000]
    time=np.tile( np.arange(0,z.shape[1]) , (z.shape[0],1) )
      #Ploteo theta y su perturbacion
    ax = axs[0,0]
    clevs1=np.arange(-15,15.5,0.5) * scale_factor
    theta_pert[theta_pert > np.max(clevs1)]=np.max(clevs1)
    theta_pert[theta_pert < np.min(clevs1)]=np.min(clevs1)
    my_map = cmap_discretize('RdBu_r',clevs1.size)
    p1=ax.contourf( time , z , theta_pert , clevs1 , cmap=my_map)
    clevs2=np.arange(270,500,5)
    p2=ax.contour( time , z , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
    p3=ax.contour( time , z , ref , [15.0] , colors='c',linestyles='solid' , linewidths=2.5 )
    ax.set_ybound( ybound )
    ax.grid()
    ax.set_title(r'${\theta}^{\prime}$ (sh., $K$) y $\theta$ (cont., $K$)')   
    #ax.set_xticks([])

    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
    cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
    m = plt.cm.ScalarMappable(cmap=my_map )
    m.set_array(theta_pert)
    m.set_clim(np.min(clevs1),np.max(clevs1))
    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
    cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
    cb.ax.tick_params(labelsize=labelsize_colorbar)

    #Defino la escala de colores para la tasa de cambio de theta con el tiempo.
    clevs1=np.arange(-0.15,0.16,0.01)  * scale_factor
    #Ploteo la tasa de cambio local de theta con el tiempo
    ax = axs[0,1]
    dthetadt_tot = dthetadt_loc - dthetadt_adv
    dthetadt_tot[dthetadt_tot > np.max(clevs1)]=np.max(clevs1)
    dthetadt_tot[dthetadt_tot < np.min(clevs1)]=np.min(clevs1)
    my_map = cmap_discretize('bwr',clevs1.size)
    p1=ax.contourf( time , z , dthetadt_tot , clevs1 , cmap=my_map)
    clevs2=np.arange(270,500,5)
    p2=ax.contour( time , z , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
    p3=ax.contour( time , z , ref , [15.0] , colors='c',linestyles='solid' , linewidths=2.5 )
    ax.set_ybound( ybound )
    ax.set_title(r'${\frac{d{\theta}}{dt}}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$) y viento (U,W) (vectores)')
    ax.grid()
    #ax.set_xticks([])
    #ax.set_yticks([])
    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
    cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
    m = plt.cm.ScalarMappable(cmap=my_map )
    m.set_array(dthetadt_loc)
    m.set_clim(np.min(clevs1),np.max(clevs1))
    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
    cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
    cb.ax.tick_params(labelsize=labelsize_colorbar)
    #Ploteo el aporte de qv al empuje y qv.
    ax = axs[1,0]
    my_map = cmap_discretize('bwr',clevs1.size)
    h_diabatic[h_diabatic > np.max(clevs1)]=np.max(clevs1)
    h_diabatic[h_diabatic < np.min(clevs1)]=np.min(clevs1)
    p1=ax.contourf( time , z , h_diabatic , clevs1 , cmap=my_map)
    clevs2=np.arange(270,500,5)
    p2=ax.contour( time , z , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
    p3=ax.contour( time , z , ref , [15.0] , colors='c',linestyles='solid' , linewidths=2.5 )
    ax.grid()
    ax.set_ybound( ybound )
    ax.set_title(r'$\dot{Q}_{lat}$ (sh., $Ks^{-1}$) y $\theta$ (cont., $K$)')

    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
    #Ploteo el aporte del calor latente. 
    ax = axs[1,1]
    res = dthetadt_loc - dthetadt_adv - h_diabatic
    my_map = cmap_discretize('bwr',clevs1.size)
    res[res > np.max(clevs1)]=np.max(clevs1)
    res[res < np.min(clevs1)]=np.min(clevs1)
    p1=ax.contourf( time , z , res , clevs1 , cmap=my_map)
    clevs2=np.arange(270,500,5)
    p2=ax.contour( time , z , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
    p3=ax.contour( time , z , ref , [15.0] , colors='c',linestyles='solid' , linewidths=2.5 )
    ax.grid()

    ax.set_ybound( ybound )
    ax.set_title(r'${\frac{d{\theta}}{dt}}_{res}$ (cont., $K$) y $\theta$ (cont., $K$)')
    #ax.set_yticks([])
    delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       
    if show :
       plt.show()

    plt.savefig(fig_name,dpi=dpi)
    plt.close()

    return

def plot_thetas_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       fig_name = plot_path + '/theta_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'vx' :
          dist=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          dist=np.tile(dist,(my_data['nz'],1))
          sw=my_data['slice_width']
          z=my_data['z'][:,:,sw,it]
          t=my_data['t'][:,:,sw,it]
          t_pert=my_data['t'][:,:,sw,it]-my_data['t0'][:,:,sw,it]
          theta=my_data['theta'][:,:,sw,it]
          theta_pert=my_data['theta'][:,:,sw,it]-my_data['theta0'][:,:,sw,it]
          thetae=my_data['thetae'][:,:,sw,it]
          thetae_pert=my_data['thetae'][:,:,sw,it]-my_data['thetae0'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]          

       if my_data['slice_type'] == 'vy' :
          dist=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          dist=np.tile(dist,(my_data['nz'],1))
          sw=my_data['slice_width']
          z=my_data['z'][:,sw,:,it]
          t=my_data['t'][:,sw,:,it]
          t_pert=my_data['t'][:,sw,:,it]-my_data['t0'][:,sw,:,it]
          theta=my_data['theta'][:,sw,:,it]
          theta_pert=my_data['theta'][:,sw,:,it]-my_data['theta0'][:,sw,:,it]
          thetae=my_data['thetae'][:,sw,:,it]
          thetae_pert=my_data['thetae'][:,sw,:,it]-my_data['thetae0'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          
       ncols=3
       nrows=1
       fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' and ybound is None  :
          ybound=[0,20000]
 
       #Ploteo theta y su perturbacion
       ax = axs[0]
       clevs1=np.arange(-15,15.5,0.5) * scale_factor
       t_pert[t_pert > np.max(clevs1)]=np.max(clevs1)
       t_pert[t_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( dist , z , t_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(0,320,5)
       p2=ax.contour( dist , z , t , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( dist , z , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.grid()
       ax.set_title(r'${T}^{\prime}$ (sh., $K$) y T (cont., $K$)')   
       #ax.set_xticks([])

       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(theta_pert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo theta y su perturbacion
       ax = axs[1]
       clevs1=np.arange(-15,15.5,0.5) * scale_factor
       theta_pert[theta_pert > np.max(clevs1)]=np.max(clevs1)
       theta_pert[theta_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( dist , z , theta_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( dist , z , theta , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( dist , z , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.grid()
       ax.set_title(r'${\theta}^{\prime}$ (sh., $K$) y $\theta$ (cont., $K$)')   
       #ax.set_xticks([])


       #Ploteo theta y su perturbacion
       ax = axs[2]
       clevs1=np.arange(-15,15.5,0.5) * scale_factor
       thetae_pert[thetae_pert > np.max(clevs1)]=np.max(clevs1)
       thetae_pert[thetae_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( dist , z , thetae_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(270,500,5)
       p2=ax.contour( dist , z , thetae , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( dist , z , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.grid()
       ax.set_title(r'${{\theta}_{e}}^{\prime}$ (sh., $K$) y ${\theta}_{e}$ (cont., $K$)')   
       #ax.set_xticks([])


       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return


def plot_vapor_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :
        
       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it]
          qv_pert=my_data['qv'][:,:,sw,it]-my_data['qv0'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it] 
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          dqvdt_loc =my_data['dqvdt_loc'][:,:,sw,it]
          dqvdt_adv =my_data['dqvdt_adv'][:,:,sw,it]
          qv_diabatic =my_data['qv_diabatic'][:,:,sw,it]                   
          
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          qv=my_data['qv'][:,sw,:,it]
          qv_pert=my_data['qv'][:,sw,:,it]-my_data['qv0'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          dqvdt_loc =my_data['dqvdt_loc'][:,sw,:,it]
          dqvdt_adv =my_data['dqvdt_adv'][:,sw,:,it]
          qv_diabatic =my_data['qv_diabatic'][:,sw,:,it]

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          qv=my_data['qv'][1,:,:,it]
          qv_pert=my_data['qv'][1,:,:,it]-my_data['qv0'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          dqvdt_loc =my_data['dqvdt_loc'][1,:,:,it]
          dqvdt_adv =my_data['dqvdt_adv'][1,:,:,it]
          qv_diabatic =my_data['qv_diabatic'][1,:,:,it]
          
       ncols=2
       nrows=2
       fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
 
       #Ploteo theta y su perturbacion
       ax = axs[0,0]
       clevs1=np.arange(-5.0,5.05,0.05) * scale_factor
       qv_pert = qv_pert * 1000.0
       qv_pert[qv_pert > np.max(clevs1)]=np.max(clevs1)
       qv_pert[qv_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , qv_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y , qv , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${q_v}^{\prime}$ (sh., $gKg^{-1}$) y $q_v$ (cont., $gKg^{-1}$)')   
       #ax.set_xticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(qv_pert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Defino la escala de colores para la tasa de cambio de theta con el tiempo.
       clevs1=np.arange(-0.05,0.055,0.005) * scale_factor

       #Ploteo la tasa de cambio local de theta con el tiempo
       ax = axs[0,1]
       dqvdt_loc = dqvdt_loc 
       dqvdt_loc[dqvdt_loc > np.max(clevs1)]=np.max(clevs1)
       dqvdt_loc[dqvdt_loc < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('bwr',clevs1.size)
       p1=ax.contourf( x , y , dqvdt_loc , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y , qv , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{\partial{q_v}}{{\partial}t}}$ (sh., $gKg^{-1}s^{-1}$) y $q_v$ (cont., $gKg^{-1}$)')
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dqvdt_loc)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[1,0]
       my_map = cmap_discretize('bwr',clevs1.size)
       dqvdt_adv =  dqvdt_adv 
       dqvdt_adv[dqvdt_adv > np.max(clevs1)]=np.max(clevs1)
       dqvdt_adv[dqvdt_adv < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqvdt_adv , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y , qv , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$-V{\nabla}{q_v}$ (sh., $gKg^{-1}$) y $q_v$ (cont., $gKg^{-1}s^{-1}$) y viento (U,W) (vectores)')
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)


       #Ploteo el aporte del calor latente. 
       #TODO reemplazar la estimacion del calor latente por el calor latente posta.
       ax = axs[1,1]
       my_map = cmap_discretize('bwr',clevs1.size)
       qv_diabatic = qv_diabatic 
       qv_diabatic[qv_diabatic > np.max(clevs1)]=np.max(clevs1)
       qv_diabatic[qv_diabatic < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , qv_diabatic , clevs1 , cmap=my_map)
       clevs2=np.array([0.005,0.01,0.05,0.1]) * scale_factor
       p2=ax.contour( x , y , ( dqvdt_loc - dqvdt_adv - qv_diabatic ) , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.array([-0.1,-0.05,-0.01,-0.005]) * scale_factor
       p2=ax.contour( x , y , ( dqvdt_loc - dqvdt_adv - qv_diabatic ) , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$cond.$ (sh., $gKg^{-1}s^{-1}$) y ${\frac{d{q_v}}{dt}}_{res}$ (cont., $gKg^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()

    return

def plot_vapor_equation_2_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'

       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          qv_pert=( my_data['qv'][:,:,sw,it]-my_data['qv0'][:,:,sw,it])  * 1.0e3
          qv=my_data['qv'][:,:,sw,it] * 1.0e3 
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          dqvdt_loc =my_data['dqvdt_loc'][:,:,sw,it] * 1.0e3
          dqvdt_adv =my_data['dqvdt_adv'][:,:,sw,it] * 1.0e3
          qv_diabatic =my_data['qv_diabatic'][:,:,sw,it] * 1.0e3                   
          
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          qv_pert= ( my_data['qv'][:,sw,:,it]-my_data['qv0'][:,sw,:,it] ) * 1.0e3
          qv=my_data['qv'][:,sw,:,it] * 1.0e3
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          dqvdt_loc =my_data['dqvdt_loc'][:,sw,:,it] * 1.0e3
          dqvdt_adv =my_data['dqvdt_adv'][:,sw,:,it] * 1.0e3
          qv_diabatic =my_data['qv_diabatic'][:,sw,:,it] * 1.0e3

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          qv_pert=( my_data['qv'][1,:,:,it]-my_data['qv0'][1,:,:,it] ) * 1.0e3
          qv=my_data['qv'][1,:,:,it] * 1.0e3
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          dqvdt_loc =my_data['dqvdt_loc'][1,:,:,it] * 1.0e3
          dqvdt_adv =my_data['dqvdt_adv'][1,:,:,it] * 1.0e3
          qv_diabatic =my_data['qv_diabatic'][1,:,:,it] * 1.0e3
          
       ncols=2
       nrows=2
       fig, axs = plt.subplots( nrows,ncols , figsize=[10,8] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.04,hspace=0.09,bottom=0.1,left=0.06,right=0.98,top=0.96)
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
 
       #Ploteo theta y su perturbacion
       ax = axs[0,0]
       clevs1=np.arange(-5.0,5.05,0.05) * scale_factor
       qv_pert = qv_pert 
       qv_pert[qv_pert > np.max(clevs1)]=np.max(clevs1)
       qv_pert[qv_pert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y  , qv_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y  , qv  , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y  , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${q_v}^{\prime}$ (sh., $gKg^{-1}$) y $q_v$ (cont., $gKg^{-1}$)')   
       #ax.set_xticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(qv_pert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Defino la escala de colores para la tasa de cambio de theta con el tiempo.
       clevs1=np.arange(-0.05,0.055,0.005) * scale_factor

       #Ploteo la tasa de cambio local de theta con el tiempo
       ax = axs[0,1]
       dqvdt_tot = ( dqvdt_loc - dqvdt_adv ) 
       dqvdt_tot[dqvdt_tot > np.max(clevs1)]=np.max(clevs1)
       dqvdt_tot[dqvdt_tot < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('bwr',clevs1.size)
       p1=ax.contourf( x , y  , dqvdt_tot , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y  , qv  , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y  , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_title(r'${\frac{d{q_v}}{dt}}$ (sh., $gKg^{-1}s^{-1}$), $q_v$ (cont., $gKg^{-1}$) y viento (U,W)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dqvdt_loc)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo el aporte de qv al empuje y qv.
       ax = axs[1,0]
       my_map = cmap_discretize('bwr',clevs1.size)
       qv_diabatic = qv_diabatic 
       qv_diabatic[qv_diabatic > np.max(clevs1)]=np.max(clevs1)
       qv_diabatic[qv_diabatic < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y  , qv_diabatic , clevs1 , cmap=my_map)
       clevs2=np.arange(0,18,1)
       p2=ax.contour( x , y  , qv  , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y  , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$evap \backslash cond.$ (sh., $gKg^{-1}s^{-1}$) y $q_v$ (cont., $gKg^{-1}s^{-1}$)')
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)

       #Ploteo el aporte del cambio de estado 
       ax = axs[1,1]
       my_map = cmap_discretize('bwr',clevs1.size)
       dqvdt_res = ( dqvdt_loc - dqvdt_adv - qv_diabatic ) 
       dqvdt_res[dqvdt_res > np.max(clevs1)]=np.max(clevs1)
       dqvdt_res[dqvdt_res < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y  , dqvdt_res  , clevs1 , cmap=my_map)
       p2=ax.contour( x , y  , qv , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y  , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{d{q_v}}{dt}}_{res}$ (sh., $gKg^{-1}s^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()

    return


def plot_water_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       fig_name = plot_path + '/water_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          qv=my_data['qv'][:,:,sw,it] * 1000.0
          qc=my_data['qc'][:,:,sw,it] * 1000.0
          qr=my_data['qr'][:,:,sw,it] * 1000.0
          qi=my_data['qi'][:,:,sw,it] * 1000.0
          qs=my_data['qs'][:,:,sw,it] * 1000.0
          qg=my_data['qg'][:,:,sw,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][:,:,sw,it] - my_data['dqvdt_adv'][:,:,sw,it] ) * 1000.0
          dqcdt=( my_data['dqcdt_loc'][:,:,sw,it] - my_data['dqcdt_adv'][:,:,sw,it] ) * 1000.0
          dqrdt=( my_data['dqrdt_loc'][:,:,sw,it] - my_data['dqrdt_adv'][:,:,sw,it] ) * 1000.0
          dqidt=( my_data['dqidt_loc'][:,:,sw,it] - my_data['dqidt_adv'][:,:,sw,it] ) * 1000.0
          dqsdt=( my_data['dqsdt_loc'][:,:,sw,it] - my_data['dqsdt_adv'][:,:,sw,it] ) * 1000.0
          dqgdt=( my_data['dqgdt_loc'][:,:,sw,it] - my_data['dqgdt_adv'][:,:,sw,it] ) * 1000.0

       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          qv=my_data['qv'][:,sw,:,it] * 1000.0
          qc=my_data['qc'][:,sw,:,it] * 1000.0
          qr=my_data['qr'][:,sw,:,it] * 1000.0
          qi=my_data['qi'][:,sw,:,it] * 1000.0
          qs=my_data['qs'][:,sw,:,it] * 1000.0
          qg=my_data['qg'][:,sw,:,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][:,sw,:,it] - my_data['dqvdt_adv'][:,sw,:,it] ) * 1000.0
          dqcdt=( my_data['dqcdt_loc'][:,sw,:,it] - my_data['dqcdt_adv'][:,sw,:,it] ) * 1000.0
          dqrdt=( my_data['dqrdt_loc'][:,sw,:,it] - my_data['dqrdt_adv'][:,sw,:,it] ) * 1000.0
          dqidt=( my_data['dqidt_loc'][:,sw,:,it] - my_data['dqidt_adv'][:,sw,:,it] ) * 1000.0
          dqsdt=( my_data['dqsdt_loc'][:,sw,:,it] - my_data['dqsdt_adv'][:,sw,:,it] ) * 1000.0
          dqgdt=( my_data['dqgdt_loc'][:,sw,:,it] - my_data['dqgdt_adv'][:,sw,:,it] ) * 1000.0
          
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          sw=my_data['slice_width']
          ref=my_data['ref'][1,:,:,it]
          qv=my_data['qv'][1,:,:,it] * 1000.0
          qc=my_data['qc'][1,:,:,it] * 1000.0
          qr=my_data['qr'][1,:,:,it] * 1000.0
          qi=my_data['qi'][1,:,:,it] * 1000.0
          qs=my_data['qs'][1,:,:,it] * 1000.0
          qg=my_data['qg'][1,:,:,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][1,:,:,it] - my_data['dqvdt_adv'][1,:,:,it] ) * 1000.0
          dqcdt=( my_data['dqcdt_loc'][1,:,:,it] - my_data['dqcdt_adv'][1,:,:,it] ) * 1000.0
          dqrdt=( my_data['dqrdt_loc'][1,:,:,it] - my_data['dqrdt_adv'][1,:,:,it] ) * 1000.0
          dqidt=( my_data['dqidt_loc'][1,:,:,it] - my_data['dqidt_adv'][1,:,:,it] ) * 1000.0
          dqsdt=( my_data['dqsdt_loc'][1,:,:,it] - my_data['dqsdt_adv'][1,:,:,it] ) * 1000.0
          dqgdt=( my_data['dqgdt_loc'][1,:,:,it] - my_data['dqgdt_adv'][1,:,:,it] ) * 1000.0          
           
       ncols=3
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       #Fijo los niveles para las tasas de cambio local
       clevs1=np.arange(-0.1,0.1+0.01,0.01) * scale_factor
       my_map = cmap_discretize('bwr',clevs1.size)   
       clevs2=np.array([0.5,1.0,2.5,5.0,10.0,15.0,20.0,25.0]) * scale_factor
       #clevs2=np.arange(0,18,0.5)
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #grafico los cambios en q_v y q_v
       ax = axs[0,0]
       my_map = cmap_discretize('bwr',clevs1.size)
       dqvdt[dqvdt > np.max(clevs1)]=np.max(clevs1)
       dqvdt[dqvdt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqvdt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qv , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_v}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_v$ (cont., $gKg^{-1}$)')
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dqvdt)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)
       #ax.set_xticks([])
       ax.grid()

       #Grafico los cambios en q_cloud y q_cloud
       ax = axs[0,1]
       dqcdt[dqcdt > np.max(clevs1)]=np.max(clevs1)
       dqcdt[dqcdt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqcdt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qc , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_c}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_c$ (cont., $gKg^{-1}$)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()

       #Grafico los cambios en q_rain y q_rain
       ax = axs[0,2]
       dqrdt[dqrdt > np.max(clevs1)]=np.max(clevs1)
       dqrdt[dqrdt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqrdt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qr , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_r}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_r$ (cont., $gKg^{-1}$)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()
            
       #Grafico los cambios en q_ice y q_ice
       ax = axs[1,0]
       dqidt[dqidt > np.max(clevs1)]=np.max(clevs1)
       dqidt[dqidt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqidt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qi , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_i}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_i$ (cont., $gKg^{-1}$)')
       ax.grid()
              
       #Grafico los cambios en q_snow y q_snow
       ax = axs[1,1]
       dqsdt[dqsdt > np.max(clevs1)]=np.max(clevs1)
       dqsdt[dqsdt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqsdt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qs , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_s}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_s$ (cont., $gKg^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       
       #Grafico los cambios en q_graup y q_graup
       ax = axs[1,2]
       dqgdt[dqgdt > np.max(clevs1)]=np.max(clevs1)
       dqgdt[dqgdt < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , dqgdt , clevs1 , cmap=my_map )
       p2=ax.contour( x , y , qg , clevs2 , colors='k',linestyles='solid' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${\frac{dq_g}{dt}}$ (sh, $gKg^{-1}s{-1}$) y $q_g$ (cont., $gKg^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return


def plot_water_equation_2_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/vapor_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       fig_name = plot_path + '/water_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          t=my_data['t'][:,:,sw,it]
          qc=my_data['qc'][:,:,sw,it] * 1000.0
          qr=my_data['qr'][:,:,sw,it] * 1000.0
          qi=my_data['qi'][:,:,sw,it] * 1000.0
          qs=my_data['qs'][:,:,sw,it] * 1000.0
          qg=my_data['qg'][:,:,sw,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][:,:,sw,it] - my_data['dqvdt_adv'][:,:,sw,it] ) * 1000.0

       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          t=my_data['t'][:,sw,:,it]
          qc=my_data['qc'][:,sw,:,it] * 1000.0
          qr=my_data['qr'][:,sw,:,it] * 1000.0
          qi=my_data['qi'][:,sw,:,it] * 1000.0
          qs=my_data['qs'][:,sw,:,it] * 1000.0
          qg=my_data['qg'][:,sw,:,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][:,sw,:,it] - my_data['dqvdt_adv'][:,sw,:,it] ) * 1000.0
          
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          sw=my_data['slice_width']
          ref=my_data['ref'][1,:,:,it]
          t=my_data['t'][1,:,:,it]
          qc=my_data['qc'][1,:,:,it] * 1000.0
          qr=my_data['qr'][1,:,:,it] * 1000.0
          qi=my_data['qi'][1,:,:,it] * 1000.0
          qs=my_data['qs'][1,:,:,it] * 1000.0
          qg=my_data['qg'][1,:,:,it] * 1000.0
          dqvdt=( my_data['dqvdt_loc'][1,:,:,it] - my_data['dqvdt_adv'][1,:,:,it] ) * 1000.0
        
           
       ncols=3
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       #Fijo los niveles para las tasas de cambio local
       clevs1=np.arange(0.1,20,0.1) * scale_factor
       my_map = cmap_discretize('YlOrRd',clevs1.size)   
       #clevs2=np.array([0.5,1.0,2.5,5.0,10.0,15.0,20.0,25.0]) * scale_factor
       #clevs2=np.arange(0,18,0.5)
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #grafico los cambios en q_v y q_v
       ax = axs[0,0]
       p1=ax.contourf( x , y , qc+qr+qi+qs+qg , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, fontsize=12)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_{tot}$ (cont., $gKg^{-1}$)')
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dqvdt)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)
       #ax.set_xticks([])
       ax.grid()

       #Grafico los cambios en q_cloud y q_cloud
       ax = axs[0,1]
       p1=ax.contourf( x , y , qc , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, inline=1, fontsize=12 , fmt='%3.0f')
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_c$ (cont., $gKg^{-1}$)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()

       #Grafico los cambios en q_rain y q_rain
       ax = axs[0,2]
       p1=ax.contourf( x , y , qr , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, inline=1, fontsize=12 , fmt='%3.0f')
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_r$ (cont., $gKg^{-1}$)')
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()
            
       #Grafico los cambios en q_ice y q_ice
       ax = axs[1,0]
       p1=ax.contourf( x , y , qi , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, inline=1, fontsize=12, fmt='%3.0f')
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_i$ (cont., $gKg^{-1}$)')
       ax.grid()
              
       #Grafico los cambios en q_snow y q_snow
       ax = axs[1,1]
       p1=ax.contourf( x , y , qs , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, inline=1, fontsize=12, fmt='%3.0f')
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_s$ (cont., $gKg^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       
       #Grafico los cambios en q_graup y q_graup
       ax = axs[1,2]
       p1=ax.contourf( x , y , qg , clevs1 , cmap=my_map )
       p3=ax.contour( x , y , t , [-40.0,-20.0,0.0] , colors='k',linestyles='dashed' , linewidths=2.0 )
       ax.clabel(p3, inline=1, fontsize=12, fmt='%3.0f')
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$q_g$ (cont., $gKg^{-1}$)')
       #ax.set_yticks([])
       ax.grid()
       
       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()

    return


def plot_ppert_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/ppert_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/ppert_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          bouy=my_data['bouy'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['u'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          lp_ppert=my_data['lp_ppert'][:,:,sw,it]
          lp_spin_tot=my_data['lp_spin_tot'][:,:,sw,it]
          lp_splat_tot=my_data['lp_splat_tot'][:,:,sw,it]
          lp_spin_pert =my_data['lp_spin_pert'][:,:,sw,it]
          lp_splat_pert =my_data['lp_splat_pert'][:,:,sw,it]
          lp_lineal =my_data['lp_lineal'][:,:,sw,it]
          lp_b =my_data['lp_b'][:,:,sw,it]
          ppert = my_data['p'][:,:,sw,it] - my_data['p0'][:,:,sw,it]
          ppert_hy = my_data['ppert_hy'][:,:,sw,it]
          ppert_nhy= my_data['ppert_nhy'][:,:,sw,it]
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          bouy=my_data['bouy'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['v'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          lp_ppert=my_data['lp_ppert'][:,sw,:,it]
          lp_spin_tot=my_data['lp_spin_tot'][:,sw,:,it]
          lp_splat_tot=my_data['lp_splat_tot'][:,sw,:,it]
          lp_spin_pert =my_data['lp_spin_pert'][:,sw,:,it]
          lp_splat_pert =my_data['lp_splat_pert'][:,sw,:,it]
          lp_lineal =my_data['lp_lineal'][:,sw,:,it]
          lp_b =my_data['lp_b'][:,sw,:,it]
          ppert = my_data['p'][:,sw,:,it] - my_data['p0'][:,sw,:,it]
          ppert_hy = my_data['ppert_hy'][:,sw,:,it]
          ppert_nhy= my_data['ppert_nhy'][:,sw,:,it]
       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          sw=my_data['slice_width']
          bouy=my_data['bouy'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          lp_ppert=my_data['lp_ppert'][1,:,:,it]
          lp_spin_tot=my_data['lp_spin_tot'][1,:,:,it]
          lp_splat_tot=my_data['lp_splat_tot'][1,:,:,it]
          lp_spin_pert =my_data['lp_spin_pert'][1,:,:,it]
          lp_splat_pert =my_data['lp_splat_pert'][1,:,:,it]
          lp_lineal =my_data['lp_lineal'][1,:,:,it]
          lp_b =my_data['lp_b'][1,:,:,it]
          ppert = my_data['p'][1,:,:,it] - my_data['p0'][1,:,:,it]
          ppert_hy = my_data['ppert_hy'][1,:,:,it]
          ppert_nhy= my_data['ppert_nhy'][1,:,:,it]          
      
       ncols=4
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #Ploteo la perturbacion de presion y la fuerza de presion
       ax = axs[0,0]
       clevs1=np.arange(-600,650,50) * scale_factor
       ppert[ppert > np.max(clevs1)]=np.max(clevs1)
       ppert[ppert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , ppert , clevs1 , cmap=my_map)
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${P}^{\prime}$ (sh., $Pa$) y $-{\frac{1}{{\rho}_0}}{\nabla}{P}^{\prime}$ (vectores)',fontsize=10)

       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(ppert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo la perturbacion de presion no hidrostatica
       ax = axs[0,1]
       ppert_hy[ppert_hy > np.max(clevs1)]=np.max(clevs1)
       ppert_hy[ppert_hy < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , ppert_hy , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${P}^{\prime}_{hydro}$ (sh., $Pa$) y viento (vectores)',fontsize=10)

       ax.grid()

       #Ploteo la perturbacion de presion hidrostatica
       ax = axs[0,2]
       ppert_nhy[ppert_nhy > np.max(clevs1)]=np.max(clevs1)
       ppert_nhy[ppert_nhy < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , ppert_nhy , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'${P}^{\prime}_{nohydro}$ (sh., $Pa$) y viento (vectores) ',fontsize=10)

       ax.grid()

       #Fijo los niveles para las componentes del empuje.
       clevs1=np.arange(-8.0,8.0+0.05,0.05) * scale_factor
       my_map = cmap_discretize('bwr',clevs1.size)

       #Ploteo laplaciano de la perturbacion de presion por empuje y empuje.
       ax = axs[0,3]
       lp_b = lp_b * 1.0e4
       lp_b[lp_b > np.max(clevs1)]=np.max(clevs1) 
       lp_b[lp_b < np.min(clevs1)]=np.min(clevs1) 
       p1=ax.contourf( x , y , -lp_b , clevs1 , cmap=my_map)
       clevs2=np.arange(0.1,1.0,0.1)*10.0e-1 * scale_factor
       p2=ax.contour( x , y , bouy , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-1.0,0.0,0.1)*10.0e-1 * scale_factor
       p2=ax.contour( x , y , bouy , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_b$ (sh., $10^{-4} ms^{-2}$) y B (cont., $10^{-1}ms^{-2}$)',fontsize=10)
       ax.grid()

       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(-lp_b)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo laplaciano de la perturbacion de presion dinamica 
       ax = axs[1,0]
       lp_dyn_tot = ( lp_spin_tot + lp_splat_tot ) * 1.0e4
       lp_dyn_tot[lp_dyn_tot > np.max(clevs1)]=np.max(clevs1)
       lp_dyn_tot[lp_dyn_tot < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_dyn_tot , clevs1 , cmap=my_map)
       clevs2=np.arange(1.0,11,1.0) * scale_factor
       p2=ax.contour( x , y , (-lp_spin_pert -lp_splat_pert) * 1.0e4 , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-10.0,0.0,1.0) * scale_factor
       p2=ax.contour( x , y , (-lp_spin_pert -lp_splat_pert) * 1.0e4 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{dyn}$ pert (sh. $10^{-4}ms^{-2}$) y tot (cont) ($ms^{-2}$)',fontsize=10)

       
       #Ploteo laplaciano de la perturbacion por spin total y asociada al viento perturbado
       ax = axs[1,1]
       lp_spin_pert = ( lp_spin_pert ) * 1.0e4
       lp_spin_pert[lp_spin_pert > np.max(clevs1)]=np.max(clevs1)
       lp_spin_pert[lp_spin_pert < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_spin_pert , clevs1 , cmap=my_map)
       clevs2=np.arange(1.0,11,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_spin_tot * 1.0e4 , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-10.0,0.0,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_spin_tot * 1.0e4 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{spin}$ pert (sh.) y tot (cont) ($10^{-4}ms^{-2}$)',fontsize=10)
       
       #Ploteo laplaciano de la perturbacion por splat total y asociada al viento perturbado
       ax = axs[1,2]
       lp_splat_pert = ( lp_splat_pert ) * 1.0e4
       lp_splat_pert[lp_splat_pert > np.max(clevs1)]=np.max(clevs1)
       lp_splat_pert[lp_splat_pert < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_splat_pert , clevs1 , cmap=my_map) 
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       clevs2=np.arange(1.0,11,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_splat_tot * 1.0e4 , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-10.0,0.0,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_splat_tot * 1.0e4 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{splat}$ pert (sh.) y tot (cont) ($10^{-4}ms^{-2}$)',fontsize=10)

       #Ploteo el termino lineal.
       ax = axs[1,3]
       lp_lineal = ( lp_lineal ) * 1.0e4
       lp_lineal[lp_lineal > np.max(clevs1)]=np.max(clevs1)
       lp_lineal[lp_lineal < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_lineal , clevs1 , cmap=my_map)
       clevs2=np.arange(-7.0,0.0,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_ppert * 1.0e4 , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       clevs2=np.arange(1.0,8.0,1.0) * scale_factor
       p2=ax.contour( x , y , -lp_ppert * 1.0e4 , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{lineal}$ (sh. $10^{-4}ms^{-2}$) y $-{\nabla}^{2}{P}^{\prime}$ (cont. $10^{-4}ms^{-2}$)',fontsize=10)

       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return

def plot_ppert_equation_2_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/ppert_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/ppert_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          bouy=my_data['bouy'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          hwind=my_data['u'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]
          lp_ppert=my_data['lp_ppert'][:,:,sw,it]
          lp_spin_tot=my_data['lp_spin_tot'][:,:,sw,it]
          lp_splat_tot=my_data['lp_splat_tot'][:,:,sw,it]
          lp_b =my_data['lp_b'][:,:,sw,it]
          ppert = my_data['p'][:,:,sw,it] - my_data['p0'][:,:,sw,it]

       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          bouy=my_data['bouy'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          hwind=my_data['v'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          lp_ppert=my_data['lp_ppert'][:,sw,:,it]
          lp_spin_tot=my_data['lp_spin_tot'][:,sw,:,it]
          lp_splat_tot=my_data['lp_splat_tot'][:,sw,:,it]
          lp_b =my_data['lp_b'][:,sw,:,it]
          ppert = my_data['p'][:,sw,:,it] - my_data['p0'][:,sw,:,it]

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          sw=my_data['slice_width']
          bouy=my_data['bouy'][1,:,:,it]
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]
          lp_ppert=my_data['lp_ppert'][1,:,:,it]
          lp_spin_tot=my_data['lp_spin_tot'][1,:,:,it]
          lp_splat_tot=my_data['lp_splat_tot'][1,:,:,it]
          lp_b =my_data['lp_b'][1,:,:,it]
          ppert = my_data['p'][1,:,:,it] - my_data['p0'][1,:,:,it]
    
      
       ncols=3
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #Ploteo la perturbacion de presion y la fuerza de presion
       ax = axs[0,0]
       clevs1=np.arange(-600,650,50) * scale_factor
       ppert[ppert > np.max(clevs1)]=np.max(clevs1)
       ppert[ppert < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , ppert , clevs1 , cmap=my_map)
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_title(r'${P}^{\prime}$ (sh., $Pa$) y $-{\frac{1}{{\rho}_0}}{\nabla}{P}^{\prime}$ (vectores)',fontsize=10)
       #ax.set_xticks([])
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(ppert)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #
       clevs1=np.arange(-8.0,8.0+0.05,0.05) * scale_factor
       my_map = cmap_discretize('bwr',clevs1.size)

       #Ploteo laplaciano de la perturbacion de presion por empuje y empuje.
       ax = axs[0,1]
       lp_b = lp_b * 1.0e4
       lp_b[lp_b > np.max(clevs1)]=np.max(clevs1) 
       lp_b[lp_b < np.min(clevs1)]=np.min(clevs1) 
       p1=ax.contourf( x , y , -lp_b , clevs1 , cmap=my_map)
       clevs2=np.arange(0.1,1.0,0.1)*10.0e-1 * scale_factor
       p2=ax.contour( x , y , bouy , clevs2 , colors='k',linestyles='solid' , linewidths=0.5 )
       clevs2=np.arange(-1.0,0.0,0.1)*10.0e-1 * scale_factor
       p2=ax.contour( x , y , bouy , clevs2 , colors='k',linestyles='dashed' , linewidths=0.5 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_b$ (sh., $10^{-4} ms^{-2}$) y B (cont., $10^{-1}ms^{-2}$)',fontsize=10)
       #ax.set_xticks([])
       #ax.set_yticks([])
       ax.grid()

       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(-lp_b)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo laplaciano de la perturbacion de presion dinamica 
       ax = axs[0,2]
       lp_dyn_tot = ( lp_spin_tot + lp_splat_tot ) * 1.0e4
       lp_dyn_tot[lp_dyn_tot > np.max(clevs1)]=np.max(clevs1)
       lp_dyn_tot[lp_dyn_tot < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_dyn_tot , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{dyn}$ pert (sh. $10^{-4}ms^{-2}$) y viento (U,W)',fontsize=10)

       
       #Ploteo laplaciano de la perturbacion por spin total y asociada al viento perturbado
       ax = axs[1,0]
       lp_spin_tot = ( lp_spin_tot  ) * 1.0e4
       lp_spin_tot[lp_spin_tot > np.max(clevs1)]=np.max(clevs1)
       lp_spin_tot[lp_spin_tot < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_spin_tot , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{spin}$ pert (sh.) y viento (U,W)',fontsize=10)
       #ax.set_yticks([])
       
       #Ploteo laplaciano de la perturbacion por splat total y asociada al viento perturbado
       ax = axs[1,1]
       lp_splat_tot = ( lp_splat_tot  ) * 1.0e4
       lp_splat_tot[lp_splat_tot > np.max(clevs1)]=np.max(clevs1)
       lp_splat_tot[lp_splat_tot < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_splat_tot , clevs1 , cmap=my_map) 
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=400.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=100.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}_{splat}$ pert (sh.) y viento (U,W)',fontsize=10)
       #ax.set_yticks([])

       #Perturbacion de P y su laplaciano
       ax = axs[1,2]
       lp_ppert = ( lp_ppert  ) * 1.0e4
       lp_ppert[lp_ppert > np.max(clevs1)]=np.max(clevs1)
       lp_ppert[lp_ppert < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , -lp_ppert , clevs1 , cmap=my_map)
       clevs2=np.arange(0,650,50) * scale_factor
       p1=ax.contour( x , y , ppert , clevs2 , colors='k' , linestyles='solid' , linewidths=2.0 )
       clevs2=np.arange(-600,0,50) * scale_factor
       p1=ax.contour( x , y , ppert , clevs2 , colors='k' , linestyles='dashed' , linewidths=2.0 )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\nabla}^{2}{P}^{\prime}$ (sh. $10^{-4}ms^{-2}$) y $P \prime$ (cont. Pa)',fontsize=10)
       #ax.set_yticks([])

       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return

def plot_vortz_equation_v( my_data , plot_path , show=False , force = False , scale_factor = 1.0 , arrow_scale_factor = 1.0 , ybound = None , xbound = None ) :
    #Ploteo un cross section vertical de la ecuacion de movimiento (una figura para w y otra para u y v)

    for it in range( my_data['nt'] ) :

       if my_data['slice_type'] == 'vx' or my_data['slice_type'] == 'vy' :
          fig_name = plot_path + '/vortz_equation_' + my_data['slice_type'] + '_i_' + str(my_data['slice_index']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
       if my_data['slice_type'] == 'h' :
          fig_name = plot_path + '/vortz_equation_' + my_data['slice_type'] + '_z_' + str(my_data['slice_z']) + '_t' + '{:02d}'.format(it + my_data['ts']) + '.png'
        
       if my_data['slice_type'] == 'vx' :
          x=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,:,sw,it]
          vortz=my_data['vortz'][:,:,sw,it]
          vortx=my_data['vortx'][:,:,sw,it]
          vorty=my_data['vorty'][:,:,sw,it]
          tilt_vortz=my_data['tilt_vortz'][:,:,sw,it]
          str_vortz=my_data['str_vortz'][:,:,sw,it]
          hwind=my_data['v'][:,:,sw,it]
          hwind0=my_data['u0'][:,:,sw,it]
          w=my_data['w'][:,:,sw,it]
          ref=my_data['ref'][:,:,sw,it]          
          dvortzdt_loc=my_data['dvortzdt_loc'][:,:,sw,it] 
          advh_vortz=my_data['advh_vortz'][:,:,sw,it] 
          advv_vortz=my_data['advz_vortz'][:,:,sw,it]
          
       if my_data['slice_type'] == 'vy' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          x=np.tile(x,(my_data['nz'],1))
          sw=my_data['slice_width']
          y=my_data['z'][:,sw,:,it]
          vortz=my_data['vortz'][:,sw,:,it]
          vortx=my_data['vortx'][:,sw,:,it]
          vorty=my_data['vorty'][:,sw,:,it]
          tilt_vortz=my_data['tilt_vortz'][:,sw,:,it]
          str_vortz=my_data['str_vortz'][:,sw,:,it]
          hwind=my_data['u'][:,sw,:,it]
          hwind0=my_data['v0'][:,sw,:,it]
          w=my_data['w'][:,sw,:,it]
          ref=my_data['ref'][:,sw,:,it]
          dvortzdt_loc =my_data['dvortzdt_loc'][:,sw,:,it]
          advh_vortz=my_data['advh_vortz'][:,sw,:,it] 
          advv_vortz=my_data['advz_vortz'][:,sw,:,it]

       if my_data['slice_type'] == 'h' :
          x=np.arange(0,my_data['dx']*my_data['nx'],my_data['dx'])
          y=np.arange(0,my_data['dy']*my_data['ny'],my_data['dy'])
          sw=my_data['slice_width']
          w=my_data['w'][1,:,:,it]
          uwind=my_data['u'][1,:,:,it]
          vwind=my_data['v'][1,:,:,it]
          uwind0=my_data['u0'][1,:,:,it]
          vwind0=my_data['v0'][1,:,:,it]
          vortz=my_data['vortz'][1,:,:,it]
          vortx=my_data['vortx'][1,:,:,it]
          vorty=my_data['vorty'][1,:,:,it]
          tilt_vortz=my_data['tilt_vortz'][1,:,:,it]
          str_vortz=my_data['str_vortz'][1,:,:,it]
          ref=my_data['ref'][1,:,:,it]       
          dvortzdt_loc =my_data['dvortzdt_loc'][1,:,:,it]
          advh_vortz=my_data['advh_vortz'][1,:,:,it] 
          advv_vortz=my_data['advz_vortz'][1,:,:,it] 
    
       ncols=3
       nrows=2
       if ybound is None :
          ybound=[0,np.nanmax(y)]
       if xbound is None :
          xbound=[0,np.nanmax(x)]
          
       fig, axs = plt.subplots( nrows,ncols , figsize=[15,9] , sharex=True , sharey=True )
       fig.subplots_adjust(wspace=0.15,hspace=0.1,bottom=0.095,left=0.045,right=0.98,top=0.96)

       #Grafico la vorticidad de eje vertical y la reflectividad.
       ax = axs[0,0]
       clevs1=np.arange(-30.0e-3,33e-3,3.0e-3) * scale_factor
       vortz[vortz > np.max(clevs1)]=np.max(clevs1)
       vortz[vortz < np.min(clevs1)]=np.min(clevs1)
       my_map = cmap_discretize('RdBu_r',clevs1.size)
       p1=ax.contourf( x , y , vortz , clevs1 , cmap=my_map)
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx],vwind[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_title(r'${\omega}_{z}$ (sh., ${s}^{-1}$) y viento (vectores)',fontsize=10)
       ax.grid()
       delta= ( np.max(clevs1)-np.min(clevs1) ) / (clevs1.size-1)
       cbar_ax = fig.add_axes([0.06, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(vortz)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #
       clevs1=np.arange(-50.0e-5,55.0e-5,5.0e-5) * scale_factor
       my_map = cmap_discretize('bwr',clevs1.size)

       #Grafico la tendencia local de la vorticidad
       ax = axs[0,1]
       dvortzdt_loc[dvortzdt_loc > np.max(clevs1)]=np.max(clevs1) 
       dvortzdt_loc[dvortzdt_loc < np.min(clevs1)]=np.min(clevs1) 
       p1=ax.contourf( x , y , dvortzdt_loc , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.set_title(r'$\frac{\partial{\omega_z}}{{\partial}t}$ (sh., ${s}^{-2}$)',fontsize=10)
       ax.grid()

       cbar_ax = fig.add_axes([0.55, 0.03, 0.4, 0.02])
       m = plt.cm.ScalarMappable(cmap=my_map )
       m.set_array(dvortzdt_loc)
       m.set_clim(np.min(clevs1),np.max(clevs1))
       delta= ( np.max(clevs1)-np.min(clevs1) )/ (clevs1.size-1)
       cb=plt.colorbar(m,cax=cbar_ax,boundaries=np.arange(np.min(clevs1),np.max(clevs1)+delta,delta),orientation='horizontal')
       cb.ax.tick_params(labelsize=labelsize_colorbar)

       #Ploteo la adveccion de vorticidad 
       ax = axs[0,2]
       advh_vortz[advh_vortz > np.max(clevs1)]=np.max(clevs1)
       advh_vortz[advh_vortz < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , advh_vortz , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx]-hwind0[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx]-uwind0[0::skipz,0::skipx],vwind[0::skipz,0::skipx]-vwind0[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{V}_{h} {\nabla}_{h} {\omega}_z $ (sh., ${s}^{-2}$)',fontsize=10)
  
       #Ploteo el termino de tilting
       ax = axs[1,0]
       tilt_vortz[tilt_vortz > np.max(clevs1)]=np.max(clevs1)
       tilt_vortz[tilt_vortz < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , tilt_vortz , clevs1 , cmap=my_map)
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       clevs2=np.arange(10,100,10) * scale_factor
       p1=ax.contour( x , y , w , clevs2 , colors='k' , linestyles='solid' , linewidths=2.0 )
       clevs2=np.arange(-20,0,5) * scale_factor
       p1=ax.contour( x , y , w , clevs2 , colors='k' , linestyles='dashed' , linewidths=2.0 )
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],vortx[0::skipz,0::skipx],vorty[0::skipz,0::skipx],scale=0.5*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r' ',fontsize=10)
       ax.set_title(r'${\omega}_z .{\nabla}w $ (sh., ${s}^{-2}$), W (cont. $m{s}^{-2}$), ${\omega}_z$ (vectores)',fontsize=10)

       
       #Ploteo el termino de stretching
       ax = axs[1,1]
       str_vortz[str_vortz > np.max(clevs1)]=np.max(clevs1)
       str_vortz[str_vortz < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , str_vortz , clevs1 , cmap=my_map) 
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       clevs2=np.arange(2.0e-2,12.0e-2,2.0e-2) * scale_factor
       p1=ax.contour( x , y , vortz , clevs2 , colors='k' , linestyles='solid' , linewidths=2.0 )
       clevs2=np.arange(-10.0e-2,0,2.0e-2) * scale_factor
       p1=ax.contour( x , y , vortz , clevs2 , colors='k' , linestyles='dashed' , linewidths=2.0 )       
       
       skipx=3
       skipz=3
       if my_data['slice_type'] == 'vy' or my_data['slice_type'] == 'vx' :
          ax.quiver(x[0::skipz,0::skipx],y[0::skipz,0::skipx],hwind[0::skipz,0::skipx]-hwind0[0::skipz,0::skipx],w[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       if my_data['slice_type'] == 'h'  :
          ax.quiver(x[0::skipz],y[0::skipz],uwind[0::skipz,0::skipx]-uwind0[0::skipz,0::skipx],vwind[0::skipz,0::skipx]-vwind0[0::skipz,0::skipx],scale=800.0*arrow_scale_factor)
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-{\omega}_z {\nabla}_h .V $ (sh., ${s}^{-2}$), ${\omega}_z$ (cont. ${1}^{-1}$), V\' (vectores)',fontsize=10)

       #ax.set_yticks([])

       #Ploteo la adveccion vertical y el residuo.
       ax = axs[1,2]
       advv_vortz[advv_vortz > np.max(clevs1)]=np.max(clevs1)
       advv_vortz[advv_vortz < np.min(clevs1)]=np.min(clevs1)
       p1=ax.contourf( x , y , advv_vortz , clevs1 , cmap=my_map)
       #res_vortz = dvortzdt_loc - advh_vortz - advv_vortz - tilt_vortz - str_vortz 
       #clevs2=np.arange(0,55.0e-5,5.0e-5) * scale_factor
       #p1=ax.contour( x , y , res_vortz , clevs2, colors='k' , linestyles='solid' )
       #clevs2=np.arange(-50.0e-5,0,5.0e-5) * scale_factor
       #p1=ax.contour( x , y , res_vortz , clevs2, colors='k' , linestyles='dashed' )
       p3=ax.contour( x , y , ref , [30.0,60.0] , colors='c',linestyles='solid' , linewidths=2.5 )
       ax.set_ybound( ybound )
       ax.set_xbound( xbound )
       ax.grid()
       ax.set_title(r'$-w \frac{{\partial \omega_z}}{\partial z}$ (sh. ${s}^{-2}$) $\frac{d{\omega_z}}{dt}_{res}$ (cont. ${s}^{-2}$)',fontsize=10)
       #ax.set_yticks([])

       if show :
          plt.show()

       plt.savefig(fig_name,dpi=dpi)
       plt.close()


    return






#Modified from http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations

def plot_sounding( my_data , plot_path , show=False )  :

    fig_name = plot_path + '/profile' + '_i_' + str(my_data['xp']) + '_j_' + str(my_data['yp']) + '_t' + str(my_data['tp']) + '.png'
    #Ejemplo tomado de https://unidata.github.io/MetPy/latest/examples/plots/Skew-T_Layout.html#sphx-glr-examples-plots-skew-t-layout-py
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(18, 12))
    #add_metpy_logo(fig, 630, 80, size='large')

    # Grid for plots
    #rotation controla la inclinacion de las isotermas. Si es 0 el grafico se asemeja a un emagrama pero 
    #si es 45 entonces se obtiene un skew-T.
    skew = SkewT(fig, rotation=45, rect=(0.05,0.05,0.50,0.90))
    #skew.ax.set_adjustable('datalim')
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-20, 30)

    # Set some better labels than the default to increase readability
    skew.ax.set_xlabel(str.upper(f'Temperature (°C)'), weight='bold')
    skew.ax.set_ylabel(str.upper(f'Pressure (hPa)'), weight='bold')
    
    p=my_data['p'].data * units.hPa
    t=my_data['t'].data * units.degC
    z=my_data['z'] * units.meter
    td=my_data['td'] * units.degC
    u=my_data['u'] * units.meter / units.second
    v=my_data['v'] * units.meter / units.second

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p , t , 'r')
    skew.plot(p , td , 'g')
    idx100=np.argmin(np.abs(np.array(p) - 100))
    skew.plot_barbs(p[0:idx100],u[0:idx100],v[0:idx100])

    # Add the relevant special lines
    skew.plot_dry_adiabats(linewidths=0.5)
    skew.plot_moist_adiabats(linewidths=0.5)
    skew.plot_mixing_lines(linewidths=0.5)

    # Calculate LCL height and plot as black dot. Because `p`'s first value is
    # ~1000 mb and its last value is ~250 mb, the `0` index is selected for
    # `p`, `T`, and `Td` to lift the parcel from the surface. If `p` was inverted,
    # i.e. start from low value, 250 mb, to a high value, 1000 mb, the `-1` index
    # should be selected.
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0] , t[0] , td[0]  )
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p , t[0] , td[0] ).to('degC')
    skew.plot(p, prof, 'k', linewidth=2)

    # Shade areas of CAPE and CIN
    skew.shade_cin(p,t,prof,alpha=0.2)
    skew.shade_cape(p,t,prof,alpha=0.2)
    # Good bounds for aspect ratio

    # Create a hodograph
    max_speed = round(np.nanmax([np.nanmax(abs(my_data['u'])),np.nanmax(abs(my_data['u']))])/10)*10
    ax = fig.add_axes([0.48,0.45,0.5,0.5])
    h = Hodograph(ax, component_range=max_speed)
    h.add_grid(increment=20, ls='-', lw=1.5, alpha=0.5)
    h.add_grid(increment=10, ls='--', lw=1, alpha=0.2)

    h.ax.set_box_aspect(1)
    h.ax.set_yticklabels([])
    h.ax.set_xticklabels([])
    h.ax.set_xticks([])
    h.ax.set_yticks([])
    h.ax.set_xlabel(' ')
    h.ax.set_ylabel(' ')
    plt.xticks(np.arange(0, 0, 1))
    plt.yticks(np.arange(0, 0, 1))
    for i in range(10, 120, 10):
       h.ax.annotate(str(i), (i, 0), xytext=(0, 2), textcoords='offset pixels',
                     clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)
    for i in range(10, 120, 10):
       h.ax.annotate(str(i), (0, i), xytext=(0, 2), textcoords='offset pixels',
                     clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)
    h.plot(my_data['u'],my_data['v'])


    # STEP 4: ADD A FEW EXTRA ELEMENTS TO REALLY MAKE A NEAT PLOT
    # First we want to actually add values of data to the plot for easy viewing
    # To do this, let's first add a simple rectangle using Matplotlib's 'patches'
    # functionality to add some simple layout for plotting calculated parameters
    #                                  xloc   yloc   xsize  ysize
    fig.patches.extend([plt.Rectangle((0.563, 0.05), 0.334, 0.37,
                                      edgecolor='black', facecolor='white',
                                      linewidth=1, alpha=1, transform=fig.transFigure,
                                      figure=fig)])

    # Now let's take a moment to calculate some simple severe-weather parameters using
    # metpy's calculations
    # Here are some classic severe parameters!
    kindex = mpcalc.k_index(p, t, td)
    total_totals = mpcalc.total_totals_index(p, t, td)

    # mixed layer parcel properties!
    ml_t, ml_td = mpcalc.mixed_layer(p, t, td, depth=50 * units.hPa)
    ml_p, _, _ = mpcalc.mixed_parcel(p, t, td, depth=50 * units.hPa)
    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, t, prof, depth=50 * units.hPa)

    # most unstable parcel properties!
    mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, t, td, depth=50 * units.hPa)
    mucape, mucin = mpcalc.most_unstable_cape_cin(p, t, td, depth=50 * units.hPa)

    # Estimate height of LCL in meters from hydrostatic thickness (for sig_tor)
    new_p = np.append(p[p > lcl_pressure], lcl_pressure)
    new_t = np.append(t[p > lcl_pressure], lcl_temperature)
    lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

    # Compute Surface-based CAPE
    sbcape, sbcin = mpcalc.surface_based_cape_cin(p, t, td)
    # Compute SRH
    (u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
    *_, total_helicity1 = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km,
                                                         storm_u=u_storm, storm_v=v_storm)
    *_, total_helicity3 = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km,
                                                         storm_u=u_storm, storm_v=v_storm)
    *_, total_helicity6 = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km,
                                                         storm_u=u_storm, storm_v=v_storm)

    # Copmute Bulk Shear components and then magnitude
    ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
    bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
    ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
    bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
    ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
    bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

    # Use all computed pieces to calculate the Significant Tornado parameter
    sig_tor = mpcalc.significant_tornado(sbcape, lcl_height,
                                         total_helicity3, bshear3).to_base_units()

    # Perform the calculation of supercell composite if an effective layer exists
    super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

    # There is a lot we can do with this data operationally, so let's plot some of
    # these values right on the plot, in the box we made
    # First lets plot some thermodynamic parameters
    plt.figtext(0.58, 0.37, 'SBCAPE: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.37, f'{sbcape:.0f~P}', weight='bold',
                fontsize=15, color='orangered', ha='right')
    plt.figtext(0.58, 0.34, 'SBCIN: ', weight='bold',
                fontsize=15, color='black', ha='left')
    plt.figtext(0.71, 0.34, f'{sbcin:.0f~P}', weight='bold',
                fontsize=15, color='lightblue', ha='right')
    plt.figtext(0.58, 0.29, 'MLCAPE: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.29, f'{mlcape:.0f~P}', weight='bold',
                fontsize=15, color='orangered', ha='right')
    plt.figtext(0.58, 0.26, 'MLCIN: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.26, f'{mlcin:.0f~P}', weight='bold',
                fontsize=15, color='lightblue', ha='right')
    plt.figtext(0.58, 0.21, 'MUCAPE: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.21, f'{mucape:.0f~P}', weight='bold',
                fontsize=15, color='orangered', ha='right')
    plt.figtext(0.58, 0.18, 'MUCIN: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.18, f'{mucin:.0f~P}', weight='bold',
                fontsize=15, color='lightblue', ha='right')
    plt.figtext(0.58, 0.13, 'TT-INDEX: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.13, f'{total_totals:.0f~P}', weight='bold',
                fontsize=15, color='orangered', ha='right')
    plt.figtext(0.58, 0.10, 'K-INDEX: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.71, 0.10, f'{kindex:.0f~P}', weight='bold',
                fontsize=15, color='orangered', ha='right')

    # now some kinematic parameters
    plt.figtext(0.73, 0.37, '0-1km SRH: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.37, f'{total_helicity1:.0f~P}',
                weight='bold', fontsize=15, color='navy', ha='right')
    plt.figtext(0.73, 0.34, '0-1km SHEAR: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.34, f'{bshear1:.0f~P}', weight='bold',
                fontsize=15, color='blue', ha='right')
    plt.figtext(0.73, 0.29, '0-3km SRH: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.29, f'{total_helicity3:.0f~P}',
                weight='bold', fontsize=15, color='navy', ha='right')
    plt.figtext(0.73, 0.26, '0-3km SHEAR: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.26, f'{bshear3:.0f~P}', weight='bold',
                fontsize=15, color='blue', ha='right')
   
    plt.figtext(0.73, 0.21, '0-6km SRH: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.21, f'{total_helicity6:.0f~P}',
                weight='bold', fontsize=15, color='navy', ha='right')
    plt.figtext(0.73, 0.18, '0-6km SHEAR: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.18, f'{bshear6:.0f~P}', weight='bold',
                fontsize=15, color='blue', ha='right')
    plt.figtext(0.73, 0.13, 'SIG TORNADO: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.13, f'{sig_tor[0]:.0f~P}', weight='bold', fontsize=15,
                color='orangered', ha='right')
    plt.figtext(0.73, 0.10, 'SUPERCELL COMP: ', weight='bold', fontsize=15,
                color='black', ha='left')
    plt.figtext(0.88, 0.10, f'{super_comp[0]:.0f~P}', weight='bold', fontsize=15,
                color='orangered', ha='right')
    if show :
       plt.show()

    plt.savefig(fig_name,dpi=dpi)
    plt.close()

def plot_pz( my_data , plot_path , show=False )  :
    fig_name = plot_path + '/pz' + '_i_' + str(my_data['xp']) + '_j_' + str(my_data['yp']) + '_t' + str(my_data['tp']) + '.png'
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(9, 7))
    p=my_data['p']
    z=my_data['z']
    plt.figure()
    plt.plot( z , p )
    plt.xlabel('Height (m)')
    plt.ylabel('Pressure (hPa)')

    if show :
       plt.show()

    plt.savefig(fig_name,dpi=dpi)
    plt.close()

def plot_profile_qv( my_data , plot_path , show=False )  :
    fig_name = plot_path + '/qv_profile' + '_i_' + str(my_data['xp']) + '_j_' + str(my_data['yp']) + '_t' + str(my_data['tp']) + '.png'
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(9, 7))
    p=my_data['p']
    qv=my_data['qv']
    plt.figure()
    plt.plot( qv*1.0e3 , p )
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xlabel('Specific humidity (g/kg)')
    plt.ylabel('Pressure (hPa)')

    if show :
       plt.show()

    plt.savefig(fig_name,dpi=dpi)
    plt.close()


def plot_profile_theta( my_data , plot_path , show=False )  :
    fig_name = plot_path + '/theta_profile' + '_i_' + str(my_data['xp']) + '_j_' + str(my_data['yp']) + '_t' + str(my_data['tp']) + '.png'
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(9, 7))
    p=my_data['p']
    theta=my_data['theta']
    thetae=my_data['thetae']
    plt.figure()
    plt.plot( theta , p ,'bo-' )
    plt.plot( thetae , p , 'ro-' )
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xlabel('Theta (blue, K) and Theta-e (red, K)')
    plt.ylabel('Pressure (hPa)')

    if show :
       plt.show()

    plt.savefig(fig_name,dpi=dpi)
    plt.close()

def profile_to_csv( my_data , data_path )  :
    import csv
    file_name = data_path + '/text_profile' + '_i_' + str(my_data['xp']) + '_j_' + str(my_data['yp']) + '_t' + str(my_data['tp']) + '.csv'

    with open(file_name , mode='w' ) as my_file :
        my_writer = csv.writer( my_file , delimiter=';',quotechar='"',quoting=csv.QUOTE_MINIMAL )
        my_writer.writerow(['P','Z','T','Td','qv','theta','thetae','u','v'])
        for iline in range( my_data['p'].size )  :
            my_writer.writerow([ str(my_data['p'][iline]),str(my_data['z'][iline]),str(my_data['t'][iline]),str(my_data['td'][iline]),str(my_data['qv'][iline]),str(my_data['theta'][iline]),str(my_data['thetae'][iline]),str(my_data['u'][iline]),str(my_data['v'][iline] ) ] )





def cmap_map(function,cmap):

    import matplotlib
    import matplotlib.cm     as cm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker


    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = [x[0] for x in cdict[key]]
    step_list = sum(list(step_dict.values()), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map( reduced_cmap, step_list)))
    new_LUT = np.array(list(map( function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  [x + (x[1], ) for x in list(this_cdict.items())]
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def cmap_xmap(function,cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    """

    import matplotlib
    import matplotlib.cm     as cm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    cdict = cmap._segmentdata
    function_to_map = lambda x : (function(x[0]), x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = list(map(function_to_map, cdict[key]))
        cdict[key].sort()
        assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."


    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    import matplotlib
    import matplotlib.cm     as cm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


