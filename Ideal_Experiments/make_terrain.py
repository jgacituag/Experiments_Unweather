#Gen topo

import numpy as np


nx = 202
dx = 200.0

hill_heigth = 1000.0
hill_width  = 5000.0


x = ( np.arange( nx ) - nx/2) * dx

terrain = hill_heigth * np.exp( - ( x ** 2 )/(hill_width**2) )

#import matplotlib.pyplot as plt
#plt.plot( x , terrain )
#plt.show()

f = open('terrain.txt', 'wb')

char = 'x heigth(x) \n'
f.write( char.encode('ascii') )

for ii in range( nx ):

    char = str(x[ii]) + ',' + str(terrain[ii]) + '\n'
    f.write( char.encode('ascii') )







