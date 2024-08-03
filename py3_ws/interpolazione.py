import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Genera un array di punti casuali (x, y, z) simulando dati da una depth camera
np.random.seed(0)
n_points = 1600
x = np.random.uniform(-10, 10, n_points)
y = np.random.uniform(-10, 10, n_points)
z = np.zeros(n_points)  # Inizializza z a zero

# Modifica i valori di z per creare una superficie che aumenta verso il centro e poi rimane stabile
z_max = 10
stable_radius = 3  # Raggio entro il quale z rimane stabile
for i in range(n_points):
    r = np.sqrt(x[i]**2 + y[i]**2)
    if r < stable_radius:
        z[i] = z_max
    else:
        z[i] = z_max * (1 - (r - stable_radius) / (10 - stable_radius))

# Filtra i punti più esterni
filter_radius = 8
mask = np.sqrt(x**2 + y**2) <= filter_radius
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]

# Genera una griglia di punti per la superficie
xi = np.linspace(-10, 10, 100)
yi = np.linspace(-10, 10, 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolazione spline
zi = griddata((x_filtered, y_filtered), z_filtered, (xi, yi), method='cubic')

# Calcola il centroide
centroid_x = np.mean(x_filtered)
centroid_y = np.mean(y_filtered)
centroid_z = griddata((x_filtered, y_filtered), z_filtered, (centroid_x, centroid_y), method='cubic')

# Calcola le derivate parziali per ottenere le normali
dzdx, dzdy = np.gradient(zi, xi[0], yi[:, 0])

# Calcola la normale al centroide
centroid_dzdx = griddata((xi.flatten(), yi.flatten()), dzdx.flatten(), (centroid_x, centroid_y), method='cubic')
centroid_dzdy = griddata((xi.flatten(), yi.flatten()), dzdy.flatten(), (centroid_x, centroid_y), method='cubic')
centroid_normal = np.array([-centroid_dzdx, -centroid_dzdy, 1])
centroid_normal /= np.linalg.norm(centroid_normal)

# Visualizza la superficie interpolata
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.6)

# Visualizza il punto centroide
ax.scatter(centroid_x, centroid_y, centroid_z, color='r', s=1)

# Visualizza la normale al centroide con lunghezza aumentata
ax.quiver(centroid_x, centroid_y, centroid_z,
          centroid_normal[0], centroid_normal[1], centroid_normal[2],
          length=0.5, color='b')  # Aumenta il valore di length per ingrandire la normale

plt.show()

