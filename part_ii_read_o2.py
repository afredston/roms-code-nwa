import pyroms
import pyroms_toolbox
import matplotlib.pyplot as plt


# Load grid
grd = pyroms.grid.get_ROMS_grid('NWA')
lon = grd.hgrid.lon_rho
lat = grd.hgrid.lat_rho

# Data file
icfile = '/Users/jeewantha/Code/NWA-SZ.HCob05T_avg_1980-06-19T01:00:00.nc'

# Get variable temp
o2 = pyroms.utility.get_nc_var('o2', icfile)[0]
print(pyroms.utility.get_nc_var('o2', icfile))
# Let's say I want to get the dimensions of temp masked array
print(o2.shape)
print(type(o2))
# Let's say I want to see the grid arranged along a certain vertices
print(o2[-1].shape)
# -1 is the surface
print(type(o2[-1]))
# 0 is the bottom
# print(o2[0])
print(o2[0, 200, 150])

# Plot 2-D view of [O2] at the bottom
plt.figure()
plt.pcolor(lon, lat, o2[0])
plt.colorbar()
plt.axis('image')

# Plot coastline
pyroms_toolbox.plot_coast_line(grd)

# Save plot
outfile = 'o2_bottom_1980_06_19.png'
plt.savefig(outfile, dpi=300, orientation='portrait')
