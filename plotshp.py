import matplotlib.pyplot as plt
import shapefile
import numpy as np

shp = shapefile.Reader("shp/G2G10")

print(shp.fields)

shapes = shp.shapes()
records = shp.records()

print(len(shapes))

plt.figure()

for shape, record in zip(shapes, records):
    points = np.array(shape.points)
    parts = shape.parts
    parts.append(len(points))

    for i in range(len(parts) - 1):
        plt.plot(points[parts[i]:parts[i+1],0], points[parts[i]:parts[i+1],1], 'k')

plt.show()
