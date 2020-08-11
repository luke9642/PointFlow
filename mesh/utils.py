import numpy as np
from mayavi import mlab
k = 0
mlab.options.offscreen = True


class Visualisation:
    _colors = ('YlGn', 'YlGnBu')
    # _colors = ('YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd')

    def __init__(self, pts, order=(0, 1, 2)):
        try:
            pts = pts.cpu().detach().numpy()
        except AttributeError:
            pass

        self.pts = pts[:, order]
        self.f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(500, 400))
        mlab.view(azimuth=120, elevation=110, figure=self.f)

    def visualize(self):
        mlab.draw()
        mlab.show()

    def save(self, path):
        mlab.draw()
        mlab.savefig(str(path), figure=self.f)

    def to_numpy(self):
        mlab.draw()
        global k
        k+=1
        print(k)
        return mlab.screenshot(self.f)


class PointCloud(Visualisation):
    def __init__(self, pts, scale=1, active_color=0, order=(0, 1, 2)):
        super().__init__(pts, order)
        mlab.points3d(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], lambda _, __, z: 1 / (1 + np.exp(-z)),
                      figure=self.f, scale_factor=.3, colormap=self._colors[active_color % len(self._colors)])
        self.f.scene.camera.zoom(scale)


class Mesh(Visualisation):
    def __init__(self, pts, triang, scale=1, active_color=0, order=(0, 1, 2)):
        super().__init__(pts, order)
        mlab.triangular_mesh(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], triang, colormap=self._colors[active_color % len(self._colors)])
        self.f.scene.camera.zoom(scale)


# colors = ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges',
#           'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy',
#           'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Vega10', 'Vega20', 'Vega20b',
#           'Vega20c', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'black-white',
#           'blue-red', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'file', 'flag', 'gist_earth',
#           'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2',
#           'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism',
#           'rainbow', 'seismic', 'spectral', 'spring', 'summer', 'terrain', 'viridis', 'winter']
