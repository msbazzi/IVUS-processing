import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def generate_3d_model(contours_dir, output_file):
    # Load contours
    outer_contours = []
    inner_contours = []
    for file in sorted(os.listdir(contours_dir)):
        if file.startswith('outer_contour_'):
            outer_contours.append(np.load(os.path.join(contours_dir, file)))
        elif file.startswith('inner_contour_'):
            inner_contours.append(np.load(os.path.join(contours_dir, file)))

    # Ensure both lists have the same length
    assert len(outer_contours) == len(inner_contours)

    # Create 3D model
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(outer_contours)):
        outer = outer_contours[i]
        inner = inner_contours[i]
        z = np.full(outer.shape[0], i)
        ax.plot(outer[:, 0], outer[:, 1], z, color='r')
        ax.plot(inner[:, 0], inner[:, 1], z, color='b')

        # Create polygons for the outer and inner contours
        verts_outer = [list(zip(outer[:, 0], outer[:, 1], z))]
        verts_inner = [list(zip(inner[:, 0], inner[:, 1], z))]
        ax.add_collection3d(Poly3DCollection(verts_outer, color='r', alpha=0.3))
        ax.add_collection3d(Poly3DCollection(verts_inner, color='b', alpha=0.3))

    # Remove grid and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    contours_dir = '/home/bazzi/TEVG/FSG/IVUS-processing/contours'
    output_file = '/home/bazzi/TEVG/FSG/IVUS-processing/3d_model.png'
    generate_3d_model(contours_dir, output_file)
