import vtk
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def read_contour_from_ctgr(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    contours = []

    for timestep in root.findall('timestep'):
        for contour in timestep.findall('contour'):
            points = []
            for point in contour.find('control_points').findall('point'):
                x = float(point.get('x'))
                y = float(point.get('y'))
                z = float(point.get('z'))
                points.append([x, y, z])
            contours.append(points)

    return contours

def create_vtk_polydata(points):
    polydata = vtk.vtkPolyData()
    points_vtk = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    num_points = len(points)
    for i, point in enumerate(points):
        points_vtk.InsertNextPoint(point)
        if i > 0:
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i - 1)
            lines.InsertCellPoint(i)
    
    # Close the contour by connecting the last point to the first
    if num_points > 2:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(num_points - 1)
        lines.InsertCellPoint(0)

    polydata.SetPoints(points_vtk)
    polydata.SetLines(lines)
    
    return polydata

def smooth_contour(contour_points, num_iterations=30, pass_band=0.1):
    polydata = create_vtk_polydata(contour_points)
    
    smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
    smooth_filter.SetInputData(polydata)
    smooth_filter.SetNumberOfIterations(num_iterations)
    smooth_filter.SetPassBand(pass_band)
    smooth_filter.Update()

    smoothed_polydata = smooth_filter.GetOutput()
    smoothed_points = []
    for i in range(smoothed_polydata.GetNumberOfPoints()):
        smoothed_points.append(smoothed_polydata.GetPoint(i))

    return smoothed_points

def preprocess_contour(contour_points):
    contour_points = np.array(contour_points)
    if len(contour_points) < 4:
        raise ValueError("Not enough points to interpolate.")
    if not np.array_equal(contour_points[0], contour_points[-1]):
        contour_points = np.vstack([contour_points, contour_points[0]])
    
    # Ensure points are sufficiently distinct
    diff = np.diff(contour_points, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    valid = np.where(dist > 0.5e-3)[0]
    contour_points = contour_points[np.append(valid, valid[-1] + 1)]
    
    if len(contour_points) < 4:
        raise ValueError("Not enough valid points to interpolate.")
    
    return contour_points

def interpolate_and_redistribute(contour_points, num_points=100):
    contour_points = preprocess_contour(contour_points)
    tck, u = splprep([contour_points[:, 0], contour_points[:, 1], contour_points[:, 2]], s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new, z_new = splev(u_new, tck, der=0)
    interpolated_points = np.vstack((x_new, y_new, z_new)).T
    return interpolated_points.tolist()

def visualize_contour(contour_points):
    x = [p[0] for p in contour_points]
    y = [p[1] for p in contour_points]

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title('Smoothed Contour')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def interpolate_contours(contours, num_interpolated_contours):
    all_interpolated_contours = []

    for i in range(len(contours) - 1):
        contour1 = contours[i]
        contour2 = contours[i + 1]

        for j in range(num_interpolated_contours + 1):
            t = j / float(num_interpolated_contours + 1)

            interpolated_contour = []
            for k in range(len(contour1)):
                x = (1 - t) * contour1[k][0] + t * contour2[k][0]
                y = (1 - t) * contour1[k][1] + t * contour2[k][1]
                z = (1 - t) * contour1[k][2] + t * contour2[k][2]
                interpolated_contour.append([x, y, z])

            all_interpolated_contours.append(interpolated_contour)

    return all_interpolated_contours

def combine_contours(contours):
    append_filter = vtk.vtkAppendPolyData()
    
    for contour in contours:
        polydata = create_vtk_polydata(contour)
        append_filter.AddInputData(polydata)
    
    append_filter.Update()
    return append_filter.GetOutput()

def clean_polydata(polydata):
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(polydata)
    clean_filter.Update()
    return clean_filter.GetOutput()

def remesh_polydata(polydata, num_subdivisions=2):
    # Convert all cells to triangles
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()

    # Subdivide triangles to create a finer mesh
    subdivision_filter = vtk.vtkLoopSubdivisionFilter()
    subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
    subdivision_filter.SetNumberOfSubdivisions(num_subdivisions)
    subdivision_filter.Update()

    return subdivision_filter.GetOutput()

def decimate_polydata(polydata, target_reduction=0.5):
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(target_reduction)  # Reduce the number of points by the target percentage
    decimate.PreserveTopologyOn()
    decimate.Update()
    return decimate.GetOutput()

def triangulate_surface(polydata):
    polydata = clean_polydata(polydata)  # Clean the polydata before triangulation

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polydata)
    delaunay.SetTolerance(0.02)  # Adjust the tolerance
    delaunay.SetAlpha(0.0)  # Add alpha parameter to avoid degenerate triangles
    delaunay.Update()
    
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(delaunay.GetOutputPort())
    surface_filter.Update()

    return surface_filter.GetOutput()

def compute_normals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()
    return normals.GetOutput()

def create_cap(points, resolution=20):
    cap_polydata = vtk.vtkPolyData()
    cap_points = vtk.vtkPoints()
    cap_polygon = vtk.vtkPolygon()

    num_points = len(points)
    cap_polygon.GetPointIds().SetNumberOfIds(num_points)

    for i, point in enumerate(points):
        cap_points.InsertNextPoint(point)
        cap_polygon.GetPointIds().SetId(i, i)

    cap_cells = vtk.vtkCellArray()
    cap_cells.InsertNextCell(cap_polygon)

    cap_polydata.SetPoints(cap_points)
    cap_polydata.SetPolys(cap_cells)

    return cap_polydata

def cap_polydata(contours_polydata, contours):
    append_filter = vtk.vtkAppendPolyData()
    append_filter.AddInputData(contours_polydata)

    # Cap first contour
    first_contour = create_cap(contours[0], resolution=50)
    append_filter.AddInputData(first_contour)

    # Cap last contour
    last_contour = create_cap(contours[-1], resolution=50)
    append_filter.AddInputData(last_contour)

    append_filter.Update()
    
    return append_filter.GetOutput()

def write_stl(polydata, filename):
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        print(f"No data to write for {filename}.")
        return

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

def main():
    inner_contours = read_contour_from_ctgr('/home/bazzi/TEVG/FSG/IVUS-processing/contours_ctgr/contours_inner.ctgr')
    outer_contours = read_contour_from_ctgr('/home/bazzi/TEVG/FSG/IVUS-processing/contours_ctgr/contours_outer.ctgr')

    print(f"Read {len(inner_contours)} inner contours.")
    print(f"Read {len(outer_contours)} outer contours.")

    # Smooth the initial contours
    smoothed_inner_contours = [smooth_contour(contour) for contour in inner_contours]
    smoothed_outer_contours = [smooth_contour(contour) for contour in outer_contours]

    # Interpolate and redistribute points uniformly around the contour
    uniform_inner_contours = [interpolate_and_redistribute(contour, num_points=20) for contour in smoothed_inner_contours]
    uniform_outer_contours = [interpolate_and_redistribute(contour, num_points=20) for contour in smoothed_outer_contours]

    # Visualize the first uniform inner contour
    visualize_contour(uniform_inner_contours[0])

    num_interpolated_contours = 4  # Further reduce the number of interpolated contours for better performance
    inner_interpolated = interpolate_contours(uniform_inner_contours, num_interpolated_contours)
    outer_interpolated = interpolate_contours(uniform_outer_contours, num_interpolated_contours)

    all_inner_contours = uniform_inner_contours + inner_interpolated
    all_outer_contours = uniform_outer_contours + outer_interpolated
    print(f"Created {len(all_outer_contours)} outer contours.")

    # Combine the contours
    inner_combined = combine_contours(all_inner_contours)
    outer_combined = combine_contours(all_outer_contours)

    inner_surface = triangulate_surface(inner_combined)
    outer_surface = triangulate_surface(outer_combined)

    # Compute normals to improve mesh quality
    inner_surface_normals = compute_normals(inner_surface)
    outer_surface_normals = compute_normals(outer_surface)
    '''
    # Remesh the surfaces to reduce the edge length
    inner_surface_remeshed = remesh_polydata(inner_surface_normals, num_subdivisions=1)
    outer_surface_remeshed = remesh_polydata(outer_surface_normals, num_subdivisions=1) 
    # Decimate the surfaces to reduce the number of points
    inner_surface_decimated = decimate_polydata(inner_surface, target_reduction=0.5)
    outer_surface_decimated = decimate_polydata(outer_surface, target_reduction=0.5)
    '''
    # Cap the surfaces using first and last contours
    inner_capped = cap_polydata(inner_surface, inner_contours)
    outer_capped = cap_polydata(outer_surface, outer_contours)

    # Write the resulting surface to STL files
    write_stl(inner_capped, 'inner_wall_capped.stl')
    write_stl(outer_capped, 'outer_wall_capped.stl')

    print('Inner and outer wall surfaces with interpolated contours have been written to STL files.')

if __name__ == "__main__":
    main()
