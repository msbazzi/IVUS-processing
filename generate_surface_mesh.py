import vtk
import xml.etree.ElementTree as ET

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

def create_surface_from_contours(contours):
    append_filter = vtk.vtkAppendPolyData()
    
    for i in range(len(contours) - 1):
        polydata1 = create_vtk_polydata(contours[i])
        polydata2 = create_vtk_polydata(contours[i + 1])
        
        loft_surface = vtk.vtkRuledSurfaceFilter()
        loft_surface.SetInputData(polydata1)
        loft_surface.SetInputData(polydata2)
        loft_surface.SetResolution(50, 5)
        loft_surface.SetRuledModeToResample()
        loft_surface.Update()
        
        append_filter.AddInputData(loft_surface.GetOutput())

    append_filter.Update()
    
    # Clean the polydata to remove any duplicate points
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(append_filter.GetOutput())
    clean_filter.Update()

    return clean_filter.GetOutput()

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

    inner_surface = create_surface_from_contours(inner_contours)
    outer_surface = create_surface_from_contours(outer_contours)

    # Write the resulting surface to STL files
    write_stl(inner_surface, 'inner_wall.stl')
    write_stl(outer_surface, 'outer_wall.stl')

    print('Inner and outer wall surfaces have been written to STL files.')

if __name__ == "__main__":
    main()
