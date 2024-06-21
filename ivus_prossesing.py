import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

# Global variables for masks
outer_mask = None
inner_mask = None

# Function to preprocess frames
def preprocess_frame(frame):
    global outer_mask, inner_mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=outer_mask)
    masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=cv2.bitwise_not(inner_mask))
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Reduce noise
    median = cv2.medianBlur(enhanced, 5)
    return median

# Function to calculate vessel thickness
def calculate_thickness(inner_contour, outer_contour):
    outer_area = cv2.contourArea(outer_contour)
    inner_area = cv2.contourArea(inner_contour)
    thickness = (outer_area - inner_area) ** 0.5  # Simplified thickness calculation
    return thickness

def draw_spline_contours(frame):
    points = []

    def draw_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Draw Points", frame)
    
    cv2.namedWindow("Draw Points")
    cv2.setMouseCallback("Draw Points", draw_points)
    cv2.imshow("Draw Points", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) > 3:
        points = np.array(points)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        return spline_points
    else:
        return np.array(points)

def main():
    global outer_mask, inner_mask

    # Load the IVUS video file
    video_path = "/home/bazzi/TEVG/FSG/200813IVUSMovie.avi"
    # Directory to save extracted frames
    frames_dir = '/home/bazzi/TEVG/FSG/frames'
    plots_dir = '/home/bazzi/TEVG/FSG/plots'
    contours_dir = '/home/bazzi/TEVG/FSG/contours'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(contours_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = 17  # Start time in seconds
    end_time = 30  # End time in seconds
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    frame_interval = max(1, total_frames // 5)  # Ensure we get 5 frames

    frame_count = 0
    processed_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break

        if frame_count >= start_frame and (frame_count - start_frame) % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f'frame_{processed_frame_count:04d}.png')
            cv2.imwrite(frame_path, frame)
            processed_frame_count += 1

        frame_count += 1

    cap.release()
    print(f'Extracted and processed {processed_frame_count} frames.')

    thicknesses = []
    for i in range(processed_frame_count):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        frame = cv2.imread(frame_path)

        # Draw spline contours for inner and outer walls
        print(f"Draw outer contour for frame {i}")
        outer_contour = draw_spline_contours(frame.copy())
        print(f"Draw inner contour for frame {i}")
        inner_contour = draw_spline_contours(frame.copy())

        # Save contours
        np.save(os.path.join(contours_dir, f'outer_contour_{i:04d}.npy'), outer_contour)
        np.save(os.path.join(contours_dir, f'inner_contour_{i:04d}.npy'), inner_contour)

        # Create masks based on the drawn contours
        inner_mask = np.zeros_like(frame[:, :, 0])
        outer_mask = np.zeros_like(frame[:, :, 0])
        if len(inner_contour) > 0:
            cv2.drawContours(inner_mask, [inner_contour], -1, 255, -1)
        if len(outer_contour) > 0:
            cv2.drawContours(outer_mask, [outer_contour], -1, 255, -1)

        # Debugging: Show the masks
        cv2.imshow(f"Inner Mask - Frame {i}", inner_mask)
        cv2.imshow(f"Outer Mask - Frame {i}", outer_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Debugging: Ensure masks are correctly applied
        masked_frame = cv2.bitwise_and(frame, frame, mask=outer_mask)
        masked_frame = cv2.bitwise_and(masked_frame, masked_frame, mask=cv2.bitwise_not(inner_mask))
        cv2.imshow(f"Masked Frame - Frame {i}", masked_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Use manually drawn contours directly
        thickness = calculate_thickness(inner_contour, outer_contour)
        if thickness is not None:
            thicknesses.append(thickness)

            # Draw contours on the frame
            cv2.drawContours(frame, [outer_contour], -1, (0, 0, 255), 2)  # Red for outer contour
            cv2.drawContours(frame, [inner_contour], -1, (0, 0, 255), 2)  # Red for inner contour

            # Save the frame with contours
            plot_path = os.path.join(plots_dir, f'plot_{i:04d}.png')
            cv2.imwrite(plot_path, frame)
            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.plot(outer_contour[:, 0], outer_contour[:, 1], 'r')
            plt.plot(inner_contour[:, 0], inner_contour[:, 1], 'r')
            plt.savefig(plot_path)
            plt.close()
        else:
            print(f"Thickness not calculated for frame {i}")

    # Save thickness values
    np.savetxt('/home/bazzi/TEVG/FSG/thicknesses.csv', thicknesses, delimiter=',')
    print(f'Saved thickness values for {len(thicknesses)} frames.')

if __name__ == "__main__":
    main()
