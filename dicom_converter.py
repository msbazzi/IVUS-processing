import cv2
import numpy as np
import os
import datetime
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, SecondaryCaptureImageStorage
import pydicom

def save_frame_as_dicom(frame, filename):
    # Define file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # Create the FileDataset instance (initially no data elements, but file_meta supplied)
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add the necessary DICOM tags
    ds.Modality = 'OT'  # Other
    ds.ContentDate = datetime.date.today().strftime('%Y%m%d')
    ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')

    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.PatientBirthDate = "20200101"
    ds.PatientSex = "O"

    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = str(1)  # Must be a string

    ds.ImageType = r"ORIGINAL\PRIMARY"
    ds.Rows, ds.Columns, _ = frame.shape
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = frame.tobytes()

    # Additional necessary tags for completeness
    ds.PatientOrientation = ""
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.FrameOfReferenceUID = generate_uid()

    # Add Group Length (0002,0000) tags to the meta information
    ds.fix_meta_info()

    # Save the dataset to file
    ds.save_as(filename)

def main():
    video_path = "/home/bazzi/TEVG/FSG/IVUS-processing/200813IVUSMovie.avi"
    dicom_dir = '/home/bazzi/TEVG/FSG/IVUS-processing/dicom'
    os.makedirs(dicom_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = 17
    end_time = 30
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame
    number_of_frames = 5
    frame_interval = max(1, total_frames // number_of_frames)
    frame_count = 0
    processed_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break

        if frame_count >= start_frame and (frame_count - start_frame) % frame_interval == 0:
            dicom_filename = os.path.join(dicom_dir, f'frame_{processed_frame_count:04d}.dcm')
            save_frame_as_dicom(frame, dicom_filename)
            processed_frame_count += 1

        frame_count += 1

    cap.release()
    print(f'Extracted and processed {processed_frame_count} frames.')

if __name__ == "__main__":
    main()
