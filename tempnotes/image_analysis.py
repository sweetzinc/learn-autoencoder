import os
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import numpy as np

def calculate_power_spectrum(image):
    """
    Calculate power spectrum in x and y directions of a 2D image by summing
    along perpendicular directions.
    
    Parameters:
    image (numpy.ndarray): 2D input image array
    
    Returns:
    dict: Dictionary containing:
        'px': Power spectrum along x-direction (summed over y)
        'py': Power spectrum along y-direction (summed over x)
        'freq_x': Frequency axis for x-direction
        'freq_y': Frequency axis for y-direction
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
    
    # Calculate 2D Fourier transform
    fourier = np.fft.fft2(image)
    
    # Shift zero frequency to center
    fourier_shifted = np.fft.fftshift(fourier)
    
    # Calculate power spectrum (magnitude squared)
    power_spectrum_2d = np.abs(fourier_shifted)**2
    
    # Get dimensions
    rows, cols = power_spectrum_2d.shape
    # Calculate center points
    center_row = rows // 2
    center_col = cols // 2

    # Calculate power spectrum along x-direction (horizontal line through center)
    px_c = power_spectrum_2d[center_row, :]
    # Calculate power spectrum along y-direction (vertical line through center)
    py_c = power_spectrum_2d[:, center_col]

    # Calculate power spectrum along x-direction by summing over y
    px = np.sum(power_spectrum_2d, axis=0)  # Sum along rows (y-direction)
    # Calculate power spectrum along y-direction by summing over x
    py = np.sum(power_spectrum_2d, axis=1)  # Sum along columns (x-direction)
    
    # Create frequency axes for proper scaling
    freq_x = np.fft.fftshift(np.fft.fftfreq(cols))
    freq_y = np.fft.fftshift(np.fft.fftfreq(rows))
    
    # Normalize the spectra
    px = px / rows  # Normalize by number of rows
    py = py / cols  # Normalize by number of columns
    
    return {
        'px': px,
        'py': py,
        'px_c': px_c,
        'py_c': py_c,  
        'freq_x': freq_x,
        'freq_y': freq_y
    }

def calculate_histogram(image):
    """
    Calculate histogram of a grayscale image with 256 levels.
    
    Parameters:
    image (numpy.ndarray): 2D input image array of type uint8
    
    Returns:
    tuple: (histogram, bin_edges) where:
        histogram (numpy.ndarray): Count of pixels for each gray level (0-255)
        bin_edges (numpy.ndarray): Array of bin edges (0-256)
    """
    # Input validation
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8")
    
    # Method 1: Using numpy.histogram
    hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    
    # Alternative Method 2: Using numpy.bincount (faster for uint8)
    # hist = np.bincount(image.ravel(), minlength=256)
    
    return {
        'histogram': hist,
        'bin_edges': bin_edges,
        'bin_centers': np.arange(256)  # Center of each bin (0-255)
    }

def process_image_directory(image_dir, output_h5_path):
    """Process all images in directory and save results to H5 file"""
    
    # Get list of image files
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Open first image to get dimensions
    test_img = np.array(Image.open(image_files[0]).convert('L'))
    rows, cols = test_img.shape
    n_images = len(image_files)
    
    # Create H5 file and datasets
    with h5py.File(output_h5_path, 'w') as f:
        # Create datasets with maxshape for appending
        specs = f.create_group('power_spectra')
        specs.create_dataset('px', shape=(0, cols), maxshape=(None, cols))
        specs.create_dataset('py', shape=(0, rows), maxshape=(None, rows))
        specs.create_dataset('px_c', shape=(0, cols), maxshape=(None, cols))
        specs.create_dataset('py_c', shape=(0, rows), maxshape=(None, rows))
        specs.create_dataset('freq_x', data=np.fft.fftshift(np.fft.fftfreq(cols)))
        specs.create_dataset('freq_y', data=np.fft.fftshift(np.fft.fftfreq(rows)))
        
        hists = f.create_group('histograms')
        hists.create_dataset('counts', shape=(0, 256), maxshape=(None, 256))
        hists.create_dataset('bin_edges', data=np.arange(257))
        
        # Add filenames dataset
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('filenames', shape=(0,), maxshape=(None,), dtype=dt)
        
        # Process each image
        for i, img_path in enumerate(image_files):
            # Read and convert image to grayscale
            img = np.array(Image.open(img_path).convert('L'))
            
            # Calculate power spectrum and histogram
            power_spec = calculate_power_spectrum(img)
            hist_data = calculate_histogram(img)
            
            # Resize datasets
            current_size = specs['px'].shape[0]
            new_size = current_size + 1
            
            specs['px'].resize(new_size, axis=0)
            specs['py'].resize(new_size, axis=0)
            specs['px_c'].resize(new_size, axis=0)
            specs['py_c'].resize(new_size, axis=0)
            hists['counts'].resize(new_size, axis=0)
            f['filenames'].resize(new_size, axis=0)
            
            # Add new data
            specs['px'][current_size] = power_spec['px']
            specs['py'][current_size] = power_spec['py']
            specs['px_c'][current_size] = power_spec['px_c']
            specs['py_c'][current_size] = power_spec['py_c']
            hists['counts'][current_size] = hist_data['histogram']
            f['filenames'][current_size] = str(img_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{n_images} images")

if __name__ == '__main__':
    # Example usage
    image_dir = "path/to/image/folder"
    output_file = "image_analysis.h5"
    process_image_directory(image_dir, output_file)
