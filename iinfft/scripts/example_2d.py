import numpy as np
import matplotlib.pyplot as plt
from iinfft.iinfft import *
#import sym_matrix
import matplotlib
from scipy.io import loadmat, savemat

matplotlib.use('TkAgg')

# Load the modified image
image = np.load('iinfft/data/modified_image.npy')

# Forward transform
N = 128
w = sobk(N, 0.5, 3, 1e-1)
transformed_data, mtot = infft_2d(image, N, w=w)

# Adjoint transform (reconstruction)
reconstructed_data = adjoint_transform_2d(
    transformed_data,
    mtot,
    data_shape=image.shape
)

# Load the .mat file
file_path = "iinfft/data/tic.mat"
mat_data = loadmat(file_path)
original_image = mat_data['M']  # Replace with your actual variable name

# Replace NaNs in the original image with interpolated values
interpolated_image = np.copy(image)
nan_mask = np.isnan(image)
interpolated_image[nan_mask] = reconstructed_data[nan_mask]

# Use "magma" colormap (low values = deep purple, high values = bright yellow)
cmap_choice = 'magma'

# Get dynamic limits for better visibility
vmin = np.nanmin(image)  
vmax = np.nanmax(image) * 0.6  # Reduce max brightness for better mid-range detail

# Define highlight region
x_min, x_max = 1000, 1200
y_min, y_max = 100, 200

# Create figure with full images and highlighted region
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

titles = ["Original TIC", "TIC with NaNs", "Reconstruction", "Original with Interpolated NaNs"]
images = [original_image, image, reconstructed_data, interpolated_image]

for ax, title, img in zip(axes.flat, titles, images):
    ax.set_title(title,fontsize=16)
    im = ax.imshow(img, aspect='auto', cmap=cmap_choice, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               edgecolor='red', facecolor='none', linewidth=2))
    ax.set_xlabel("Modulation No.")  # Add your custom label here
    ax.set_ylabel("Acquisition No.")
    plt.colorbar(im, ax=ax)

plt.tight_layout()

# Save full image with highlights
full_image_path = "full_image_highlighted.png"
plt.savefig(full_image_path, dpi=300)
print(f"Saved full image with highlight: {full_image_path}")

# Create figure with zoomed-in region
fig_zoom, axes_zoom = plt.subplots(2, 2, figsize=(12, 10))

for ax, title, img in zip(axes_zoom.flat, titles, images):
    ax.set_title(f"{title} (Zoomed)")
    im = ax.imshow(img[y_min:y_max, x_min:x_max], aspect='auto', cmap=cmap_choice, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax)

plt.tight_layout()

# Save zoomed-in region
zoomed_image_path = "zoomed_region.png"
plt.savefig(zoomed_image_path, dpi=300)
print(f"Saved zoomed-in image: {zoomed_image_path}")

plt.show()

mdict = {
    'tic_rec': interpolated_image,
    'tic_ori': image, # you can add more arrays here, e.g.
    # 'another_array': some_other_numpy_array
}

# 3. Save to ‘data.mat’ (by default this writes MATLAB level‑5 .mat files)
savemat('export.mat', mdict)
