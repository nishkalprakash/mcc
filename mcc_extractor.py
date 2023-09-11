# %%
# from os import path
# if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
#     !wget https://biolab.csr.unibo.it/samples/fr/files.zip
#     !unzip files.zip

# %%
from pathlib import Path
Path.cwd()

# %%
import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
# from ipywidgets import interact

# %%
# Path to the image to be loaded
img_path = Path(r"D:\fvc_fingerprint_datasets\ASRA\FVC2002_DB1A_ASRA_Auto\1_1.tif") 

# %% [markdown]
# # Step 1: Fingerprint segmentation

# %% [markdown]
# First of all we load a fingerprint image: it will be stored in memory as a NumPy array.

# %%
fingerprint = cv.imread(img_path.as_posix(), cv.IMREAD_GRAYSCALE)
show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')

# %% [markdown]
# The segmentation step is aimed at separating the fingerprint area (foreground) from the background.
# The foreground is characterized by the presence of a striped and oriented pattern; background presents a uniform pattern.   
# We will use a very simple technique based on the magnitude of the local gradient.

# %%
# Calculate the local gradient (using Sobel filters)
gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
show((gx, 'Gx'), (gy, 'Gy'))

# %%
# Calculate the magnitude of the gradient for each pixel
gx2, gy2 = gx**2, gy**2
gm = np.sqrt(gx2 + gy2)
show((gx2, 'Gx**2'), (gy2, 'Gy**2'), (gm, 'Gradient magnitude'))

# %%
# Integral over a square window
sum_gm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
show(sum_gm, 'Integral of the gradient magnitude')

# %%
# Use a simple threshold for segmenting the fingerprint pattern
thr = sum_gm.max() * 0.2
mask = cv.threshold(sum_gm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
show(fingerprint, mask, cv.merge((mask, fingerprint, fingerprint)))

# %% [markdown]
# # Step 2: Estimation of local ridge orientation

# %% [markdown]
# The local ridge orientation at $(j,i)$ is the angle $\theta_{j,i}\in[0,180°[$ that the fingerprint ridges form with the horizontal axis in an arbitrary small neighborhood centered in $(j,i)$.  
# For each pixel, we will estimate the local orientation from the gradient $[Gx,Gy]$, which we already computed in the segmentation step (see *A.M. Bazen and S.H. Gerez, "Systematic methods for the computation of the directional fields and singular points of fingerprints," in IEEE tPAMI, July 2002*).  
# 
# The ridge orientation is estimated as ortoghonal to the gradient orientation, averaged over a window $W$.  
# 
# $G_{xx}=\sum_W{G_x^2}$, $G_{yy}=\sum_W{G_y^2}$, $G_{xy}=\sum_W{G_xG_y}$
# 
# $\theta=\frac{\pi}{2} + \frac{phase(G_{xx}-G_{yy}, 2G_{xy})}{2}$
# 
# For each orientation, we will also calculate a confidence value (strength), which measures how much all gradients in $W$ share the same orientation.  
# 
# $strength = \frac{\sqrt{(G_{xx}-G_{yy})^2+(2G_{xy})^2}}{G_{xx}+G_{yy}}$

# %%
W = (23, 23)
gxx = cv.boxFilter(gx2, -1, W, normalize = False)
gyy = cv.boxFilter(gy2, -1, W, normalize = False)
gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
gxx_gyy = gxx - gyy
gxy2 = 2 * gxy

orientations = (cv.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
sum_gxx_gyy = gxx + gyy
strengths = np.divide(cv.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)
show(draw_orientations(fingerprint, orientations, strengths, mask, 1, 16), 'Orientation image')

# %% [markdown]
# # Step 3: Estimation of local ridge frequency

# %% [markdown]
# The local ridge frequency $f_{j,i}$ at $(j,i)$ is the number of ridges per unit length along a hypothetical segment centered in $(j,i)$ and orthogonal to the local ridge orientation $\theta_{j,i}$.
# 
# For simplicity, we will assume a constant frequency over all the fingerprint and estimate its reciprocal (the ridge-line period) from a small region of the image.

# %%
region = fingerprint[10:90,80:130]
show(region)

# %% [markdown]
# Then the *x-signature* is computed from the region and the ridge-line period is estimated as the average number of pixels between two consecutive peaks (see *L. Hong, Y. Wan and A. Jain, "Fingerprint image enhancement: algorithm and performance evaluation," in IEEE tPAMI, Aug. 1998*)

# %%
# before computing the x-signature, the region is smoothed to reduce noise
smoothed = cv.blur(region, (5,5), -1)
xs = np.sum(smoothed, 1) # the x-signature of the region
print(xs)

# %%
x = np.arange(region.shape[0])
f, axarr = plt.subplots(1,2, sharey = True)
axarr[0].imshow(region,cmap='gray')
axarr[1].plot(xs, x)
axarr[1].set_ylim(region.shape[0]-1,0)
plt.show()

# %%
# Find the indices of the x-signature local maxima
local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]

# %%
x = np.arange(region.shape[0])
plt.plot(x, xs)
plt.xticks(local_maxima)
plt.grid(True, axis='x')
plt.show()

# %%
# Calculate all the distances between consecutive peaks
distances = local_maxima[1:] - local_maxima[:-1]
print(distances)

# %%
# Estimate the ridge line period as the average of the above distances
ridge_period = np.average(distances)
print(ridge_period)

# %% [markdown]
# # Step 4: Fingerprint enhancement

# %% [markdown]
# In order to enhance the fingerprint pattern, we will perform a *contextual convolution* with a bank of Gabor filters.  
# In this simple example we are using a constant ridge-line frequency, hence all the filters will have the same period and the only parameter will be the number of orientations (or_count).  
# As it is a contextual convolution, a different filter should be applied to each pixel, according to the corresponding ridge-line orientation. Unfortunately this kind of operation is not available in OpenCV and implementing it in Python would be very inefficient; hence, we will apply all the filters to the whole image (that is, producing a filtered image for each filter) and then we will assemble the enhanced image taking the right pixel from each filtered image, using the discretized orientation indices as a lookup table.

# %%
# Create the filter bank
or_count = 8
gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]

# %%
show(*gabor_bank)

# %%
# Filter the whole image with each filter
# Note that the negative image is actually used, to have white ridges on a black background as a result
nf = 255-fingerprint
all_filtered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gabor_bank])
show(nf, *all_filtered)

# %%
y_coords, x_coords = np.indices(fingerprint.shape)
# For each pixel, find the index of the closest orientation in the gabor bank
orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
# Take the corresponding convolution result for each pixel, to assemble the final result
filtered = all_filtered[orientation_idx, y_coords, x_coords]
# Convert to gray scale and apply the mask
enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
show(fingerprint, enhanced)

# %% [markdown]
# # Step 5: Detection of minutiae positions

# %% [markdown]
# In this simple example, minutiae will be detected from the *ridge line skeleton*, obtained by binarizing and thinning the enhanced ridge lines.

# %%
# Binarization
_, ridge_lines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
show(fingerprint, ridge_lines, cv.merge((ridge_lines, fingerprint, fingerprint)))

# %%
# Thinning
skeleton = cv.ximgproc.thinning(ridge_lines, thinningType = cv.ximgproc.THINNING_GUOHALL)
show(skeleton, cv.merge((fingerprint, fingerprint, skeleton)))

# %% [markdown]
# Then, for each pixel $p$ of the skeleton, the *crossing number* $cn(p)$ is computed as the number of transitions from black to white pixels in its 8-neighborhood
# 
# ```
#    v[0] v[1] v[2]
#    v[7]   p  v[3]
#    v[6] v[5] v[4]
# ```
# $cn(v)=\sum_{i=0}^7\begin{cases} 1 & \mbox{if } v[i]<v[(i+1) \mod 8] \\ 0 & \mbox{otherwise} \end{cases}$

# %%
def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))

# %% [markdown]
# To efficiently compute all the crossing numbers, a 3x3 filter is used to convert each possible 8-neighborhood into a byte value (by considering each pixel as a bit).  
# Then a lookup table maps each byte value [0,255] into the corresponding crossing number.

# %%
# Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
cn_filter = np.array([[  1,  2,  4],
                      [128,  0,  8],
                      [ 64, 32, 16]
                     ])

# %%
# Create a lookup table that maps each byte value to the corresponding crossing number
all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

# %%
cn_lut

# %%
a = all_8_neighborhoods[100]
show(*[np.insert(a, 4, 2).reshape(3,-1) for a in all_8_neighborhoods], max_per_row = 16)

# %%
# Skeleton: from 0/255 to 0/1 values
skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
# Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255]
cn_values = cv.filter2D(skeleton01, -1, cn_filter, borderType = cv.BORDER_CONSTANT)
# Apply the lookup table to obtain the crossing number of each pixel
cn = cv.LUT(cn_values, cn_lut)
# Keep only crossing numbers on the skeleton
cn[skeleton==0] = 0

# %%
terminations = np.zeros_like(cn)
terminations[cn == 1] = 255
biufurcations = np.zeros_like(cn)
biufurcations[cn == 3] = 255
show(skeleton, terminations, biufurcations)

# %% [markdown]
# The list of minutiae is finally obtained from the coordinates of pixels with crossing number 1 (terminations) or 3 (bifurcations).  
# Each minutia is stored as a tuple $(x, y, t)$ where $t$ is $true$ for terminations.  
# Note that, for now, we are not computing the minutiae *direction*, but only their location and type.

# %%
# crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]

# %%
show(draw_minutiae(fingerprint, minutiae), skeleton, draw_minutiae(skeleton, minutiae))

# %% [markdown]
# From the above image we can note that near the borders of the pattern many false minutiae are detected: we can remove them by computing the *distance transform* of the segmentation mask and choosing a threshold so that minutiae too close to the mask border are excluded.

# %%
# A 1-pixel background border is added to the mask before computing the distance transform
mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 50, 50, 50, 50, cv.BORDER_CONSTANT), cv.DIST_C, 3)[50:-50,50:-50]
show(mask, mask_distance)

# %%
filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>25, minutiae))

# %%
show(draw_minutiae(fingerprint, filtered_minutiae), skeleton, draw_minutiae(skeleton, filtered_minutiae))

# %% [markdown]
# # Step 6: Estimation of minutiae directions

# %% [markdown]
# The direction of a termination will be computed by following the ridge-line until another minutia is found or a distance of 20 pixels has been traveled.  
# The direction of a bifurcation will be computed by considering the directions ($\theta_1$, $\theta_2$, $\theta_3$) of its three branches and calculating the mean of the two closest ones ($\theta_1$ and $\theta_2$ in the example below).  
# Note that this is a simplified definition of minutiae direction, not completely consistent to the ISO standard minutiae-based template format, which is based on ridge and valley skeletons (see *ISO/IEC 19794-2, 2005*).
# 
# <img src="https://biolab.csr.unibo.it/samples/fr/images/min_directions.png">

# %% [markdown]
# In order to follow the ridge-line, for each position on the skeleton, the position of the next pixel to be visited must be determined. In the following, the eight possible direction of movement will be encoded as integer numbers in [0,7], following the same ordering previously used for the 8-neighborhood of a pixel $p$:
# 
# ```
#    0  1  2
#    7  p  3
#    6  5  4
# ```
# 
# The following function, given the previous direction of movement and the values of the 8 neighboring pixels, returns the  directions towards neighboring skeleton pixels, excluding the previously-visited one. A special value (8) indicates that there is no previous direction: it will be used at the first step.

# %%
def compute_next_ridge_following_directions(previous_direction, values):
    next_positions = np.argwhere(values!=0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        #                                except the direction, if any, that corresponds to the previous position
        next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (previous_direction + 4) % 8: # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1] # removes it
    return next_positions

# %% [markdown]
# It is always a good idea to avoid unnecessary computations: the next cell prepares two look-up tables.

# %%
r2 = 2**0.5 # sqrt(2)

# The eight possible (x, y) offsets with each corresponding Euclidean distance
xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

# LUT: for each 8-neighborhood and each previous direction [0,8],
#      where 8 means "none", provides the list of possible directions
nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

# %% [markdown]
# The next function follows the skeleton until another minutia is found or a distance of 20 pixels has been traveled. If a minimum length of 10 pixels has been reached, it returns the corresponding angle, otherwise it returns None.

# %%
def follow_ridge_and_compute_angle(x, y, d = 8):
    px, py = x, y
    length = 0.0
    while length < 20: # max length followed
        next_directions = nd_lut[cn_values[py,px]][d]
        if len(next_directions) == 0:
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break # another minutia found: we stop here
        # Only the first direction has to be followed
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox ; py += oy ; length += l
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py+y, px-x) if length >= 10 else None

# %% [markdown]
# Finally, the cell below estimates all minutiae directions:
# - in case of a termination, it simply calls the previous function,
# - in case of a bifurcation, it follows the three branches and if all the three angles are valid, it computes the mean of the two closest ones.
# 
# The list of minutiae is finally obtained, with each minutia stored as a tuple $(x, y, t, d)$ where $t$ is $true$ for terminations and $d$ is the minutia direction in radians.

# %%
valid_minutiae = []
for x, y, term in filtered_minutiae:
    d = None
    if term: # termination: simply follow and compute the direction
        d = follow_ridge_and_compute_angle(x, y)
    else: # bifurcation: follow each of the three branches
        dirs = nd_lut[cn_values[y,x]][8] # 8 means: no previous direction
        if len(dirs)==3: # only if there are exactly three branches
            angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
            if all(a is not None for a in angles):
                a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                d = angle_mean(a1, a2)
    if d is not None:
        valid_minutiae.append( (x, y, term, d) )

# %%
show(draw_minutiae(fingerprint, valid_minutiae))

# %% [markdown]
# # Step 7: Creation of local structures

# %% [markdown]
# In this section, starting from minutiae positions and directions, we will create local structures invariant for translation and rotation, which can be used for comparing fingerprints without a pre-alignment step.  
# We will use a simplified version of Minutia Cylinder-Code (MCC, see *Minutia Cylinder-Code: a new representation and matching technique for fingerprint recognition", IEEE tPAMI 2010*): while MCC local structures can be represented as 3D structures (cylinders), where the base encodes spatial relationships between minutiae and the height directional relationships, here we will consider only the base of the cylinders, which is rotated according to the minutia direction and discretized into a fixed number of cells.
# 
# <img src="https://biolab.csr.unibo.it/samples/fr/images/simple_mcc.png">
# 

# %%
# Compute the cell coordinates of a generic local structure
mcc_radius = 70
mcc_size = 16

g = 2 * mcc_radius / mcc_size
x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
y = x[..., np.newaxis]
iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
ref_cell_coords = np.column_stack((x[ix], x[iy]))

# %%
mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2

def Gs(t_sqr):
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

def Psi(v):
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))

# %%
# n: number of minutiae
# c: number of cells in a local structure

xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae]) # matrix with all minutiae coordinates and directions (n x 3)

# rot: n x 2 x 2 (rotation matrix for each minutia)
d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

# rot@ref_cell_coords.T : n x 2 x c
# xy : n x 2
xy = xyd[:,:2]
# cell_coords: n x c x 2 (cell coordinates for each local structure)
cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

# cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
# xy                                 : (1 x 1) x n x 2
# cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
# dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

# cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
cs = Gs(dists)
diag_indices = np.arange(cs.shape[0])
cs[diag_indices,:,diag_indices] = 0 # remove the contribution of each minutia to its own cells

# local_structures : n x c (cell values for each local structure)
local_structures = Psi(np.sum(cs, -1))

# %%
@interact(i=(0,len(valid_minutiae)-1))
def test(i=0):
    show(draw_minutiae_and_cylinder(fingerprint, ref_cell_coords, valid_minutiae, local_structures, i))

# %% [markdown]
# # Step 8: Fingerprint comparison

# %% [markdown]
# We started from an image (*fingerprint*), obtained a list of minutiae (*valid_minutiae*) and their corresponding local structures (matrix *local_structures* with a row for each minutia):

# %%
print(f"""Fingerprint image: {fingerprint.shape[1]}x{fingerprint.shape[0]} pixels
Minutiae: {len(valid_minutiae)}
Local structures: {local_structures.shape}""")

# %% [markdown]
# In the following we will more concisely name them f1, m1, ls1:

# %%
f1, m1, ls1 = fingerprint, valid_minutiae, local_structures

# %% [markdown]
# Then we will load analogous data of another fingerprint: f2, m2, ls2

# %%
ofn = 'samples/sample_1_2' # Fingerprint of the same finger
ofn = 'samples/sample_2' # Fingerprint of a different finger
f2, (m2, ls2) = cv.imread(f'{ofn}.png', cv.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()

# %% [markdown]
# The similarity between two local structures $r_1$ and $r_2$ can be simply computed as one minus their normalized Euclidean distance:
# 
# $similarity(r_1, r_2) = 1 - \frac {\| r_1 - r_2 \|}{\|r_1\| + \|r_2\|}$
# 
# The following cell computes the matrix of all normalized Euclidean distances between local structures in ls1 and ls2.

# %%
# Compute all pairwise normalized Euclidean distances between local structures in v1 and v2
# ls1                       : n1 x  c
# ls1[:,np.newaxis,:]       : n1 x  1 x c
# ls2                       : (1 x) n2 x c
# ls1[:,np.newaxis,:] - ls2 : n1 x  n2 x c
# dists                     : n1 x  n2
dists = np.sqrt(np.sum((ls1[:,np.newaxis,:] - ls2)**2, -1))
dists /= (np.sqrt(np.sum(ls1**2, 1))[:,np.newaxis] + np.sqrt(np.sum(ls2**2, 1))) # Normalize as in eq. (17) of MCC paper

# %% [markdown]
# In the next cell, we finally compare the two fingerprints by using the Local Similarity Sort (LSS) technique (see eq. 23 in the MCC paper), which simply selects the num_p highest similarities (i.e. the smallest distances) and compute the comparison *score* as their average. The indices of the corresponding minutia pairs are stored in *pairs*, and used in the last cell to show the result.

# %%
# Select the num_p pairs with the smallest distances (LSS technique)
num_p = 2 # For simplicity: a fixed number of pairs
pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper
print(f'Comparison score: {score:.2f}')

# %%
@interact(i = (0,len(pairs[0])-1), show_local_structures = False)
def show_pairs(i=0, show_local_structures = False):
    show(draw_match_pairs(f1, m1, ls1, f2, m2, ls2, ref_cell_coords, pairs, i, show_local_structures))

# %%



