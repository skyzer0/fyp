import cv2
import numpy as np

# Load the numpy array
array_path = '/home/simtech/IsaacLab/source/standalone/tutorials/06_own_testing/output/camera/distance_to_image_plane_6_17.npy'
numpy_array = np.load(array_path)

# Convert numpy array to OpenCV array
opencv_array = cv2.cvtColor(numpy_array.astype(np.float32), cv2.COLOR_GRAY2BGR)

# Display or use the OpenCV array as needed
cv2.imshow('OpenCV Array', opencv_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
