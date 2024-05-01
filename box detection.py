#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Load image
image = cv2.imread(r'C:\Users\MSI\Downloads\box.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (Canny)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and draw bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import cv2
import numpy as np

# Load image
image = cv2.imread(r'C:\Users\MSI\Downloads\boxx.png')

# Resize image to improve processing speed
resized_image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Convert image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection (Canny)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and draw bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

# Load ToF amplitude image
image = cv2.imread(r'C:\Users\MSI\Downloads\boxx.png', cv2.IMREAD_GRAYSCALE)

# Thresholding to create binary image
_, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and draw bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import cv2
import numpy as np

# Load RANSAC image
image = cv2.imread(r'C:\Users\MSI\Downloads\boxx.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv2.Canny(gray, 50, 150)

# Find lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Iterate through lines and draw them on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection from RANSAC Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import cv2
import numpy as np

# Load floor mask image
floor_mask = cv2.imread(r'C:\Users\MSI\Downloads\boxx.png', 0)

# Apply threshold to create binary image
ret, thresh = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    # Approximate polygonal curves to reduce the number of vertices
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If contour has 4 vertices, it's likely a rectangle representing a box
    if len(approx) == 4:
        # Draw contour on original image
        cv2.drawContours(floor_mask, [approx], 0, (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection from Floor Mask', floor_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import cv2
import numpy as np

# Load filtered floor mask image
filtered_floor_mask = cv2.imread(r'C:\Users\MSI\Downloads\boxx.png', 0)

# Apply threshold to create binary image
ret, thresh = cv2.threshold(filtered_floor_mask, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    # Approximate polygonal curves to reduce the number of vertices
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If contour has 4 vertices, it's likely a rectangle representing a box
    if len(approx) == 4:
        # Draw contour on original image
        cv2.drawContours(filtered_floor_mask, [approx], 0, (0, 255, 0), 2)

# Display result
cv2.imshow('Box Detection from Filtered Floor Mask', filtered_floor_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

# Load image
image = cv2.imread(r'C:\Users\MSI\Downloads\box.jpg', 0)

# Perform edge detection
edges = cv2.Canny(image, 50, 150)

# Apply RANSAC algorithm for line fitting
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display result
cv2.imshow('Box Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import cv2
import numpy as np

# Load image
image = cv2.imread(r'C:\Users\MSI\Downloads\boxx.jpg', 0)

# Perform edge detection
edges = cv2.Canny(image, 50, 150)

# Apply RANSAC algorithm for line fitting
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

# Draw detected lines on the image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display result
cv2.imshow('Box Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




