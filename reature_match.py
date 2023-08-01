import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img1 = cv.imread("data/2-d.png", cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('data/target.png', cv.IMREAD_GRAYSCALE) # trainImage
img3 =  cv.imread('data/target2.png', cv.IMREAD_GRAYSCALE) # trainImage

plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.title('Image 1')
plt.axis('off')

points1 = []
count1 = 0

def on_click(event):
    global count1
    if event.button == 1 and count1 < 4:
        points1.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        fig.canvas.draw()
        count1 = count1 + 1
        if count1 == 4:
            fig.canvas.mpl_disconnect(cid)
            plt.close() 

fig = plt.gcf()
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

print("选择图一的点的坐标：")
print(points1)


plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.title('Image 2')
plt.axis('off')

points2 = []
count2 = 0

def on_click2(event):
    global count2
    if event.button == 1 and count2 < 4:
        points2.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        fig.canvas.draw()
        count2 = count2 + 1
        if count2 == 4:
            fig.canvas.mpl_disconnect(cid2)
            plt.close()  
fig = plt.gcf()
cid2 = fig.canvas.mpl_connect('button_press_event', on_click2)

plt.show()


print("选择图二的点的坐标：")
for point2 in points2:
    print(point2)


M, mask = cv.findHomography(np.float32(points1).reshape(-1, 1, 2), np.float32(points2).reshape(-1, 1, 2), cv.RANSAC, 5.0)
print("变换矩阵:")
print(M)


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des2,des3,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
 if m.distance < 0.7*n.distance:
    good.append(m)

MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


M2, mask2 = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
print("变换矩阵2:")
print(M2)

fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax1.set_title('Image 1')
ax1.axis('off')

ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
ax2.set_title('Image 2')
ax2.axis('off')

ax3.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
ax3.set_title('Image 3')
ax3.axis('off')

curve_points1 = []
curve_points1_du=[]
curve_points2 = []
curve_point_store=[]
curve_points3 = []
def on_drag(event):
    global curve_points1,curve_points1_du, curve_points2, curve_points3,curve_point_store
    if event.button == 1 and event.inaxes == ax1:
        curve_points1.append((event.xdata, event.ydata))
        curve_points1_du.append((event.xdata, event.ydata))
        ax1.plot(event.xdata, event.ydata, 'ro',markersize=0.1)
        fig.canvas.draw()
        ax1.plot(*zip(*curve_points1), color='r')     
        curve_points2 = cv.perspectiveTransform(np.float32(curve_points1_du).reshape(-1, 1, 2), M)
        latest_curve_points2 = curve_points2[-1]
        if latest_curve_points2[0][0] < 0 or latest_curve_points2[0][0] >= img2.shape[1] or latest_curve_points2[0][1] < 0 or latest_curve_points2[0][1] >= img2.shape[0]:
            curve_point_store.append(curve_points1_du[-1])
            curve_points1_du = curve_points1_du[:-1]
            curve_points2 = curve_points2[:-1]
            curve_points3=cv.perspectiveTransform(np.float32(curve_point_store).reshape(-1, 1, 2), M2@M)
        ax2.plot(*zip(*curve_points2.reshape(-1, 2)), color='r')
        if (len(curve_points3)!=0):
            ax3.plot(*zip(*curve_points3.reshape(-1, 2)), color='r')                
        fig.canvas.draw()
        
def on_release(event):
    if event.button == 1 and event.inaxes == ax1:
        curve_points1.clear()
 
        

cid_drag = fig.canvas.mpl_connect('motion_notify_event', on_drag)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
plt.tight_layout()
plt.show()