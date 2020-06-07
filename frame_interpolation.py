import cv2
import sys
import numpy as np
import pickle
import numpy as np
import os

BLUR_OCC = 3

def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    holes=None
    new_holes = np.zeros((flow.shape[0], flow.shape[1]))
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            val = flow[i][j]
            if val[0] > np.power(10,9) or val[1] > np.power(10,9):
                new_holes[i][j] = 0
            elif np.isnan(val[0]) or np.isnan(val[1]):
                new_holes[i][j] = 0
            elif np.isinf(val[0]) or np.isinf(val[1]):
                new_holes[i][j] = 0 
            else:
                new_holes[i][j] = 1
    holes = new_holes
    return holes

def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h,w,_ = flow.shape 
    has_hole=1
    while has_hole==1:
        foo = 1
        # ===== loop all pixel in x, then in y
        for y in range(0, h):
            for x in range(0,w):
                avg_u = 0
                avg_v = 0
                good_n = 0
                if (y == 0 and x == 0 and holes[y][x] == 0):   
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1				
                    if (holes[y+1][x+1] == 1):
                        avg_u = avg_u + flow[y+1][x+1][0]
                        avg_v = avg_v + flow[y+1][x+1][1]
                        good_n += 1
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v

                avg_u = 0
                avg_v = 0
                good_n = 0 
                if (y == 0 and x == w-1 and holes[y][x] == 0):   #index 0,w-1
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1				
                    if (holes[y+1][x-1] == 1):
                        avg_u = avg_u + flow[y+1][x-1][0]
                        avg_v = avg_v + flow[y+1][x-1][1]
                        good_n += 1
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v
                
                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y == h-1 and x == 0 and holes[y][x] == 0):   #index h-1,0
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1				
                    if (holes[y-1][x+1] == 1):
                        avg_u = avg_u + flow[y-1][x+1][0]
                        avg_v = avg_v + flow[y-1][x+1][1]
                        good_n += 1
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v
                
                avg_u = 0
                avg_v = 0
                good_n = 0 
                if (y == h-1 and x == w-1 and holes[y][x] == 0):   #index h-1,w-1
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1				
                    if (holes[y-1][x-1] == 1):
                        avg_u = avg_u + flow[y-1][x-1][0]
                        avg_v = avg_v + flow[y-1][x-1][1]
                        good_n += 1
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v
			
                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y == 0 and x > 0 and x < w-1 and holes[y][x] == 0):   #top row
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y+1][x-1] == 1):
                        avg_u = avg_u + flow[y+1][x-1][0]
                        avg_v = avg_v + flow[y+1][x-1][1]
                        good_n += 1				
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1
                    if (holes[y+1][x+1] == 1):
                        avg_u = avg_u + flow[y+1][x+1][0]
                        avg_v = avg_v + flow[y+1][x+1][1]
                        good_n += 1
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1				
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v
			
                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y == h-1 and x > 0 and x < w-1 and holes[y][x] == 0):   #bottom row
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y-1][x-1] == 1):
                        avg_u = avg_u + flow[y-1][x-1][0]
                        avg_v = avg_v + flow[y-1][x-1][1]
                        good_n += 1				
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1
                    if (holes[y-1][x+1] == 1):
                        avg_u = avg_u + flow[y-1][x+1][0]
                        avg_v = avg_v + flow[y-1][x+1][1]
                        good_n += 1
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1				
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v
                
                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y > 0 and y < h-1 and x == 0 and holes[y][x] == 0):   #leftmost column
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1
                    if (holes[y-1][x+1] == 1):
                        avg_u = avg_u + flow[y-1][x+1][0]
                        avg_v = avg_v + flow[y-1][x+1][1]
                        good_n += 1				
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1
                    if (holes[y+1][x+1] == 1):
                        avg_u = avg_u + flow[y+1][x+1][0]
                        avg_v = avg_v + flow[y+1][x+1][1]
                        good_n += 1
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1				
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v

                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y > 0 and y < h-1 and x == w-1 and holes[y][x] == 0):   #rightmost column
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1
                    if (holes[y-1][x-1] == 1):
                        avg_u = avg_u + flow[y-1][x-1][0]
                        avg_v = avg_v + flow[y-1][x-1][1]
                        good_n += 1				
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y+1][x-1] == 1):
                        avg_u = avg_u + flow[y+1][x-1][0]
                        avg_v = avg_v + flow[y+1][x-1][1]
                        good_n += 1
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1				
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v

                avg_u = 0
                avg_v = 0
                good_n = 0  
                if (y != 0 and y != h-1 and x != 0 and x != w-1 and holes[y][x] == 0):   
                    if (holes[y-1][x] == 1):
                        avg_u = avg_u + flow[y-1][x][0]
                        avg_v = avg_v + flow[y-1][x][1]
                        good_n += 1
                    if (holes[y-1][x-1] == 1):
                        avg_u = avg_u + flow[y-1][x-1][0]
                        avg_v = avg_v + flow[y-1][x-1][1]
                        good_n += 1				
                    if (holes[y][x-1] == 1):
                        avg_u = avg_u + flow[y][x-1][0]
                        avg_v = avg_v + flow[y][x-1][1]
                        good_n += 1
                    if (holes[y+1][x-1] == 1):
                        avg_u = avg_u + flow[y+1][x-1][0]
                        avg_v = avg_v + flow[y+1][x-1][1]
                        good_n += 1
                    if (holes[y+1][x] == 1):
                        avg_u = avg_u + flow[y+1][x][0]
                        avg_v = avg_v + flow[y+1][x][1]
                        good_n += 1				
                    if (holes[y+1][x+1] == 1):
                        avg_u = avg_u + flow[y+1][x+1][0]
                        avg_v = avg_v + flow[y+1][x+1][1]
                        good_n += 1
                    if (holes[y][x+1] == 1):
                        avg_u = avg_u + flow[y][x+1][0]
                        avg_v = avg_v + flow[y][x+1][1]
                        good_n += 1
                    if (holes[y-1][x+1] == 1):
                        avg_u = avg_u + flow[y-1][x+1][0]
                        avg_v = avg_v + flow[y-1][x+1][1]
                        good_n += 1				
                    if (good_n > 0):
                        avg_u = avg_u / good_n
                        avg_v = avg_v / good_n
                        foo = 0
                        holes[y][x] = 1
                        flow[y][x][0] = avg_u
                        flow[y][x][1] = avg_v

        if (foo == 1):
            has_hole = 0

    return flow

def myroundfunc(num):
    n = int(num)
    if num - n >= 0.5:
        return n + 1
    else:
        return n

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    height,width,_ = flow0.shape
    occ0 = np.zeros([height,width],dtype=np.float32)
    occ1 = np.zeros([height,width],dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4',diff)

    for y in range(1):
        for x in range(width):
            if flow1[y][x][0] != flow1_step4[y][x][0] or flow1[y][x][1] != flow1_step4[y][x][1]:
                print(y,x)
                print([flow1[y][x], flow1_step4[y][x]])

    # ==================================================
    # ===== main part of step 5
    # ==================================================
    for y in range(0,height):         
        for x in range(0,width):
            locu = flow0[y][x][0] + x
            locv = flow0[y][x][1] + y

            if np.isnan(flow1[y][x][0]) or np.isnan(flow1[y][x][1]) or flow1[y][x][0] >= 1e10 or flow1[y][x][1] >= 1e10 or np.isinf(flow1[y][x][0]) or np.isinf(flow1[y][x][1]):
                occ0[y][x] = 1

            if (locv <= -1.5 or locv >= height - 0.5 or locu <= -1.5 or  locu >= width -0.5):  #if out of bounds
                occ1[y][x] = 1
            else:                                               
                y1 = int(locv)
                x1 = int(locu)
            
                if (locv < 0):
                    y1 = 0
                if locu < 0:
                    x1 = 0
                if locv >= 0:
                    y1 = myroundfunc(locv)
                if locu >= 0:
                    x1 = myroundfunc(locu)
                if (locv >= height):
                    y1 = height-1
                if (locu >= width):
                    x1 = width-1
                if np.sum(np.abs(flow0[y][x] - flow1[y1][x1])) > 0.5:
                    occ1[y][x] = 1
    return occ0,occ1

def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    iflow = None
    iflow = np.zeros_like(flow)

    l = []
    for y in range(iflow.shape[0]):
        l.append([])
        for x in range(iflow.shape[1]):
            l[y].append([])

    for y in range(iflow.shape[0]):
        for x in range(iflow.shape[1]):
            u = flow[y][x][0]
            v = flow[y][x][1]

            u1 = u + 0.5
            v1 = v + 0.5

            u2 = u - 0.5
            v2 = v - 0.5

            u3 = u + 0.5
            v3 = v - 0.5

            u4 = u - 0.5
            v4 = v + 0.5

            u5 = u + 0.5
            v5 = v 

            u6 = u - 0.5
            v6 = v 

            u7 = u 
            v7 = v + 0.5

            u8 = u
            v8 = v - 0.5

            x1 = int(myroundfunc(x + t * u1))
            y1 = int(myroundfunc(y + t * v1))

            x2 = int(myroundfunc(x + t * u2))
            y2 = int(myroundfunc(y + t * v2))

            x3 = int(myroundfunc(x + t * u3))
            y3 = int(myroundfunc(y + t * v3))

            x4 = int(myroundfunc(x + t * u4))
            y4 = int(myroundfunc(y + t * v4))

            x5 = int(myroundfunc(x + t * u5))
            y5 = int(myroundfunc(y + t * v5))

            x6 = int(myroundfunc(x + t * u6))
            y6 = int(myroundfunc(y + t * v6))

            x7 = int(myroundfunc(x + t * u7))
            y7 = int(myroundfunc(y + t * v7))

            x8 = int(myroundfunc(x + t * u8))
            y8 = int(myroundfunc(y + t * v8))

            if (y1 >= 0 and y1 < frame1.shape[0] and x1 >= 0 and x1 < frame1.shape[1]):
                l[y1][x1].append([y,x])

            if (y2 >= 0 and y2 < frame1.shape[0] and x2 >= 0 and x2 < frame1.shape[1]):
                l[y2][x2].append([y,x])

            if (y3 >= 0 and y3 < frame1.shape[0] and x3 >= 0 and x3 < frame1.shape[1]):
                l[y3][x3].append([y,x])

            if (y4 >= 0 and y4 < frame1.shape[0] and x4 >= 0 and x4 < frame1.shape[1]):
                l[y4][x4].append([y,x])

            if (y5 >= 0 and y5 < frame1.shape[0] and x5 >= 0 and x5 < frame1.shape[1]):
                l[y5][x5].append([y,x])
    
            if (y6 >= 0 and y6 < frame1.shape[0] and x6 >= 0 and x6 < frame1.shape[1]):
                l[y6][x6].append([y,x])

            if (y7 >= 0 and y7 < frame1.shape[0] and x7 >= 0 and x7 < frame1.shape[1]):
                l[y7][x7].append([y,x])

            if (y8 >= 0 and y8 < frame1.shape[0] and x8 >= 0 and x8 < frame1.shape[1]):
                l[y8][x8].append([y,x])

    for y1 in range(len(l)):
        for x1 in range(len(l[0])):
            listpoints = l[y1][x1]
            if y1 == 0 and x1 == 33:
                print("0,33",listpoints)
                for point in listpoints:
                    print(frame0[point[0]][point[1]],frame1[y][x])
                    print(flow[point[0]][point[1]])
          
            if len(listpoints) > 0:
                if listpoints.count(listpoints[0]) == len(listpoints):
                    iflow[y1][x1] = flow[listpoints[0][0]][listpoints[0][1]]
                else:
                    lr = 1e70
                    lg = 1e70
                    lb = 1e70
                    lowest = np.array([lr,lg,lb])
                    low_point = []

                    for point in listpoints:
                        rgb = np.sum(np.absolute(frame0[point[0]][point[1]] - frame1[y1][x1]))
                        rgbarray = frame0[point[0]][point[1]] - frame1[y1][x1]
                        
                        if rgb < np.sum(np.absolute(lowest)):
                            low_point = point

                    iflow[y1][x1] = flow[low_point[0]][low_point[1]]
            else:
                iflow[y1][x1] = 1e10

    if iflow is None:
        print("NONE")
    return iflow

def bilinear(frame, x, y):
    i = int(np.floor(x))
    j = int(np.floor(y))
    a = x - i
    b = y - j

    if i >= frame.shape[1]-1 or j >= frame.shape[0]-1:
        res = frame[frame.shape[0]-1][frame.shape[1]-1]
    else:
        if (j >=0 and i >=0):
            res  = ((1-a)*(1-b))*(frame[j,i])+(a*(1-b))*(frame[j,i+1])+(a*b)*(frame[j+1,i+1])+((1-a)*b)*(frame[j+1,i])
        else:
            res = 0
    return res

def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''
    iframe = np.zeros_like(frame0).astype(np.float32)
    h,w,_ = iframe.shape
    for y in range(h):
        for x in range(w):
            occ0[y][x]  = np.round(occ0[y][x])
            occ1[y][x]  = np.round(occ1[y][x])

    for y in range(h):
        for x in range(w):
            locu0 = x - t * iflow[y][x][0]
            locv0 = y - t * iflow[y][x][1]
            locu1 = x + (1-t) * iflow[y][x][0] 
            locv1 = y + (1-t) * iflow[y][x][1]

            y0 = int(np.round(locv0))
            x0 = int(np.round(locu0))
            y1 = int(np.round(locv1))
            x1 = int(np.round(locu1))

            b0 = bilinear(frame0, locu0, locv0)
            b1 = bilinear(frame1, locu1, locv1)

            if (y0 < 0 or y0 >= frame0.shape[0] or x0 < 0 or x0 >= frame0.shape[1]):
                iframe[y][x] = b1
                continue
            
            if(y1 < 0 or y1 >= frame1.shape[0] or x1 < 0 or x1 >= frame1.shape[1]):
                iframe[y][x] = b0
                continue 
                
            if occ0[y0][x0] == 0 and occ1[y1][x1] == 0:     #blend
                iframe[y][x] = (1 - t) * frame0[y0][x0] + t * frame1[y1][x1]

            elif occ0[y0][x0] == 1 and  occ1[y1][x1] == 0:
                iframe[y][x] = b1

            elif occ1[y1][x1] == 1 and  occ0[y0][x0] == 0:
                iframe[y][x] = b0

            else:
                iframe[y][x] = b1
    return iframe

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    im = cv2.GaussianBlur(im, (5,5),0)
    return im

def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) 
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb'))
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) 
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t

if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]
    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)
    
    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))

