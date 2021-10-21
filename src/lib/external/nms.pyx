# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

cdef np.float32_t getSizeAwareIOUThresh(np.float32_t w, np.float32_t h, np.float32_t lowerBound=20.0,
                                        np.float32_t upperBound=120.0, np.float32_t minThresh=0.4, np.float32_t maxThresh=0.7):
    cdef np.float32_t boxSize
    boxSize = (w + h) / 2.0
    if boxSize <= lowerBound:
        return minThresh
    elif boxSize >= upperBound:
        return maxThresh
    return minThresh + (maxThresh - minThresh) * (boxSize - lowerBound) / (upperBound - lowerBound)

def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep

def soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0,
             int opt_sna=0, float sna_threshold=0.8, int opt_sai=0):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov,auxProposalNumber,auxMaxConf, auxMaxX1, auxMaxX2, auxMaxY1, auxMaxY2

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        # apply sai
        if opt_sai:
            Nt = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 30.0, 100.0, 0.45, 0.6)
            #print("ashen_debug Nt is {} and ih is {} and iw is {}", Nt, ty2 - ty1, tx2 - tx1)
            sna_threshold = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 50.0, 80.0, 0.8, 0.9)
            #print("ashen_debug sna_threshold is {} and ih is {} and iw is {}", sna_threshold, ty2 - ty1, tx2 - tx1)

        # vars for sna
        auxMaxConf = 0.0
        auxProposalNumber = 0.0
        auxMaxX1 = tx1
        auxMaxX2 = tx2
        auxMaxY1 = ty1
        auxMaxY2 = ty2

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    if ov >= sna_threshold:
                        auxProposalNumber = auxProposalNumber + 1.0
                        if boxes[pos, 4] > auxMaxConf:
                            auxMaxConf = boxes[pos, 4]
                            auxMaxX1 = boxes[pos, 0]
                            auxMaxY1 = boxes[pos, 1]
                            auxMaxX2 = boxes[pos, 2]
                            auxMaxY2 = boxes[pos, 3]

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
        if opt_sna == 1:
            boxes[i,4] = boxes[i,4] + (1.0 - boxes[i,4]) * (auxProposalNumber / (auxProposalNumber + 1.0)) * auxMaxConf
            boxes[i,0] = (boxes[i,4] * boxes[i, 0] + auxMaxConf * auxMaxX1) / (boxes[i,4] + auxMaxConf)
            boxes[i,1] = (boxes[i,4] * boxes[i, 1] + auxMaxConf * auxMaxY1) / (boxes[i,4] + auxMaxConf)
            boxes[i,2] = (boxes[i,4] * boxes[i, 2] + auxMaxConf * auxMaxX2) / (boxes[i,4] + auxMaxConf)
            boxes[i,3] = (boxes[i,4] * boxes[i, 3] + auxMaxConf * auxMaxY2) / (boxes[i,4] + auxMaxConf)

    keep = [i for i in range(N)]
    return keep

def swap_array_val(np.ndarray[float, ndim=2] arrs, unsigned int idx1, unsigned int idx2):
    cdef unsigned int D = arrs.shape[1]
    cdef float tmpVal
    if idx1 == idx2:
        return
    for i in range(D):
        tmpVal = arrs[idx1, i]
        arrs[idx1, i] = arrs[idx2, i]
        arrs[idx2, i] = tmpVal

def soft_bp_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0,
             int opt_sna=0, float sna_threshold=0.8, int opt_sai=0):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef int maxiter = 2
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov,auxProposalNumber,auxMaxConf, auxMaxX1, auxMaxX2, auxMaxY1, auxMaxY2
    cdef np.ndarray[np.float32_t, ndim=2] posTerms = np.zeros([N, 6], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] negTerms = np.zeros([N, 3], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] iou_thresholds = np.zeros(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] alphas = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] betas = np.ones(10, dtype=np.float32)
    # Suppress mat: [i, j]=m indicates that Det[i] suppressed Det[j] for m times.
    cdef np.ndarray[np.int_t, ndim=2] suppressRecodMat = np.zeros([N, N], dtype=np.int)
    cdef int maxSuppressTime = 1
    cdef int suppressIdx
    iou_thresholds[0] = Nt
    iou_thresholds[1] = 0.75
    iou_thresholds[2] = 0.8
    alphas[0] = 1.0
    betas[0] = 1.0
    alphas[1] = 1.0
    betas[1] = 1.0
    alphas[2] = 1.0
    betas[2] = 1.0

    for iter in range(maxiter):
        posTerms = np.zeros([N, 6], dtype=np.float32)
        negTerms = np.zeros([N, 6], dtype=np.float32)
        for i in range(N):
            tx1 = boxes[i,0]
            ty1 = boxes[i,1]
            tx2 = boxes[i,2]
            ty2 = boxes[i,3]
            ts = boxes[i,4]
            if ts <= threshold:
                continue

            # apply sai
            if opt_sai:
                Nt = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 30.0, 100.0, 0.45, 0.6)
                sna_threshold = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 50.0, 80.0, 0.8, 0.9)

            # vars for sna
            auxMaxConf = 0.0
            auxProposalNumber = 0.0
            auxMaxX1 = tx1
            auxMaxX2 = tx2
            auxMaxY1 = ty1
            auxMaxY2 = ty2

            pos = 0
            # NMS iterations, note that N changes if detection boxes fall below threshold
            for pos in range(N):
                x1 = boxes[pos, 0]
                y1 = boxes[pos, 1]
                x2 = boxes[pos, 2]
                y2 = boxes[pos, 3]
                s = boxes[pos, 4]

                if pos == i or s <= threshold or s > ts or suppressRecodMat[i, pos] >= maxSuppressTime:
                    continue

                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                iw = (min(tx2, x2) - max(tx1, x1) + 1)
                if iw > 0:
                    ih = (min(ty2, y2) - max(ty1, y1) + 1)
                    if ih > 0:
                        ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                        ov = iw * ih / ua #iou between max box and detection box

                        if ov > iou_thresholds[iter]:
                            #if ov > negTerms[pos, 0]:
                            #    negTerms[pos, 0] = ov
                            #    negTerms[pos, 2] = (float)(i)
                            if ts > negTerms[pos, 1]:
                                negTerms[pos, 0] = ov
                                negTerms[pos, 1] = ts
                                negTerms[pos, 2] = (float)(i)

                        if ov >= sna_threshold:
                            auxProposalNumber = auxProposalNumber + 1.0
                            if boxes[pos, 4] > auxMaxConf:
                                auxMaxConf = boxes[pos, 4]
                                auxMaxX1 = boxes[pos, 0]
                                auxMaxY1 = boxes[pos, 1]
                                auxMaxX2 = boxes[pos, 2]
                                auxMaxY2 = boxes[pos, 3]

            if opt_sna == 1:
                posTerms[i, 0] = auxProposalNumber
                posTerms[i, 1] = auxMaxConf
                posTerms[i, 2] = auxMaxX1
                posTerms[i, 3] = auxMaxY1
                posTerms[i, 4] = auxMaxX2
                posTerms[i, 5] = auxMaxY2

        for i in range(N):
            if boxes[i, 4] <= threshold:
                continue
            boxes[i, 4] = min(1.0, boxes[i, 4] + alphas[iter] * (1.0 - boxes[i,4]) * (posTerms[i, 0] / (posTerms[i, 0] + 1.0)) * posTerms[i, 1])
            boxes[i, 0] = (boxes[i, 4] * boxes[i, 0] + posTerms[i, 1] * posTerms[i, 2]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 1] = (boxes[i, 4] * boxes[i, 1] + posTerms[i, 1] * posTerms[i, 3]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 2] = (boxes[i, 4] * boxes[i, 2] + posTerms[i, 1] * posTerms[i, 4]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 3] = (boxes[i, 4] * boxes[i, 3] + posTerms[i, 1] * posTerms[i, 5]) / (boxes[i, 4] + posTerms[i, 1])
            if negTerms[i, 0] > 0.01:
                boxes[i, 4] = boxes[i, 4] - betas[iter] * boxes[i, 4] * negTerms[i, 0]
                suppressIdx = (int)(negTerms[i, 2])
                suppressRecodMat[suppressIdx, i] = suppressRecodMat[suppressIdx, i] + 1

    for i in range(N):
        if boxes[i, 4] < threshold:
            swap_array_val(boxes, i, N-1)
            swap_array_val(negTerms, i, N-1)
            swap_array_val(posTerms, i, N-1)
            N = N - 1

    keep = [i for i in range(N)]
    return keep

def soft_bp_nms_v2(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.01, unsigned int method=0,
             int opt_sna=0, float sna_threshold=0.8, int opt_sai=0):
    cdef unsigned int N = boxes.shape[0]

    # Pre-calculate area sizes.
    cdef np.ndarray[np.float32_t, ndim=1] areas

    cdef float iw, ih, box_area
    cdef float ua, inter
    cdef int pos = 0
    cdef int maxiter = 2
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov,auxProposalNumber,auxMaxConf, auxMaxX1, auxMaxX2, auxMaxY1, auxMaxY2
    cdef np.ndarray[np.float32_t, ndim=2] posTerms = np.zeros([N, 6], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] negTerms = np.zeros([N, 3], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] iou_thresholds = np.zeros(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] alphas = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] betas = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] m_w1 = np.ones(10, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] m_w2 = np.ones(10, dtype=np.float32)
    # Suppress mat: [i, j]=m indicates that Det[i] suppressed Det[j] for m times.
    cdef np.ndarray[np.int_t, ndim=2] suppressRecodMat = np.zeros([N, N], dtype=np.int)
    cdef int maxSuppressTime = 1
    cdef float momentum = 0.0
    cdef int suppressIdx
    iou_thresholds[0] = Nt
    iou_thresholds[1] = 0.75
    iou_thresholds[2] = 0.8
    alphas[0] = 1.0
    betas[0] = 1.0
    alphas[1] = 1.0
    betas[1] = 1.0
    alphas[2] = 1.0
    betas[2] = 1.0
    m_w1[0] = 1.0
    m_w2[0] = 0.0
    m_w1[1] = 0.0
    m_w2[1] = 1.0

    for iter in range(maxiter):
        posTerms = np.zeros([N, 6], dtype=np.float32)
        negTerms = np.zeros([N, 6], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        for i in range(N):
            tx1 = boxes[i,0]
            ty1 = boxes[i,1]
            tx2 = boxes[i,2]
            ty2 = boxes[i,3]
            ts = boxes[i,4]
            tarea = areas[i]
            if ts <= threshold:
                continue

            # apply sai
            if opt_sai:
                Nt = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 30.0, 100.0, 0.45, 0.6)
                sna_threshold = getSizeAwareIOUThresh(tx2 - tx1, ty2 - ty1, 50.0, 80.0, 0.8, 0.9)

            # vars for sna
            auxMaxConf = 0.0
            auxProposalNumber = 0.0
            auxMaxX1 = tx1
            auxMaxX2 = tx2
            auxMaxY1 = ty1
            auxMaxY2 = ty2

            pos = 0
            # NMS iterations, note that N changes if detection boxes fall below threshold
            for pos in range(N):
                x1 = boxes[pos, 0]
                y1 = boxes[pos, 1]
                x2 = boxes[pos, 2]
                y2 = boxes[pos, 3]
                s = boxes[pos, 4]
                area = areas[pos]


                if pos == i or s <= threshold or s > ts:
                    continue

                # area = (x2 - x1 + 1) * (y2 - y1 + 1)
                iw = (min(tx2, x2) - max(tx1, x1) + 1)
                if iw > 0:
                    ih = (min(ty2, y2) - max(ty1, y1) + 1)
                    if ih > 0:
                        inter = iw * ih
                        ua = float(tarea + area - inter)
                        ov = inter / ua #iou between max box and detection box

                        if ov > iou_thresholds[iter] and suppressRecodMat[i, pos] < maxSuppressTime:
                            #if ov > negTerms[pos, 0]:
                            #    negTerms[pos, 0] = ov
                            #    negTerms[pos, 2] = (float)(i)
                            momentum = m_w1[iter] * (ts / s) + m_w2[iter] * (ov / iou_thresholds[iter])
                            if momentum > negTerms[pos, 1]:
                                negTerms[pos, 0] = ov
                                negTerms[pos, 1] = momentum
                                negTerms[pos, 2] = (float)(i)

                        if ov >= sna_threshold:
                            auxProposalNumber = auxProposalNumber + 1.0
                            if boxes[pos, 4] > auxMaxConf:
                                auxMaxConf = boxes[pos, 4]
                                auxMaxX1 = boxes[pos, 0]
                                auxMaxY1 = boxes[pos, 1]
                                auxMaxX2 = boxes[pos, 2]
                                auxMaxY2 = boxes[pos, 3]

            if opt_sna == 1:
                posTerms[i, 0] = auxProposalNumber
                posTerms[i, 1] = auxMaxConf
                posTerms[i, 2] = auxMaxX1
                posTerms[i, 3] = auxMaxY1
                posTerms[i, 4] = auxMaxX2
                posTerms[i, 5] = auxMaxY2

        for i in range(N):
            if boxes[i, 4] <= threshold:
                continue
            boxes[i, 4] = min(1.0, boxes[i, 4] + alphas[iter] * (1.0 - boxes[i,4]) * (posTerms[i, 0] / (posTerms[i, 0] + 1.0)) * posTerms[i, 1])
            boxes[i, 0] = (boxes[i, 4] * boxes[i, 0] + posTerms[i, 1] * posTerms[i, 2]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 1] = (boxes[i, 4] * boxes[i, 1] + posTerms[i, 1] * posTerms[i, 3]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 2] = (boxes[i, 4] * boxes[i, 2] + posTerms[i, 1] * posTerms[i, 4]) / (boxes[i, 4] + posTerms[i, 1])
            boxes[i, 3] = (boxes[i, 4] * boxes[i, 3] + posTerms[i, 1] * posTerms[i, 5]) / (boxes[i, 4] + posTerms[i, 1])
            if negTerms[i, 0] > 0.01:
                boxes[i, 4] = boxes[i, 4] - betas[iter] * boxes[i, 4] * negTerms[i, 0]
                suppressIdx = (int)(negTerms[i, 2])
                suppressRecodMat[suppressIdx, i] = suppressRecodMat[suppressIdx, i] + 1

    for i in range(N):
        if boxes[i, 4] < threshold:
            swap_array_val(boxes, i, N-1)
            swap_array_val(negTerms, i, N-1)
            swap_array_val(posTerms, i, N-1)
            N = N - 1

    keep = [i for i in range(N)]
    return keep

def soft_nms_39(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
    cdef float tmp

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        for j in range(5, 39):
            tmp = boxes[i, j]
            boxes[i, j] = boxes[maxpos, j]
            boxes[maxpos, j] = tmp

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        for j in range(5, 39):
                            tmp = boxes[pos, j]
                            boxes[pos, j] = boxes[N - 1, j]
                            boxes[N - 1, j] = tmp
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

def soft_nms_merge(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0, float weight_exp=6):
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
    cdef float mx1,mx2,my1,my2,mts,mbs,mw

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        mx1 = boxes[i, 0] * boxes[i, 5]
        my1 = boxes[i, 1] * boxes[i, 5]
        mx2 = boxes[i, 2] * boxes[i, 6]
        my2 = boxes[i, 3] * boxes[i, 6]
        mts = boxes[i, 5]
        mbs = boxes[i, 6]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    mw  = (1 - weight) ** weight_exp
                    mx1 = mx1 + boxes[pos, 0] * boxes[pos, 5] * mw
                    my1 = my1 + boxes[pos, 1] * boxes[pos, 5] * mw
                    mx2 = mx2 + boxes[pos, 2] * boxes[pos, 6] * mw
                    my2 = my2 + boxes[pos, 3] * boxes[pos, 6] * mw
                    mts = mts + boxes[pos, 5] * mw
                    mbs = mbs + boxes[pos, 6] * mw

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

        boxes[i, 0] = mx1 / mts
        boxes[i, 1] = my1 / mts
        boxes[i, 2] = mx2 / mbs
        boxes[i, 3] = my2 / mbs

    keep = [i for i in range(N)]
    return keep
