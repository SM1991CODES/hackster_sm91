from ctypes import *
import cv2
import numpy as np
import runner
import os
import xir.graph
import pathlib
import xir.subgraph
import threading
import time
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def preprocess_fn(image_path):
    '''
    Image pre-processing.
    Rearranges from BGR to RGB then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    # image = cv2.imread(image_path)
    image = np.load(image_path)
    print(image.shape)

    test_x = image[0:64, 0:256, 3]  # get only depth channel
    plt.imshow(test_x)
    plt.imsave("depth.png", test_x)
    test_x /= np.max(test_x)
    
    return test_x

def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children
            if s.metadata.get_attr_str ("device") == "DPU"]
    return sub 


def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    outputHeight = outputTensors[0].dims[1]
    outputWidth = outputTensors[0].dims[2]
    outputChannel = outputTensors[0].dims[3]
    outputSize = outputHeight*outputWidth*outputChannel

    batchSize = inputTensors[0].dims[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        outputData.append(np.empty((runSize,outputHeight,outputWidth,outputChannel), dtype = np.float32, order = 'C'))
        inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j,...] = img[(count+j)% n_of_images].reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        # get predictions direclty here
        predictions = outputData[0][0]
        predictions = softmax(predictions, -1)  # softmax along channels
        print("predictions shape: ", predictions.shape)
        mask = np.argmax(predictions, -1)  # along channels
        print("mask shape: ", mask.shape)

        img_test = np.squeeze(imageRun[0], -1)
        plt.imshow(img_test)
        plt.imsave("depth_"+str(count)+'.png', img_test)
        # mask = mask[0]
        plt.imshow(mask)
        plt.imsave("keypoints_"+str(count)+".png", mask)


        # for j in range(len(outputData)):
        #     outputData[j] = outputData[j].reshape(runSize, outputSize)

        '''store output vectors '''
        # for j in range(runSize):
        #     out_q[write_index] = outputData[0][j]
        #     write_index += 1
        
        count = count + runSize
        


def app(image_dir,threads,model):

    listimage=os.listdir(image_dir)
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.graph.Graph.deserialize(pathlib.Path(model))
    subgraphs = get_subgraph (g)
    assert len(subgraphs) == 1 # only one DPU kernel
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(runner.Runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print('Pre-processing',runTotal,'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir,listimage[i])
        img.append(preprocess_fn(path))

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps,runTotal, timetotal))


    ''' post-processing '''
    # classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']  
    # correct = 0
    # wrong = 0
    # print('output buffer length:',len(out_q))
    # for i in range(len(out_q)):
    #     argmax = np.argmax((out_q[i]))
    #     prediction = classes[argmax]
    #     ground_truth, _ = listimage[i].split('_')
    #     if (ground_truth==prediction):
    #         correct += 1
    #     else:
    #         wrong += 1
    # accuracy = correct/len(out_q)
    # print('Correct:',correct,'Wrong:',wrong,'Accuracy:', accuracy)


if __name__ == "__main__":
    print("hello to Xilinx app development using DPU")
    os.system('dexplorer -w')

    app("/home/root/Vitis-AI/dex_read", 1, "/home/root/Vitis-AI/Custom_models/dpu_u3d_kp.elf")
