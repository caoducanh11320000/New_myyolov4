#include <iostream>
#include "trt_inference.h"
// #include "yololayer.h" // thu ko include xem co bao loi ko

// #define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
// #define NMS_THRESH 0.4
// #define BBOX_CONF_THRESH 0.5
// #define BATCH_SIZE 1

using namespace IMXAIEngine;
using namespace nvinfer1;

// // do the thay doi sau
// static const int INPUT_H = Yolo::INPUT_H;
// static const int INPUT_W = Yolo::INPUT_W;
// static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
// static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
// const char* INPUT_BLOB_NAME = "data";
// const char* OUTPUT_BLOB_NAME = "prob";
// static Logger gLogger;

TRT_Inference test1;
std::vector<std::string> file_image;

int main(int argc, char** argv){

    cudaSetDevice(DEVICE);

    if (argc == 2 && std::string(argv[1]) == "-s") {
        // co the goi ham API model o day
        trt_error error1=  test1.trt_APIModel();     
    } 
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        // goi ham init
        test1.init_inference(argv[2],file_image);
    } 
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    /// thuc hien ham do_Inference o day
    std::string folder= std::string(argv[2]);
    test1.trt_detection(folder, file_image);

}