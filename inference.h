
#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>

struct YoloResult
{
    cv::Rect box;
    cv::Mat boxMask; 
    float conf{};
    int classId;
};


class YOLOPredictor
{
public:
    explicit YOLOPredictor(std::nullptr_t) {};
    YOLOPredictor(const std::string& modelPath,
        const bool& isGPU,
        float confThreshold,
        float iouThreshold,
        float maskThreshold);
    // ~YOLOPredictor();
    std::vector<YoloResult> predict(cv::Mat& image);
    int classNums = 80;

private:
    Ort::Env env{ nullptr };
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };
    Ort::AllocatorWithDefaultOptions allocator;

    void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<YoloResult> postprocessing(const cv::Size& resizedImageShape,
        const cv::Size& originalImageShape,
        std::vector<Ort::Value>& outputTensors);

    static void getBestClassInfo(std::vector<float>::iterator it,
        float& bestConf,
        int& bestClassId,
        const int _classNums);
    
    //bool isDynamicInputShape{};

    //std::vector<const char*> inputNames;
    //std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::string input_name;
    std::string output_name;
    //std::vector<const char*> outputNames;
    //std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    //    std::vector<const char*> input_node_names;
    //    std::vector<const char*> output_node_names;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    //std::vector<std::vector<int64_t>> outputShapes;

    float confThreshold = 0.3f;
    float iouThreshold = 0.4f;

    bool hasMask = false;
    float maskThreshold = 0.5f;
};
