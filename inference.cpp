#include "inference.h"
#include "utils.h"


std::vector<std::string> objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    //std::cout << "object names loaded \n";
    return file_lines;
}




YOLOPredictor::YOLOPredictor(const std::string& modelPath,
    const bool& isGPU,
    float confThreshold,
    float iouThreshold,
    float maskThreshold)
{
    this->confThreshold = confThreshold;
    this->iouThreshold = iouThreshold;
    this->maskThreshold = maskThreshold;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLOV8");
    sessionOptions = Ort::SessionOptions();
    wchar_t path[1024];
    mbstowcs_s(nullptr, path, 1024, modelPath.c_str(), 1024);
    session = Ort::Session(env, path, sessionOptions);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> net_input_shape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    input_shape.assign(net_input_shape.begin(), net_input_shape.end());
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    std::vector<int64_t> net_output_shape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    output_shape.assign(net_output_shape.begin(), net_output_shape.end());
    
    Ort::AllocatedStringPtr in_name_allocator = session.GetInputNameAllocated(0, allocator);
    input_name = in_name_allocator.get();
    Ort::AllocatedStringPtr out_name_allocator = session.GetOutputNameAllocated(0, allocator);
    output_name = out_name_allocator.get();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    //bool isGPU = false;
    //std::cout << "   isGPU   " << isGPU << std::endl;
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: CUDA GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    sessionOptions.SetIntraOpNumThreads(1);
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);


    
}

void YOLOPredictor::getBestClassInfo(std::vector<float>::iterator it,
    float& bestConf,
    int& bestClassId,
    const int _classNums)
{
    // first 4 element are box
    bestClassId = 4;
    bestConf = 0;

    for (int i = 4; i < _classNums + 4; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 4;
        }
    }
}


void YOLOPredictor::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, cv::Size((int)this->input_shape[3], (int)this->input_shape[2]),
        cv::Scalar(114, 114, 114), false,
        false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}



// ----------------------------   NMS  --------------------------
static float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
    int xx1, yy1, xx2, yy2;
    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}


typedef struct {
    cv::Rect box;
    float confidence;
    int index;
}BBOX;


bool compScore(BBOX a, BBOX b)
{
    if (a.confidence > b.confidence)
        return true;
    else
        return false;
}
//input:  boxes: 原始检测框集合;
//input:  confidences：原始检测框对应的置信度值集合
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output:  indices  经过上面两个阈值过滤后剩下的检测框的index
void nms_boxes(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float confThreshold, float nmsThreshold, std::vector<int>& indices)
{
    BBOX bbox;
    std::vector<BBOX> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++) {
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), compScore);

    int updated_size = bboxes.size();
    for (i = 0; i < updated_size; i++) {
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (j = i + 1; j < updated_size; j++) {
            float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold) {
                bboxes.erase(bboxes.begin() + j);
                updated_size = bboxes.size();
            }
        }
    }
}

//  ---------------------------------------------------------------


std::vector<YoloResult> YOLOPredictor::postprocessing(const cv::Size& resizedImageShape,
    const cv::Size& originalImageShape,
    std::vector<Ort::Value>& outputTensors)
{

    // for box
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float* boxOutput = outputTensors[0].GetTensorMutableData<float>();
    //[1,4+n,8400]=>[1,8400,4+n] or [1,4+n+32,8400]=>[1,8400,4+n+32]
    cv::Mat output0 = cv::Mat(cv::Size((int)this->output_shape[2], (int)this->output_shape[1]), CV_32F, boxOutput).t();
    float* output0ptr = (float*)output0.data;
    int rows = (int)this->output_shape[2];
    int cols = (int)this->output_shape[1];
    //std::cout << rows << cols << std::endl;

    //std::vector<std::vector<float>> picked_proposals;
    //cv::Mat mask_protos;

    for (int i = 0; i < rows; i++)
    {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold)
        {
           
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    //cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);
    nms_boxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);
 
    std::vector<YoloResult> results;
    for (int idx : indices)
    {
        YoloResult res;
        res.box = cv::Rect(boxes[idx]);
       
        res.boxMask = cv::Mat::zeros((int)this->input_shape[2], (int)this->input_shape[3], CV_8U);

        utils::scaleCoords(res.box, res.boxMask, this->maskThreshold, resizedImageShape, originalImageShape);
        res.conf = confs[idx];
        res.classId = classIds[idx];
        results.emplace_back(res);
    }

    return results;
}

std::vector<YoloResult> YOLOPredictor::predict(cv::Mat& image)
{
    float* blob = nullptr;
    this->preprocessing(image, blob, input_shape);

    size_t inputTensorSize = utils::vectorProduct(input_shape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        input_shape.data(), input_shape.size()));
    
    const char* input_name_ptr = input_name.c_str();
    const char* output_name_ptr = output_name.c_str();
    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{ nullptr },
        & input_name_ptr, inputTensors.data(), 1, & output_name_ptr, 1);
        
    cv::Size resizedShape = cv::Size((int)input_shape[3], (int)input_shape[2]);
    std::vector<YoloResult> result = this->postprocessing(resizedShape,
        image.size(),
        outputTensors);

    delete[] blob;

    return result;
}
