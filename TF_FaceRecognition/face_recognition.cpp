#include <fstream>
#include <utility>
#include <vector>
#include <Windows.h>
#include <highgui.hpp>
#include <imgproc.hpp>

#include "math_functions.h"

#include "tensorflow/core/public/session.h"


// 读取图并加载参数
tensorflow::Status LoadGraph(const tensorflow::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
	}
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	tensorflow::Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return tensorflow::Status::OK();
}

//将opencv的Mat数组填充到Tenflow的tensor中
tensorflow::Tensor convertMatToTensor(std::vector<cv::Mat> images) {
	int width = images[0].cols;
	int height = images[0].rows;
	int channel = images[0].channels();
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ (int64)images.size(),height, width, channel }));
	auto input_tensor_map = input_tensor.tensor<float, 4>();

	const float* source_row;
	const float* source_pixel;
	const float* source_value;

	for (int i = 0; i < images.size(); i++) {
		for (int y = 0; y < height; y++) {
			source_row = (float*)images[i].data + (y * width * channel);
			for (int x = 0; x < width; x++) {
				source_pixel = source_row + (x * channel);
				for (int c = 0; c < channel; c++) {
					source_value = source_pixel + c;
					input_tensor_map(i, y, x, c) = *source_value;
				}
			}
		}
	}

	return input_tensor;
}

int main(int argc, char* argv[]) {
	tensorflow::string image_path0 = "Aaron_Eckhart_0001.png";
	tensorflow::string image_path1 = "Lili_Taylor_0002.png";
	tensorflow::string graph = "../model/tensorflow/20180402-114759.pb";	//图与参数模型
	tensorflow::string input_layer = "input:0";	//输入层
	tensorflow::string output_layer = "embeddings:0";	//输出层

	//读取两张人脸图像
	cv::Mat image0 = cv::imread(image_path0);
	cv::Mat image1 = cv::imread(image_path1);

	//本模型输入图像尺寸为160x160，因此需要将图像resize到160x160
	cv::resize(image0, image0, cv::Size(160, 160));
	cv::resize(image1, image1, cv::Size(160, 160));

	//创建两个float型Mat图像
	cv::Mat image_input0(image0.rows, image0.cols, CV_32F);
	cv::Mat image_input1(image1.rows, image1.cols, CV_32F);

	//将两张人脸图像从BGR转换成RGB
	cv::cvtColor(image0, image0, CV_BGR2RGB);
	cv::cvtColor(image1, image1, CV_BGR2RGB);

	//将两张人脸图像从uchar转换成float,并将像素值从0——255归一化到-0.5——0.5
	image0.convertTo(image_input0, CV_32F, 1 / 255.0, -0.5);
	image1.convertTo(image_input1, CV_32F, 1 / 255.0, -0.5);

	//将两张图像赋值给Mat向量
	std::vector<cv::Mat> images;
	images.push_back(image_input0);
	images.push_back(image_input1);

	// 创建tensorflow session并加载模型
	std::unique_ptr<tensorflow::Session> session;
	tensorflow::Status load_graph_status = LoadGraph(graph, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	// 创建tensorflow的输入tensor
	tensorflow::Tensor input_tensor = convertMatToTensor(images);
	tensorflow::Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
	phase_train.scalar<bool>()() = false;

	//运行tensorflow，并获得结果
	std::vector<tensorflow::Tensor> outputs;
	tensorflow::Status run_status = session->Run({ { input_layer, input_tensor }, { "phase_train:0",  phase_train } }, { output_layer }, {}, &outputs);
	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	//每张图像会输出一个512维的特征向量。将outputs中的1024维向量分成两个512维向量
	auto outputs_map = outputs[0].tensor<float, 2>();
	float *feature0 = new float[512];
	float *feature1 = new float[512];
	for (int i = 0; i < 512; i++) {
		feature0[i] = outputs_map(0, i);
		feature1[i] = outputs_map(1, i);
	}

	//计算两个向量的相似度，即余下距离
	std::cout << "相似度:" << simd_dot(feature0, feature1, 512) / (sqrt(simd_dot(feature0, feature0, 512))* sqrt(simd_dot(feature1, feature1, 512))) << std::endl;
	
	system("pause");
	return 0;
}
