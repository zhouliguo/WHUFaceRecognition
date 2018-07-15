#include <fstream>
#include <utility>
#include <vector>
#include <Windows.h>
#include <highgui.hpp>
#include <imgproc.hpp>

#include "math_functions.h"

#include "tensorflow/core/public/session.h"


// ��ȡͼ�����ز���
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

//��opencv��Mat������䵽Tenflow��tensor��
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
	tensorflow::string graph = "../model/tensorflow/20180402-114759.pb";	//ͼ�����ģ��
	tensorflow::string input_layer = "input:0";	//�����
	tensorflow::string output_layer = "embeddings:0";	//�����

	//��ȡ��������ͼ��
	cv::Mat image0 = cv::imread(image_path0);
	cv::Mat image1 = cv::imread(image_path1);

	//��ģ������ͼ��ߴ�Ϊ160x160�������Ҫ��ͼ��resize��160x160
	cv::resize(image0, image0, cv::Size(160, 160));
	cv::resize(image1, image1, cv::Size(160, 160));

	//��������float��Matͼ��
	cv::Mat image_input0(image0.rows, image0.cols, CV_32F);
	cv::Mat image_input1(image1.rows, image1.cols, CV_32F);

	//����������ͼ���BGRת����RGB
	cv::cvtColor(image0, image0, CV_BGR2RGB);
	cv::cvtColor(image1, image1, CV_BGR2RGB);

	//����������ͼ���ucharת����float,��������ֵ��0����255��һ����-0.5����0.5
	image0.convertTo(image_input0, CV_32F, 1 / 255.0, -0.5);
	image1.convertTo(image_input1, CV_32F, 1 / 255.0, -0.5);

	//������ͼ��ֵ��Mat����
	std::vector<cv::Mat> images;
	images.push_back(image_input0);
	images.push_back(image_input1);

	// ����tensorflow session������ģ��
	std::unique_ptr<tensorflow::Session> session;
	tensorflow::Status load_graph_status = LoadGraph(graph, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	// ����tensorflow������tensor
	tensorflow::Tensor input_tensor = convertMatToTensor(images);
	tensorflow::Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
	phase_train.scalar<bool>()() = false;

	//����tensorflow������ý��
	std::vector<tensorflow::Tensor> outputs;
	tensorflow::Status run_status = session->Run({ { input_layer, input_tensor }, { "phase_train:0",  phase_train } }, { output_layer }, {}, &outputs);
	if (!run_status.ok()) {
		LOG(ERROR) << "Running model failed: " << run_status;
		return -1;
	}

	//ÿ��ͼ������һ��512ά��������������outputs�е�1024ά�����ֳ�����512ά����
	auto outputs_map = outputs[0].tensor<float, 2>();
	float *feature0 = new float[512];
	float *feature1 = new float[512];
	for (int i = 0; i < 512; i++) {
		feature0[i] = outputs_map(0, i);
		feature1[i] = outputs_map(1, i);
	}

	//�����������������ƶȣ������¾���
	std::cout << "���ƶ�:" << simd_dot(feature0, feature1, 512) / (sqrt(simd_dot(feature0, feature0, 512))* sqrt(simd_dot(feature1, feature1, 512))) << std::endl;
	
	system("pause");
	return 0;
}
