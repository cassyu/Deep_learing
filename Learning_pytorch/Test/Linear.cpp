
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

cv::Mat binear(cv::Mat img, double rx, double ry)
{
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();

	int resize_width = (int)(width * rx);
	int resize_height = (int)(height * ry);

	int x_before, y_before;
	double dx, dy;

	double val;

	double wrate = width / resize_width;
	double hrate = height / resize_height;


	cv::Mat out = cv::Mat::zeros(resize_width,resize_height,CV_8UC3);
	double y;
	double x;

	int src_x_0;
	int src_y_0;
	int src_x_1;
	int src_y_1;

	for (int y = 0; y < resize_height; y++)
	{
		y = (y + 0.5) * hrate - 0.5;//源图上的坐标
		for (int x = 0; x < resize_width; x++)
		{
			x = (x + 0.5) * wrate - 0.5;//原图上的x坐标
			//需要计算周围四个点的所占的比例
			src_x_0 = int(floor(x));
			src_y_0 = int(floor(y));

			src_x_1 = fmin(src_x_0 + 1, width - 1);
			src_y_1 = fmin(src_y_0 + 1,height -1);

			for (int c = 0; c < channel; c++)
			{
				val =(x - src_x_0)*(y - src_y_0)* img.at<cv::Vec3b>(src_x_0,src_y_0)[c]
					+ (x - src_x_0)*(src_y_0 - y)* img.at<cv::Vec3b>(src_x_0, src_y_1)[c]
					+ (src_x_1 - x)*(y - src_y_0)* img.at<cv::Vec3b>(src_x_1, src_y_0)[c]
					+ (src_x_1 - x)*(src_y_0-1)* img.at<cv::Vec3b>(src_x_1, src_y_1)[c];

				out.at<cv::Vec3b>(y, x)[c] = (uchar)val;
			}
		}
	}

	return out;
}

int main(int argc, const char* argv[]) {
	// read image
	cv::Mat img = cv::imread("lena.jpg", cv::IMREAD_COLOR);
    
	cv::Mat out = binear(img, 1.5, 1.5);

	//cv::imwrite("out.jpg", out);
	cv::imshow("answer", out);
	cv::waitKey(0);
	//cv::destroyAllWindows();

	return 0;
}