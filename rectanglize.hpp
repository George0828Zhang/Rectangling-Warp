#ifndef _RECT_HPP 
#define _RECT_HPP
void cropOuter(cv::Mat3b& img);
void localWarping(
	cv::Mat3b const& img, 
	cv::Mat1b& border, 
	cv::Mat2i& displacement
	);
void UnwarpGrid(
	cv::Mat2i const& displacement, 
	std::vector<cv::Point>& vertex_map, 
	std::vector<cv::Vec4i>& quads, 
	std::vector<int>& bound_types, 
	int rowdiv
	);
void GetLines(
	cv::Mat3b const& img, 
	std::vector<cv::Vec4i>& out
	);
#endif