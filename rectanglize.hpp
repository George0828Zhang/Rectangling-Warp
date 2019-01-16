#ifndef _RECT_HPP 
#define _RECT_HPP
// constexpr char TOP = 0;
// constexpr char BOTTOM = 1;
// constexpr char LEFT = 2;
// constexpr char RIGHT = 3;
// constexpr char CORNER_A = 4;
// constexpr char CORNER_B = 5;
// constexpr char CORNER_C = 6;
// constexpr char CORNER_D = 7;
constexpr char TOP = 1;
constexpr char BOTTOM = 2;
constexpr char LEFT = 3;
constexpr char RIGHT = 4;
constexpr char CORNER_A = 5;
constexpr char CORNER_B = 6;
constexpr char CORNER_C = 7;
constexpr char CORNER_D = 8;
void cropOuter(cv::Mat3b& img);
void localWarping(
	cv::Mat3b const& img, 
	cv::Mat1b& border, 
	cv::Mat2i& sourcepix
	);
void UnwarpGrid(
	cv::Mat2i const& sourcepix, 
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