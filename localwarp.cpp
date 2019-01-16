#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cassert>
#include <queue>
#include <algorithm>
#include <cmath>
#include <string>
#include "rectanglize.hpp"

constexpr float inf = 100000000;
constexpr float DARKTHRES = 2.5f;
// constexpr char TOP = 0;
// constexpr char BOTTOM = 1;
// constexpr char LEFT = 2;
// constexpr char RIGHT = 3;

void GetLines(cv::Mat3b const& img, std::vector<cv::Vec4i>& out){
	cv::Mat response;
	const double CannyThres[] = {50, 200};
	const int CannyAperture = 3;
	const bool CannyL2Grad = true;
	cv::Canny( img, response, CannyThres[0], CannyThres[1], CannyAperture, CannyL2Grad );

	const double HoughDistRes = 1;
	const double HoughAngRes = CV_PI/180;
	const double HoughMinLen = 20;//30
	const double HoughMaxGap = 5;//10
	const int HoughThres = 70;//80
	cv::HoughLinesP( response, out, HoughDistRes, HoughAngRes, HoughThres, HoughMinLen, HoughMaxGap );
}
void inc(int& v, int i, int cap){
	v = v==cap ? cap+1 : std::min(v+i, cap);
}
void UnwarpGrid(
	cv::Mat2i const& sourcepix, 
	std::vector<cv::Point>& vertex_map, 
	std::vector<cv::Vec4i>& quads,
	std::vector<int>& bound_types,
	int rowdiv)
{
	int r = sourcepix.rows, c = sourcepix.cols;
	int g = (int)std::floor(r/rowdiv);
	int virt_elem = (int)std::ceil(r/(double)g)+1;
	int hori_elem = (int)std::ceil(c/(double)g)+1;
	int V = virt_elem * hori_elem;
	vertex_map.resize(V);
	bound_types.resize(V);
	// int p = 0;
	for(int i = 0; i < virt_elem; i++){
		for(int j = 0; j < hori_elem; j++){
			int x = std::min(j * g, c - 1);
			int y = std::min(i * g, r - 1);
			int p = j*virt_elem + i;
			// vertex_map[p] = cv::Point(-displacement[y][x]) + cv::Point(x, y);
			vertex_map[p] = cv::Point(sourcepix[y][x]);
			

			if(y==0){
				if(x == 0){// top left
					bound_types[p] = CORNER_A;
				}else if(x == c - 1){//top right
					bound_types[p] = CORNER_B;
				}else// top
					bound_types[p] = TOP;
			}else if(y == r - 1){
				if(x == 0){// bottom left
					bound_types[p] = CORNER_C;
				}else if(x == c - 1){// bottom right
					bound_types[p] = CORNER_D;
				}else// bottom
					bound_types[p] = BOTTOM;
			}else if(x == 0){// left
				bound_types[p] = LEFT;
			}else if(x == c - 1){// right
				bound_types[p] = RIGHT;
			}else// not bound
				bound_types[p] = -1;

			if(x > 0 && y > 0){
				quads.push_back({p - virt_elem - 1, p - 1, p - virt_elem, p});
			}

			// p++;
		}
	}
	// for(int x = 0; x < c; inc(x, g, c-1)){
	// 	for(int y = 0; y < r; inc(y, g, r-1)){
	// 		vertex_map[p] = cv::Point(-displacement[y][x]) + cv::Point(x, y);            
			

	// 		if(y==0){
	// 			if(x == 0){// top left
	// 				bound_types[p] = 4;
	// 			}else if(x == c - 1){//top right
	// 				bound_types[p] = 5;
	// 			}else// top
	// 				bound_types[p] = 0;
	// 		}else if(y == r - 1){
	// 			if(x == 0){// bottom left
	// 				bound_types[p] = 6;
	// 			}else if(x == c - 1){// bottom right
	// 				bound_types[p] = 7;
	// 			}else// bottom
	// 				bound_types[p] = 1;
	// 		}else if(x == 0){// left
	// 			bound_types[p] = 2;
	// 		}else if(x == c - 1){// right
	// 			bound_types[p] = 3;
	// 		}else// not bound
	// 			bound_types[p] = -1;

	// 		if(x > 0 && y > 0){
	// 			quads.push_back({p - virt_elem - 1, p - 1, p - virt_elem, p});
	// 		}

	// 		p++;
	// 	}
	// }
}

void shift(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed, 
	cv::Mat2i& displacement, 
	cv::Mat1f& energy, 
	cv::Point2i const& at, 
	char type)
{
	int r = img.rows, c = img.cols;
	cv::Mat1b tmp;
	cv::Mat1f tmp2;
	cv::Mat2i tmp3;
	cv::Rect from, cpto;
	cv::Vec2i offset;

	switch(type){
		case 'u':
		from = cv::Rect(at.x, 1, 1, at.y);
		cpto = cv::Rect(at.x, 0, 1, at.y);
		offset = cv::Vec2i(0, -1);
		break;
		case 'd':
		from = cv::Rect(at.x, at.y, 1, r-1-at.y);
		cpto = cv::Rect(at.x, at.y+1, 1, r-1-at.y);
		offset = cv::Vec2i(0, 1);
		break;
		case 'l':
		from = cv::Rect(1, at.y, at.x, 1);
		cpto = cv::Rect(0, at.y, at.x, 1);
		offset = cv::Vec2i(-1, 0);
		break;
		case 'r':
		from = cv::Rect(at.x, at.y, c-1-at.x, 1);
		cpto = cv::Rect(at.x+1, at.y, c-1-at.x, 1);
		offset = cv::Vec2i(1, 0);
		break;
	}	
	img(from).copyTo(tmp);
	tmp.copyTo(img(cpto));
	border(from).copyTo(tmp);
	tmp.copyTo(border(cpto));
	seamed(from).copyTo(tmp);
	tmp.copyTo(seamed(cpto));

	energy(from).copyTo(tmp2);
	tmp2.copyTo(energy(cpto));

	// displacement(cpto) += offset;

	displacement(from).copyTo(tmp3);
	tmp3.copyTo(displacement(cpto));
}
void SeamHorizontal(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed,
	cv::Mat2i& displacements,
	cv::Mat1f& energy, 
	bool up)
{
	int r = img.rows, c = img.cols;
	double penalty = 2.0;

	cv::Mat1i phi(img.size());
	cv::Mat1f delta(img.size());

	// r = 0
	for(int i = 0; i < r; i++){
		delta[i][0] = energy[i][0];        
	}

	// r > 0
	for(int j = 1; j < c; j++){
		for(int i = 0; i < r; i++){
			double mn = delta[i][j-1];
			phi[i][j] = i;
			if(i>0 && delta[i-1][j-1]<mn){
				mn = delta[i-1][j-1];
				phi[i][j] = i - 1;
			}
			if(i<r-1 && delta[i+1][j-1]<mn){
				mn = delta[i+1][j-1];
				phi[i][j] = i + 1;
			}
			delta[i][j] = energy[i][j] + mn;
			if(seamed[i][j]){
				delta[i][j] += penalty;
			} 
		}
	}

	int best_i = 0;	
	for(int i = 1; i < r; i++){
		if(delta[i][c-1]<delta[best_i][c-1]){
			best_i = i;
		}
	}

	// back-tracking	
	for(int j = c-1; j >= 0; j--){
		
		// if(up && best_i > 0){
		// 	displacements[best_i][j][0] += 1;
		// }
		// else if(best_i < r-1){          
		// 	displacements[best_i][j][1] += 1;

		// }
		seamed[best_i][j] = 255;
		if(up && best_i > 0){
			shift(img, border, seamed, displacements, energy, cv::Point2i(j, best_i),'u');
			// displacement(cv::Rect(j, 0, 1, best_i)) += cv::Vec2i(0, -1);
		}
		else if(best_i < r-1){
			shift(img, border, seamed, displacements, energy, cv::Point2i(j, best_i),'d');
			// displacement(cv::Rect(j, best_i+1, 1, r-1-best_i)) += cv::Vec2i(0, 1);
		}

		best_i = phi[best_i][j];
	}
}







// ///////////////////////////////////////
// ///////////////////////////////////////







void SeamVertical(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed, 
	cv::Mat2i& displacements, 
	cv::Mat1f& energy, 
	bool left)
{
	int r = img.rows, c = img.cols;
	double penalty = 2.0;

	cv::Mat1i phi(img.size());
	cv::Mat1f delta(img.size());

	// c = 0
	for(int j = 0; j < c; j++){
		delta[0][j] = energy[0][j];
	}

	// c > 0
	for(int i = 1; i < r; i++){
		for(int j = 0; j < c; j++){
			double mn = delta[i-1][j];
			phi[i][j] = j;
			if(j>0 && delta[i-1][j-1]<mn){
				mn =delta[i-1][j-1];
				phi[i][j] = j - 1;
			}
			if(j<c-1 && delta[i-1][j+1]<mn){
				mn = delta[i-1][j+1];
				phi[i][j] = j + 1;
			}
			delta[i][j] = energy[i][j] + mn;
			if(seamed[i][j]){
				delta[i][j] += penalty;
			} 
		}
	}

	int best_j = 0;	
	for(int j = 1; j < c; j++){
		if(delta[r-1][j]<delta[r-1][best_j]){
			best_j = j;
		}
	}
	// back-tracking	
	for(int i = r-1; i >= 0; i--){
		
		// if(left && best_j > 0){
		// 	displacements[i][best_j][2] += 1;
		// }
		// else if(best_j < c-1){
		// 	displacements[i][best_j][3] += 1;
		// }

		seamed[i][best_j] = 255;

		if(left && best_j > 0){
			shift(img, border, seamed, displacements, energy, cv::Point2i(best_j, i),'l');
			// displacement(cv::Rect(i, 0, 1, best_j)) += cv::Vec2i(-1, 0);
		}
		else if(best_j < c-1){
			shift(img, border, seamed, displacements, energy, cv::Point2i(best_j, i),'r');
			// displacement(cv::Rect(i, best_j+1, 1, c-1-best_j)) += cv::Vec2i(1, 0);
		}
		best_j = phi[i][best_j];
	}
}









int subImage0(cv::Mat1b const& border, cv::Rect& out){
	// longest border 
	int r = border.rows, c = border.cols;
	cv::Point3i top(0), bott(0), left(0), right(0);
	// std::vector<int> start(std::max(c, r), 0);
	// for(int j = 0; j < c; j++){
	// 	if(border[0][j]){
	// 		start[j] = j>0 ? start[j-1] : j;
	// 		if(j - start[j] + 1 > top.z){
	// 			top = { start[j], j, j - start[j] + 1};
	// 		}
	// 	}else{
	// 		start[j] = j+1;
	// 	}
	// }
	// for(int j = 0; j < c; j++){
	// 	if(border[r-1][j]){
	// 		start[j] = j > 0 ? start[j-1] : j;
	// 		if(j - start[j] + 1 > bott.z){
	// 			bott = { start[j], j, j - start[j] + 1};
	// 		}
	// 	}else{
	// 		start[j] = j+1;
	// 	}
	// }
	// for(int i = 0; i < r; i++){
	// 	if(border[i][0]){
	// 		start[i] = i > 0 ? start[i-1] : i;
	// 		if(i - start[i] + 1 > left.z){
	// 			left = { start[i], i, i - start[i] + 1};
	// 		}
	// 	}else{
	// 		start[i] = i+1;
	// 	}
	// }
	// for(int i = 0; i < r; i++){
	// 	if(border[i][c-1]){
	// 		start[i] = i > 0 ? start[i-1] : i;
	// 		if(i - start[i] + 1 > right.z){
	// 			right = { start[i], i, i - start[i] + 1};
	// 		}
	// 	}else{
	// 		start[i] = i+1;
	// 	}
	// }
	for(int j = 0, begin = -1, end = -1; j < c; j++){
		if(border[0][j]){
			if(begin==-1)
				begin = j;
			end = j + 1;
		}else{
			begin = -1;
			end = -1;
		}
		if(end - begin > top.z){
			top = {begin, end, end - begin};
		}
	}
	for(int j = 0, begin = -1, end = 0; j < c; j++){
		if(border[r-1][j]){
			if(begin==-1)
				begin = j;
			end = j + 1;
		}else{
			begin = -1;
			end = -1;
		}
		if(end - begin > bott.z){
			bott = {begin, end, end - begin};
		}
	}
	for(int i = 0, begin = -1, end = 0; i < r; i++){
		if(border[i][0]){
			if(begin==-1)
				begin = i;
			end = i + 1;
		}else{
			begin = -1;
			end = -1;
		}
		if(end - begin > left.z){
			left = {begin, end, end - begin};
		}
	}
	for(int i = 0, begin = -1, end = 0; i < r; i++){
		if(border[i][c-1]){
			if(begin==-1)
				begin = i;
			end = i + 1;
		}else{			
			begin = -1;
			end = -1;
		}
		if(end - begin > right.z){
			right = {begin, end, end - begin};
		}
	}

	int Max = std::max({top.z, bott.z, left.z, right.z});
	
	if(Max==0){
		return -1;
	}
	else if(Max==top.z){
		out = cv::Rect( top.x, 0, top.z, r );
		return TOP;
	}else if(Max==bott.z){
		out = cv::Rect( bott.x, 0, bott.z, r );
		return BOTTOM;
	}
	else if(Max==left.z){
		out = cv::Rect( 0, left.x, c, left.z );
		return LEFT;
	}
	else if(Max==right.z){
		out = cv::Rect( 0, right.x, c, right.z );
		return RIGHT;
	}
	return -1;
}

double density(cv::Mat1b const& mat){
	double response = cv::countNonZero(mat);
	double area = mat.rows*mat.cols;
	return response? response / area : 0;
}


int subImage(cv::Mat1b const& border, cv::Rect& out){
	// longest border 
	int r = border.rows, c = border.cols;
	int SeamSize = 0, Type = -1;
	cv::Point from = {0,0}, to = {0,0};

	// vertical
	cv::Mat1i vertical = (border>0)/255;
	cv::Mat1i horizontal = (border>0)/255;

	// std::cerr << "[debug] " << vertical << std::endl;

	for(int x = 0; x < c; x++){
		for(int y = 0; y < r; y++){
			cv::Point midpoint;

			if(x && horizontal[y][x]){
				horizontal[y][x] += horizontal[y][x-1];
				midpoint = {x - horizontal[y][x]/2 + 1, y};
				if(horizontal[y][x]>SeamSize){
					if(y == 0 || border(midpoint)==TOP){
						Type = TOP;
					}else if(y == r - 1 || border(midpoint)==BOTTOM){
						Type = BOTTOM;
					}else continue;

					SeamSize = horizontal[y][x];
					from = {x - horizontal[y][x] + 1, y};
					to = {x, y};
				}
			}
			if(y && vertical[y][x]){
				vertical[y][x] += vertical[y-1][x];
				midpoint = {x, y - vertical[y][x]/2 + 1};
				if(vertical[y][x]>SeamSize){
					if(x == 0 || border(midpoint)==LEFT){
						Type = LEFT;
					}else if(x == c - 1 || border(midpoint)==RIGHT){
						Type = RIGHT;
					}else continue;

					SeamSize = vertical[y][x];
					from = {x, y - vertical[y][x] + 1};
					to = {x, y};
				}
			}
		}
	}
	
	if(from - to == cv::Point(0,0)){
		return -1;
	}
	
	if(Type==TOP){
		out = cv::Rect( from.x, from.y, SeamSize, r - from.y );
	}else if(Type==BOTTOM){
		out = cv::Rect( from.x, 0, SeamSize, from.y + 1);
	}
	else if(Type==LEFT){
		out = cv::Rect( from.x, from.y, c - from.x, SeamSize );
	}
	else if(Type==RIGHT){
		out = cv::Rect( 0, from.y, from.x + 1, SeamSize );
	}
	return Type;


	// TODO: dynamic programming
}

void FindBorder(cv::Mat1b const& img, cv::Mat1b& out){
	int r = img.rows, c = img.cols;
	out = cv::Mat1b::zeros(img.size());

	std::vector<cv::Mat1b> borders(4);
	for(auto& p : borders){
		p = out.clone();
	}
	
	std::queue<cv::Point> test;
	std::queue<int> type;
	// connected graph
	// find first dead pixel
	cv::Mat1b visit = cv::Mat1b::zeros(img.size());
	// cv::Mat1b (cv::Mat1f(img)/255.0);
	// cv::Mat1b minpool(cv::Mat1f(img)/255.0);
	// cv::GaussianBlur(img, minpool, cv::Size(11, 11), 0, 0, cv::BORDER_DEFAULT);
	// auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );
	// cv::erode( img, minpool, element );

	for(int y = 0; y < r; y++){
		test.push(cv::Point(0, y));
		test.push(cv::Point(c-1, y));
		type.push(LEFT);
		type.push(RIGHT);
		visit[y][0] = 1;
		visit[y][c-1] = 1;
	}
	for(int x = 0; x < c; x++){
		test.push(cv::Point(x, 0));
		test.push(cv::Point(x, r-1));
		type.push(TOP);
		type.push(BOTTOM);
		visit[0][x] = 1;
		visit[r-1][x] = 1;
	}
	
	cv::Point offsets[4] = {{1, 0},{-1, 0},{0, 1},{0, -1}};

	while(!test.empty()){
		cv::Point pix = test.front();
		int tp = type.front();

		if(img(pix) < DARKTHRES){
			// out(pix) = tp;
			// switch(tp){
			// 	case TOP:
			// 		top_border(pix) = 255;
			// 	break;
			// 	case BOTTOM:
			// 		bottom_border(pix) = 255;
			// 	break;
			// 	case LEFT:
			// 		left_border(pix) = 255;
			// 	break;
			// 	case RIGHT:
			// 		right_border(pix) = 255;
			// 	break;
			// 	default:
			// 		out(pix) = 255;
			// 	break;
			// }
			borders[tp-TOP](pix) = 255;

			for(int nei = 0; nei < 4; nei++){
				cv::Point neighbor = pix + offsets[nei];
				if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < c && neighbor.y < r && !visit(neighbor)){
					visit(neighbor) = 1;
					test.push(neighbor);
					type.push(tp);
				}
			}
		}

		type.pop();
		test.pop();        
	}


	for(auto& map : borders){
		// int dilation_size = 3;
		// auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
		// cv::Mat tmp;
		// cv::erode( map, tmp, element );
		// cv::GaussianBlur(tmp, map, cv::Size(11, 11), 0, 0, cv::BORDER_DEFAULT);

		// cv::imshow("Display Image", map);
		// cv::waitKey(0);

		cv::Mat tmp;
		cv::GaussianBlur(map, tmp, cv::Size(7, 7), 0, 0, cv::BORDER_DEFAULT);
		cv::threshold(tmp, tmp, 0, 255, cv::THRESH_BINARY);

		int dilation_size = 3;
		auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
		
		cv::erode( tmp, map, element );
		
		// cv::imshow("Display Image", map);
		// cv::waitKey(0);

		map /= 255;
	}
	// out = borders[0]*TOP + borders[1]*BOTTOM + borders[2]*LEFT + borders[3]*RIGHT;
	out = borders[0]*TOP + (borders[1] - (borders[1] & borders[0]))*BOTTOM;
	out = out + (borders[2] - (borders[2] & (out>0)))*LEFT;
	out = out + (borders[3] - (borders[3] & (out>0)))*RIGHT;
	 // cv::imshow("Display Image", out*50);
		// cv::waitKey(0);

	// int dilation_size = 3;
	// auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
	// cv::Mat tmp;
	// cv::dilate( out, tmp, element );
	// cv::GaussianBlur(tmp, out, cv::Size(13, 13), 0, 0, cv::BORDER_DEFAULT);
	// // cv::GaussianBlur(out, out, cv::Size(13, 13), 0, 0, cv::BORDER_DEFAULT);
	// cv::threshold(out, out, 0, 255, cv::THRESH_BINARY);
}
void localWarping(cv::Mat3b const& image, cv::Mat1b& border, cv::Mat2i& sourcepix){
	int r = image.rows;
	int c = image.cols;
	int tp;

	cv::Mat1b img;
	cv::Mat1b border_tmp;
	cv::Mat1b seamed;
	cv::Mat1b tmp;
	// cv::Mat4i disps = cv::Mat4i::zeros(r, c);

	cv::Rect sub;
	cv::Rect from, cpto;

	cv::cvtColor(image, img, cv::COLOR_BGR2GRAY);	
	FindBorder(img, border);
	border.copyTo(border_tmp);
	seamed = cv::Mat1b(img.size(), 0);

	cv::Mat1f grad_x;
	cv::Mat1f grad_y;
	cv::Mat1f energy;

	// std::vector<cv::Rect> subseq;
	// std::vector<char> tpseq;

	// std::vector<int> work;
	// while((tp = subImage(border_tmp, sub)) != -1){
	// 	subseq.push_back(sub);
	// 	tpseq.push_back(tp);

	// 	work.push_back(sub.width*(!(tp/2))+sub.height*(tp/2));
	// 	if(work.size()>1)
	// 		work.back() += work[work.size()-2];

	// 	cv::Rect window;
	// 	cv::Point move;
	// 	switch(tp){
	// 		case 0:
	// 		window = cv::Rect(sub.x, 0, sub.width, 1);
	// 		move = cv::Point(0, 1);			
	// 		// cerr << "Up" << endl;
	// 		break;
	// 		case 1:
	// 		window = cv::Rect(sub.x, r-1, sub.width, 1);
	// 		move = cv::Point(0, -1);
	// 		// cerr << "Down" << endl;
	// 		break;
	// 		case 2:
	// 		window = cv::Rect(0, sub.y, 1, sub.height);
	// 		move = cv::Point(1, 0);
	// 		// cerr << "Left" << endl;
	// 		break;
	// 		case 3:
	// 		window = cv::Rect(c-1, sub.y, 1, sub.height);
	// 		move = cv::Point(-1, 0);
	// 		// cerr << "Right" << endl;
	// 		break;
	// 	}
		
	// 	while(cv::countNonZero(border_tmp(window))>0){
	// 		cv::Rect src(window.x+move.x, window.y+move.y, window.width, window.height);
	// 		if(src.x < 0 || src.y < 0 || src.x >= c || src.y >= r)
	// 			break;
	// 		border_tmp(src).copyTo(border_tmp(window));
	// 		window = src;
	// 	}
	// }
	sourcepix = cv::Mat2i::zeros(image.size());
	for(int x = 0; x < c; x++){
		for(int y = 0; y < r; y++){
			sourcepix[y][x] = cv::Vec2i(x, y);
		}
	}

	cv::Sobel(img, grad_x, grad_x.type(), 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(img, grad_y, grad_y.type(), 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
	cv::normalize(grad_x, grad_x, -1., 1., cv::NORM_MINMAX, -1);
	cv::normalize(grad_y, grad_y, -1., 1., cv::NORM_MINMAX, -1);
	energy = grad_x.mul(grad_x) + grad_y.mul(grad_y) + cv::Mat1f(border>0)*inf;

	// for(int i = 0; i < subseq.size(); i++){
	while((tp = subImage(border, sub)) != -1){
		// sub = subseq[i];
		auto img_sub = img(sub);
		auto border_sub = border(sub);        
		auto seamed_sub = seamed(sub);
		auto energy_sub = energy(sub);
		// auto disp_sub = disps(sub);
		auto src_sub = sourcepix(sub);

		// switch(tpseq[i]){
		switch(tp){
			case TOP:
			// top
			SeamHorizontal(img_sub, border_sub, seamed_sub, src_sub, energy_sub, true);
			// cerr << "1 seams up." << endl;
			break;
			case BOTTOM:
			// bottom
			SeamHorizontal(img_sub, border_sub, seamed_sub, src_sub, energy_sub, false);
			// cerr << "1 seams down." << endl;
			break;
			case LEFT:
			// left
			SeamVertical(img_sub, border_sub, seamed_sub, src_sub, energy_sub, true);
			// cerr << "1 seams left." << endl;
			break;
			case RIGHT:
			// right
			SeamVertical(img_sub, border_sub, seamed_sub, src_sub, energy_sub, false);
			// cerr << "1 seams right." << endl;
			break;
		}
		// cv::imshow("Display Image", img);
		// cv::waitKey(0);

		// std::string bar(30, '.');
		// std::fill(bar.begin(), bar.begin() + (int)std::floor(30 * work[i] / work.back()),'=');
		// std::cerr << "(" << i+1 << "/" << subseq.size() << ") seams done. Progress [" << bar << "]\r";
	}
	std::cerr << std::endl << "Local warping complete." << std::endl;

	// compute displacement
	// cerr << "all zero? " << (cv::countNonZero(disps)==0) << endl;
	// for(int i = 1; i < r; i++){
	// 	for(int j = 0; j < c; j++){
	// 		disps[r-1-i][j][0] += disps[r-i][j][0];
	// 		disps[i][j][1] += disps[i-1][j][1];
	// 	}
	// }
	// for(int j = 1; j < c; j++){
	// 	for(int i = 0; i < r; i++){
	// 		disps[i][c-1-j][2] += disps[i][c-j][2];
	// 		disps[i][j][3] += disps[i][j-1][3];
	// 	}
	// }
	// for(int i = 0; i < r; i++){
	// 	for(int j = 0; j < c; j++){
	// 		int y = disps[i][j][1] - disps[i][j][0]; 
	// 		int x = disps[i][j][3] - disps[i][j][2]; 
	// 		displacement[i][j] = cv::Vec2i(x, y);
	// 	}
	// }


	// local warping
	// cv::Mat3b imgsrc;
	// // img.copyTo(imgsrc);
	// cv::Mat3b color(image.size());
	// for(int i = 0; i < r; i++){
	// 	for(int j = 0; j < c; j++){
	// 		color[i][j] = image( cv::Point(j, i) + cv::Point(-displacement[i][j]) );
	// 	}
	// }
	cv::Mat3b color(image.size());
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			color[i][j] = image( cv::Point(sourcepix[i][j]) );
		}
	}
	// cv::Mat1b imgsrc;
	// img.copyTo(imgsrc);
	// for(int i = 0; i < r; i++){
	// 	for(int j = 0; j < c; j++){
	// 		// img[i][j] = imgsrc( cv::Point(j, i) + cv::Point(-displacement[i][j]) );
	// 		img[i][j] = imgsrc( cv::Point(displacement[i][j]) );
	// 		displacement[i][j] = cv::Vec2i(j, i) - displacement[i][j];
	// 	}
	// }

	cv::imwrite("results/out.png", color);
	cv::imwrite("results/seams.png", seamed);
	// cv::imwrite("results/border.png", border>0);

	// cv::Mat2i displace = displacement.mul(displacement);
	// cv::normalize(displace, displace, 0, 255, cv::NORM_MINMAX, -1);    
	// std::vector<cv::Mat> out(2);
	// cv::split(displace, out);
	// out.push_back(cv::Mat1i::zeros(displace.size()));	
	// std::swap(out[1], out[2]);
	// cv::Mat merged;
	// cv::merge(out, merged);
	// cv::Mat3b transed(merged);
	// cv::imwrite("results/displacement.png", transed);
}


void cropOuter(cv::Mat3b& img){
	int r = img.rows, c = img.cols;
	int up = 0, down = r - 1, left = 0, right = c - 1;

	cv::Mat1b grey;
	cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
	cv::Mat minpool;
	auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );
	cv::erode( grey, minpool, element );
	grey = minpool;

	bool f = false;
	for(int i = 0; !f && i < r; i++){
		for(int j = 0; !f && j < c; j++){
			if(grey[i][j] > DARKTHRES){
				up = i;
				f = true;
			}
		}
	}
	f = false;
	for(int i = r-1; !f && i >= 0; i--){
		for(int j = 0; !f && j < c; j++){
			if(grey[i][j] > DARKTHRES){
				down = i;
				f = true;
			}
		}
	}
	f = false;
	for(int j = 0; !f && j < c; j++){
		for(int i = up; !f && i <= down; i++){
			if(grey[i][j] > DARKTHRES){
				left = j;
				f = true;
			}
		}
	}
	f = false;
	for(int j = c-1; !f && j >= 0; j--){
		for(int i = up; !f && i <= down; i++){
			if(grey[i][j] > DARKTHRES){
				right = j;
				f = true;
			}
		}
	}
	img = img(cv::Rect(left, up, right - left + 1, down - up + 1));
}

int subImagexxx(cv::Mat1b const& border, cv::Rect& out){
	// longest border 
	int r = border.rows, c = border.cols;
	int SeamSize = 0, Type = -1;
	cv::Point from = {0,0}, to = {0,0};

	// vertical
	cv::Mat1i vertical = (border>0)/255;
	cv::Mat1i horizontal = (border>0)/255;

	// std::cerr << "[debug] " << vertical << std::endl;

	for(int x = 0; x < c; x++){
		for(int y = 0; y < r; y++){
			cv::Point from2, to2;

			if(x && horizontal[y][x]){
				horizontal[y][x] += horizontal[y][x-1];
				from2 = {x - horizontal[y][x] + 1, y};
				to2 = {x, y};
				// std::cerr << "[debug] " << from2 << to2 << std::endl;
				if(horizontal[y][x]>SeamSize && (border(from2)==TOP||border(from2)==BOTTOM||border(to2)==TOP||border(to2)==BOTTOM)){
					SeamSize = horizontal[y][x];
					from = from2;
					to = to2;
				}
			}
			if(y && vertical[y][x]){
				vertical[y][x] += vertical[y-1][x];
				from2 = {x, y - vertical[y][x] + 1};
				to2 = {x, y};
				// std::cerr << "[debug] " << from2 << to2 << std::endl;
				if(vertical[y][x]>SeamSize && (border(from2)==LEFT||border(from2)==RIGHT||border(to2)==LEFT||border(to2)==RIGHT)){
					SeamSize = vertical[y][x];
					from = from2;
					to = to2;
				}
			}
		}
	}
	
	if(from - to == cv::Point(0,0)){
		return -1;
	}
	if(from.y == to.y){
		out = cv::Rect( std::min(from.x, to.x), 0, SeamSize, from.y + 1 );
		double lstdense = density(border(out));
		char tp = BOTTOM;
		for(int y = 0; y < r; y++){
			cv::Rect region( std::min(from.x, to.x), std::min(y, from.y), SeamSize, std::abs(y - from.y)+1);
			double cur = density(border(region));
			if(cur < lstdense){
				out = region;
				lstdense = cur;
				tp = (y > from.y) ? TOP : BOTTOM;
			}
		}
		return tp;
	}

	if(from.x == to.x){
		out = cv::Rect( 0, std::min(from.y, to.y), from.x + 1, SeamSize );
		double lstdense = density(border(out));
		char tp = RIGHT;
		for(int x = 0; x < c; x++){
			cv::Rect region( std::min(x, from.x), std::min(from.y, to.y), std::abs(x - from.x)+1, SeamSize);
			double cur = density(border(region));
			if(cur < lstdense){
				out = region;
				lstdense = cur;
				tp = (x > from.x) ? LEFT : RIGHT;
			}
		}
		return tp;
	}

	std::cerr << "[ERR] Some error in border detection has occurred." << std::endl;
	exit(1);

	// TODO: dynamic programming
}