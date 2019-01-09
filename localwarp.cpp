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
	cv::Mat2i const& displacement, 
	std::vector<cv::Point>& vertex_map, 
	std::vector<cv::Vec4i>& quads,
	std::vector<int>& bound_types,
	int rowdiv)
{
	int r = displacement.rows, c = displacement.cols;
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
			vertex_map[p] = cv::Point(-displacement[y][x]) + cv::Point(x, y);            
			

			if(y==0){
				if(x == 0){// top left
					bound_types[p] = 4;
				}else if(x == c - 1){//top right
					bound_types[p] = 5;
				}else// top
					bound_types[p] = 0;
			}else if(y == r - 1){
				if(x == 0){// bottom left
					bound_types[p] = 6;
				}else if(x == c - 1){// bottom right
					bound_types[p] = 7;
				}else// bottom
					bound_types[p] = 1;
			}else if(x == 0){// left
				bound_types[p] = 2;
			}else if(x == c - 1){// right
				bound_types[p] = 3;
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

	displacement(cpto) += offset;
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









int subImage(cv::Mat1b const& border, cv::Rect& out){
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
		return 0;
	}else if(Max==bott.z){
		out = cv::Rect( bott.x, 0, bott.z, r );
		return 1;
	}
	else if(Max==left.z){
		out = cv::Rect( 0, left.x, c, left.z );
		return 2;
	}
	else if(Max==right.z){
		out = cv::Rect( 0, right.x, c, right.z );
		return 3;
	}
	return -1;
}
void FindBorder(cv::Mat1b const& img, cv::Mat1b& out){
	int r = img.rows, c = img.cols;
	out = cv::Mat1b::zeros(img.size());
	
	std::queue<cv::Point> test;
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
		visit[y][0] = 1;
		visit[y][c-1] = 2;
	}
	for(int x = 0; x < c; x++){
		test.push(cv::Point(x, 0));
		test.push(cv::Point(x, r-1));
		visit[0][x] = 3;
		visit[r-1][x] = 4;
	}
	
	cv::Point offsets[4] = {{1, 0},{-1, 0},{0, 1},{0, -1}};

	while(!test.empty()){
		cv::Point pix = test.front();

		if(img(pix) < DARKTHRES){
			out(pix) = 255;

			// cv::Point neighbor = pix + offsets[visit(pix)-1];
			// if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < c && neighbor.y < r && !visit(neighbor)){
			// 	visit(neighbor) = visit(pix);
			// 	test.push(neighbor);
			// }
			for(int nei = 0; nei < 4; nei++){
				cv::Point neighbor = pix + offsets[nei];
				if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < c && neighbor.y < r && !visit(neighbor)){
					visit(neighbor) = visit(pix);
					test.push(neighbor);
				}
			}
		}

		test.pop();        
	}

	int dilation_size = 3;
	auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
	cv::Mat tmp;
	cv::dilate( out, tmp, element );
	cv::GaussianBlur(tmp, out, cv::Size(13, 13), 0, 0, cv::BORDER_DEFAULT);
	// cv::GaussianBlur(out, out, cv::Size(13, 13), 0, 0, cv::BORDER_DEFAULT);
	cv::threshold(out, out, 0, 255, cv::THRESH_BINARY);
}
void localWarping(cv::Mat3b const& image, cv::Mat1b& border, cv::Mat2i& displacement){
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

	std::vector<cv::Rect> subseq;
	std::vector<char> tpseq;

	std::vector<int> work;
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

	cv::Sobel(img, grad_x, grad_x.type(), 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(img, grad_y, grad_y.type(), 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
	cv::normalize(grad_x, grad_x, -1., 1., cv::NORM_MINMAX, -1);
	cv::normalize(grad_y, grad_y, -1., 1., cv::NORM_MINMAX, -1);
	energy = grad_x.mul(grad_x) + grad_y.mul(grad_y) + cv::Mat1f(border)*inf;

	// for(int i = 0; i < subseq.size(); i++){
	while((tp = subImage(border, sub)) != -1){
		// sub = subseq[i];
		auto img_sub = img(sub);
		auto border_sub = border(sub);        
		auto seamed_sub = seamed(sub);
		auto energy_sub = energy(sub);
		// auto disp_sub = disps(sub);
		auto disp_sub = displacement(sub);

		// switch(tpseq[i]){
		switch(tp){
			case 0:
			// top
			SeamHorizontal(img_sub, border_sub, seamed_sub, disp_sub, energy_sub, true);
			// cerr << "1 seams up." << endl;
			break;
			case 1:
			// bottom
			SeamHorizontal(img_sub, border_sub, seamed_sub, disp_sub, energy_sub, false);
			// cerr << "1 seams down." << endl;
			break;
			case 2:
			// left
			SeamVertical(img_sub, border_sub, seamed_sub, disp_sub, energy_sub, true);
			// cerr << "1 seams left." << endl;
			break;
			case 3:
			// right
			SeamVertical(img_sub, border_sub, seamed_sub, disp_sub, energy_sub, false);
			// cerr << "1 seams right." << endl;
			break;
		}
		// cv::imshow("Display Image", border);
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
	cv::Mat1b imgsrc;
	img.copyTo(imgsrc);
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			img[i][j] = imgsrc( cv::Point(j, i) + cv::Point(-displacement[i][j]) );
		}
	}
	
	cv::imwrite("results/out.png", img);
	cv::imwrite("results/seams.png", seamed);
	cv::imwrite("results/border.png", border);

	cv::Mat2i displace = displacement.mul(displacement);
	cv::normalize(displace, displace, 0, 255, cv::NORM_MINMAX, -1);    
	std::vector<cv::Mat> out(2);
	cv::split(displace, out);
	out.push_back(cv::Mat1i::zeros(displace.size()));	
	std::swap(out[1], out[2]);
	cv::Mat merged;
	cv::merge(out, merged);
	cv::Mat3b transed(merged);
	cv::imwrite("results/displacement.png", transed);
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