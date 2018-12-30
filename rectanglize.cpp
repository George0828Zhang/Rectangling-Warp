#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cassert>
#include <limits>
#include <queue>
#include <algorithm>
#include <cmath>
#include "Array.hpp"

constexpr double inf = std::numeric_limits<double>::infinity();
const cv::Vec3b DARKPIX = cv::Vec3b({0,0,0});

void cropOuter(cv::Mat3b& img);
void localWarping(cv::Mat3b const& img, cv::Mat2i& displacement);
void UnwarpGrid(cv::Mat2i const& displacement, std::vector<cv::Point>& out);
int main(int argc, char** argv )
{
	cv::Mat3b image;
	image = cv::imread( argv[1], cv::IMREAD_COLOR );
	if ( !image.data )
	{
		printf("No image data \n");
		return -1;
	}
	// cv::resize(image, image, cv::Size(), 0.5, 0.5, cv::INTER_CUBIC);
	// cv::imwrite("pano_compact.png", image);
	cropOuter(image);

	cv::Mat2i displace = cv::Mat2i::zeros(image.size());

	localWarping(image, displace);

	std::vector<cv::Point> Mesh(350);
	UnwarpGrid(displace, Mesh);

	for(auto& p : Mesh){
		cv::circle(image, p, 2.5, cv::Scalar(255.0), -1);
	}
	cout << Mesh.size() << endl;
	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	cv::imshow("Display Image", image);
    cv::waitKey(0);
 //    displace = displace.mul(displace);
 //    cv::normalize(displace, displace, 0, 255, cv::NORM_MINMAX, -1);
 //    cv::Mat2b dis(displace);
 //    std::vector<cv::Mat> out(2);
 //    cv::split(dis, out);
 //    out.push_back(cv::Mat1b::zeros(dis.size()));
 //    std::swap(out[1], out[2]);
 //    cv::Mat tmp;
 //    cv::merge(out, tmp);
	// cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	// cv::imshow("Display Image", tmp);
 //    cv::waitKey(0);



	return 0;
}

void inc(int& v, int i, int cap){
	v = cap-v<i/2? cap+1 : std::min(v+i, cap);
}
void UnwarpGrid(cv::Mat2i const& displacement, std::vector<cv::Point>& out){
	int r = displacement.rows, c = displacement.cols;
	int V = out.size();
	int g = (int)std::floor(((r + c)+std::sqrt((r + c)*(r + c)+4*r*c*(V-1)))/(V-1)/2.);
	V = (int)(std::ceil(r/(double)g)+1)*(std::ceil(c/(double)g)+1);
	out.resize(V);
	int p = 0;
	for(int x = 0; x < c && p < V; inc(x, g, c-1)){
		for(int y = 0; y < r && p < V; inc(y, g, r-1)){
			out[p++] = cv::Point(-displacement[y][x]) + cv::Point(x, y);
		}
	}
	out.resize(p);
}
void shift(
	cv::Mat2i& displacement,
	cv::Point2i const& at, 
	char type)
{
	int r = displacement.rows, c = displacement.cols;
	cv::Rect from, cpto;
	cv::Vec2i offset;

	switch(type){
		case 'u':
		// from = cv::Rect(at.x, 1, 1, at.y);
		cpto = cv::Rect(at.x, 0, 1, at.y);
		offset = cv::Vec2i(0, -1);
		break;
		case 'd':
		// from = cv::Rect(at.x, at.y, 1, r-1-at.y);
		cpto = cv::Rect(at.x, at.y+1, 1, r-1-at.y);
		offset = cv::Vec2i(0, 1);
		break;
		case 'l':
		// from = cv::Rect(1, at.y, at.x, 1);
		cpto = cv::Rect(0, at.y, at.x, 1);
		offset = cv::Vec2i(-1, 0);
		break;
		case 'r':
		// from = cv::Rect(at.x, at.y, c-1-at.x, 1);
		cpto = cv::Rect(at.x+1, at.y, c-1-at.x, 1);
		offset = cv::Vec2i(1, 0);
		break;
	}
	displacement(cpto) += offset;
}

DyArray<double> delta;
DyArray<int> phi;
void SeamHorizontal(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed, 
	cv::Mat2i& displacement, 
	cv::Mat1f& energy, 
	bool up)
{
	int r = img.rows, c = img.cols;
	
	delta.reshape({r, c});
	phi.reshape({r, c});    

	double penalty = 2.0;

	delta.clear(penalty);
	phi.clear(-1);

	// r = 0
	for(int i = 0; i < r; i++){
		delta[{i, 0}] = energy[i][0];        
	}

	// r > 0
	for(int j = 1; j < c; j++){
		for(int i = 0; i < r; i++){
			double mn = delta[{i,j-1}];
			phi[{i,j}] = i;
			if(i>0 && delta[{i-1,j-1}]<mn){
				mn = delta[{i-1,j-1}];
				phi[{i,j}] = i - 1;
			}
			if(i<r-1 && delta[{i+1,j-1}]<mn){
				mn = delta[{i+1,j-1}];
				phi[{i,j}] = i + 1;
			}
			delta[{i,j}] = energy[i][j] + mn;
			if(border[i][j]){
				delta[{i,j}] = inf;
			}else if(seamed[i][j]){
				delta[{i,j}] += penalty;
			} 
		}
	}

	int best_i = 0;	
	for(int i = 1; i < r; i++){
		if(delta[{i,c-1}]<delta[{best_i,c-1}]){
			best_i = i;
		}
	}

	// back-tracking	
	for(int j = c-1; j >= 0; j--){
		
		if(up && best_i > 0){
			shift(displacement, cv::Point2i(j, best_i),'u');
		}
		else if(best_i < r-1){
			shift(displacement, cv::Point2i(j, best_i),'d');
		}

		seamed[best_i][j] = 255;
		best_i = phi[{best_i, j}];
	}
}







// ///////////////////////////////////////
// ///////////////////////////////////////







void SeamVertical(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed, 
	cv::Mat2i& displacement, 
	cv::Mat1f& energy, 
	bool left)
{
	int r = img.rows, c = img.cols;
		
	delta.reshape({r, c});
	phi.reshape({r, c});    

	double penalty = 2.0;

	delta.clear(penalty);
	phi.clear(-1);

	// c = 0
	for(int j = 0; j < c; j++){
		delta[{0, j}] = energy[0][j];
	}

	// c > 0
	for(int i = 1; i < r; i++){
		for(int j = 0; j < c; j++){
			double mn = delta[{i-1,j}];
			phi[{i, j}] = j;
			if(j>0 && delta[{i-1,j-1}]<mn){
				mn =delta[{i-1,j-1}];
				phi[{i,j}] = j - 1;
			}
			if(j<c-1 && delta[{i-1,j+1}]<mn){
				mn = delta[{i-1,j+1}];
				phi[{i,j}] = j + 1;
			}
			delta[{i,j}] = energy[i][j] + mn;
			if(border[i][j]){
				delta[{i,j}] = inf;
			}else if(seamed[i][j]){
				delta[{i,j}] += penalty;
			} 
		}
	}

	int best_j = 0;	
	for(int j = 1; j < c; j++){
		if(delta[{r-1,j}]<delta[{r-1,best_j}]){
			best_j = j;
		}
	}
	// back-tracking	
	for(int i = r-1; i >= 0; i--){
		
		if(left && best_j > 0){
			shift(displacement, cv::Point2i(best_j, i),'l');
		}
		else if(best_j < c-1){
			shift(displacement, cv::Point2i(best_j, i),'r');
		}

		seamed[i][best_j] = 255;
		best_j = phi[{i, best_j}];
	}
}









int subImage(cv::Mat1b const& border, cv::Rect& out){
	// longest border 
	int r = border.rows, c = border.cols;
	cv::Point3i top(0), bott(0), left(0), right(0);
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

		if(img(pix) == 0){
			out(pix) = 255;

			cv::Point neighbor = pix + offsets[visit(pix)-1];
			if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < c && neighbor.y < r && !visit(neighbor)){
				visit(neighbor) = visit(pix);
				test.push(neighbor);
			}
		}

		test.pop();        
	}

	int dilation_size = 5;
	auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
	cv::Mat tmp;
	cv::dilate( out, tmp, element );
	cv::GaussianBlur(tmp, out, cv::Size(11, 11), 0, 0, cv::BORDER_DEFAULT);
	cv::threshold(out, out, 0, 255, cv::THRESH_BINARY);
}
void localWarping(cv::Mat3b const& image, cv::Mat2i& displacement){
	int r = image.rows;
	int c = image.cols;
	int tp;

	cv::Mat1b img;
	cv::Mat1b border;
    cv::Mat1b border_tmp;
	cv::Mat1b seamed;
    cv::Mat1b tmp;

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

    while((tp = subImage(border_tmp, sub)) != -1){
        subseq.push_back(sub);
        tpseq.push_back(tp);

        cv::Rect window;
        cv::Point move;
        switch(tp){
            case 0:
            // top
            window = cv::Rect(sub.x, 0, sub.width, 1);
            move = cv::Point(0, 1);
            // cerr << "Up" << endl;
            break;
            case 1:
            // bottom
            window = cv::Rect(sub.x, r-1, sub.width, 1);
            move = cv::Point(0, -1);
            // cerr << "Down" << endl;
            break;
            case 2:
            // left
            window = cv::Rect(0, sub.y, 1, sub.height);
            move = cv::Point(1, 0);
            // cerr << "Left" << endl;
            break;
            case 3:
            // right
            window = cv::Rect(c-1, sub.y, 1, sub.height);
            move = cv::Point(-1, 0);
            // cerr << "Right" << endl;
            break;
        }
        while(cv::countNonZero(border_tmp(window))>0){
            cv::Rect src(window.x+move.x, window.y+move.y, window.width, window.height);
            border_tmp(src).copyTo(border_tmp(window));
            window = src;
        }
    }

    cerr << "Begin" << endl;
    cv::Sobel(img, grad_x, grad_x.type(), 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(img, grad_y, grad_y.type(), 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
    cv::normalize(grad_x, grad_x, -1., 1., cv::NORM_MINMAX, -1);
    cv::normalize(grad_y, grad_y, -1., 1., cv::NORM_MINMAX, -1);
    energy = grad_x.mul(grad_x) + grad_y.mul(grad_y);

    for(int i = 0; i < subseq.size(); i++){
        sub = subseq[i];
        auto img_sub = img(sub);
        auto border_sub = border(sub);
        auto disp_sub = displacement(sub);
        auto seamed_sub = seamed(sub);
        auto energy_sub = energy(sub);

        switch(tpseq[i]){
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
    }
    cerr << "Done" << endl;

    // local warping
    cv::Mat1b imgsrc;
    img.copyTo(imgsrc);
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            img[i][j] = imgsrc( cv::Point(j, i) + cv::Point(-displacement[i][j]) );
        }
    }
	
	cv::imshow("Display Image", img);
	cv::waitKey(0);
	cv::imshow("Display Image", seamed);
	cv::waitKey(0);
	cv::imwrite("out.png", img);
	cv::imwrite("seams.png", seamed);
}


void cropOuter(cv::Mat3b& img){
	int r = img.rows, c = img.cols;
	int up = 0, down = r - 1, left = 0, right = c - 1;
	bool f = false;
	for(int i = 0; !f && i < r; i++){
		for(int j = 0; !f && j < c; j++){
			if(img[i][j]!=DARKPIX){
				up = i;
				f = true;
			}
		}
	}
	f = false;
	for(int i = r-1; !f && i >= 0; i--){
		for(int j = 0; !f && j < c; j++){
			if(img[i][j]!=DARKPIX){
				down = i;
				f = true;
			}
		}
	}
	f = false;
	for(int j = 0; !f && j < c; j++){
		for(int i = up; !f && i <= down; i++){
			if(img[i][j]!=DARKPIX){
				left = j;
				f = true;
			}
		}
	}
	f = false;
	for(int j = c-1; !f && j >= 0; j--){
		for(int i = up; !f && i <= down; i++){
			if(img[i][j]!=DARKPIX){
				right = j;
				f = true;
			}
		}
	}
	img = img(cv::Rect(left, up, right - left + 1, down - up + 1));
}