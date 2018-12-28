#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <limits>
#include <queue>
#include "Array.h"

constexpr double inf = std::numeric_limits<double>::infinity();
const cv::Vec3b DARKPIX = cv::Vec3b({0,0,0});

// using namespace cv;
// using namespace std;
void cropOuter(cv::Mat3b& img);
void FindBorder(cv::Mat3b const& img, cv::Mat1b& out);
void SeamVertical(cv::Mat3b& img, cv::Mat1i& displacement, int ubound, int dbound);
void SeamHorizontal(cv::Mat3b& img, cv::Mat1b& border, cv::Mat1i& displacement, bool up);
int main(int argc, char** argv )
{
    cv::Mat3b image;
    image = cv::imread( argv[1], cv::IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cropOuter(image);
    // cerr << "Crop success." << endl;
    cv::Mat1b border;
    FindBorder(image, border);
    // cerr << "Border success." << endl;
    cv::Mat1i displace(image.size(), 0);
    // for(int i = 0; i < 25; i++){        
    //     SeamHorizontal(image, border, displace, i%2);
    // }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    cv::imshow("Display Image", border);
    cv::waitKey(0);
    cv::imwrite("image.png", image);
    cv::imwrite("border.png", border);

    return 0;
}


void cropOuter(cv::Mat3b& img){
    int r = img.rows, c = img.cols;
    int up = 0, down = r - 1, left = 0, right = c - 1;
    for(int i = 0; up<0 && i < r; i++){
        for(int j = 0; up<0 && j < c; j++){
            if(img[i][j]!=DARKPIX){
                up = i;
            }
        }
    }
    for(int i = r-1; down<0 && i >= 0; i--){
        for(int j = 0; down<0 && j < c; j++){
            if(img[i][j]!=DARKPIX){
                down = i;
            }
        }
    }
    for(int j = 0; left<0 && j < c; j++){
        for(int i = up; left<0 && i <= down; i++){
            if(img[i][j]!=DARKPIX){
                left = i;
            }
        }
    }
    for(int j = c-1; right<0 && j >= 0; j--){
        for(int i = up; right<0 && i <= down; i++){
            if(img[i][j]!=DARKPIX){
                right = i;
            }
        }
    }
    img = img(cv::Rect(left, up, right - left + 1, down - up + 1));
}
void FindBorder(cv::Mat3b const& img, cv::Mat1b& out){
    int r = img.rows, c = img.cols;
    out = cv::Mat1b::zeros(img.size());
    
    std::queue<cv::Point> test;
    // connected graph
    // find first dead pixel
    cv::Mat1b visit = cv::Mat1b::zeros(img.size());
    for(int y = 0; y < r; y++){
        test.push(cv::Point(0, y));
        test.push(cv::Point(c-1, y));
        visit[0][y] = 1;
        visit[c-1][y] = 1;
    }
    for(int x = 0; x < c; x++){
        test.push(cv::Point(x, 0));
        test.push(cv::Point(x, r-1));
        visit[x][0] = 1;
        visit[x][r-1] = 1;
    }

    while(!test.empty()){
        cv::Point pix = test.front();

        if(img(pix) == DARKPIX){
            out(pix) = 255;

            for(int x = -1; x < 2; x++){
                for(int y = -1; y < 2; y++){
                    if(x || y){
                        cv::Point neighbor(pix.x + x, pix.y + y);
                        if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < c && neighbor.y < r && !visit(neighbor)){
                            visit(neighbor) = 1;
                            test.push(neighbor);
                        }
                    }
                }
            }
        }

        test.pop();        
    }
}

DyArray<double> delta;
DyArray<int> phi;
void SeamHorizontal(cv::Mat3b& img, cv::Mat1b& border, cv::Mat1i& displacement, bool up){
    int r = img.rows, c = img.cols;
    cv::Mat1b grey;
    cv::Mat1f grad_x, grad_y, energy;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::Sobel(grey, grad_x, grad_x.type(), 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grey, grad_y, grad_y.type(), 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
    cv::normalize(grad_x, grad_x, -1, 1, cv::NORM_MINMAX, -1);
    cv::normalize(grad_y, grad_y, -1, 1, cv::NORM_MINMAX, -1);
    energy = grad_x.mul(grad_x) + grad_y.mul(grad_y);
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", energy);
    // cv::waitKey(0);
    
    delta.reshape({r, c});
    phi.reshape({r, c});    

    double penalty;
    cv::minMaxLoc(energy, NULL, &penalty);
    penalty *= c;

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
                delta[{i,j}] += penalty;
            }  
        }
    }

    int best_i = 0;
    
    for(int i = 1; i < r; i++){
        if(delta[{i,c-1}]<delta[{best_i,c-1}])
            best_i = i;
    }

    // back-tracking
    
    for(int j = c-1; j >= 0; j--){
        cv::Mat3b tmp;
        cv::Mat1b tmp2;
        // cout << "er" << endl;
        if(up && best_i > 0){
            img(cv::Rect(j, 1, 1, best_i)).copyTo(tmp);
            tmp.copyTo(img(cv::Rect(j, 0, 1, best_i)));

            border(cv::Rect(j, 1, 1, best_i)).copyTo(tmp2);
            tmp2.copyTo(border(cv::Rect(j, 0, 1, best_i)));

            displacement(cv::Rect(j, 0, 1, best_i)) += 1;            
        }
        else if(best_i < r-1){
            img(cv::Rect(j, best_i, 1, r-1-best_i)).copyTo(tmp);
            tmp.copyTo(img(cv::Rect(j, best_i+1, 1, r-1-best_i)));

            border(cv::Rect(j, best_i, 1, r-1-best_i)).copyTo(tmp2);
            tmp2.copyTo(border(cv::Rect(j, best_i+1, 1, r-1-best_i)));

            displacement(cv::Rect(j, best_i+1, 1, r-1-best_i)) += 1;
        }
        // cout << "er2" << endl;

        border[best_i][j] = 255;

        best_i = phi[{best_i, j}];
    }
}
void SeamVertical(cv::Mat3b& img, cv::Mat1i& displacement, int ubound, int dbound){
    int r = img.rows, c = img.cols;
    cv::Mat1b grey;
    cv::Mat1f grad_x, grad_y, energy;
    cv::cvtColor(img(cv::Rect(0,ubound, c, dbound - ubound + 1)), grey, cv::COLOR_BGR2GRAY);
    cv::Sobel(grey, grad_x, grad_x.type(), 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(grey, grad_y, grad_y.type(), 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
    
    energy = grad_x.mul(grad_x) + grad_y.mul(grad_y);
    
    delta.reshape({r, c});
    phi.reshape({r, c});
    delta.clear(inf);
    phi.clear(-1);

    // r = 0
    for(int j = 0; j < c; j++){
        delta[{0,j}] = energy[0][j];
    }

    // r > 0
    for(int i = 1; i < r; i++){
        for(int j = 0; j < c; j++){
            if(img[i][j] == cv::Vec3b({0,0,0})){
                delta[{i,j}] = inf;
            }
            else{
                double mn = delta[{i-1,j}];
                phi[{i,j}] = j;
                if(j>0 && delta[{i-1,j-1}]<mn){
                    mn = delta[{i-1,j-1}];
                    phi[{i,j}] = j - 1;
                }
                if(j<c-1 && delta[{i-1,j+1}]<mn){
                    mn = delta[{i-1,j+1}];
                    phi[{i,j}] = j + 1;
                }
                delta[{i,j}] = energy[i][j] + mn;

            }           
        }
    }

    int best_j = 0;
    for(int j = 1; j < c; j++){
        if(delta[{r-1,j}]<delta[{r-1,best_j}])
            best_j = j;
    }

    // back-tracking
    for(int i = r-1; i >= 0; i--){
        img(cv::Rect(best_j, i, c-1-best_j, 1)).copyTo(img(cv::Rect(best_j+1, i, c-1-best_j, 1)));
        best_j = phi[{i, best_j}];
    }
}
