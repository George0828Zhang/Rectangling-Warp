#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cassert>
#include <queue>
#include <algorithm>
#include <cmath>
#include "LineSeg.h"
#define ARMA_DONT_USE_WRAPPER
#define ARMA_NO_DEBUG
#include <armadillo>
#include <unistd.h>
#include "rectanglize.hpp"
#define SHOWIMGS

constexpr float inf = 100000000;
constexpr float DOWNSCALE = 2;
constexpr float LAMBDA_L = 100;
constexpr int HORIZONTAL = 0;
constexpr int VERTICAL = 1;

void Converge(
	int Rows, int Cols,
	std::vector<cv::Vec4i> const& Quads, 
	std::vector<int> const& Boundary_types, 
	std::vector<cv::Point> vertex_map, 
	std::vector<cv::Vec4i> const& seg_lines,
	cv::Mat3b& img );
void Generating_C_mut_vertices_to_e_Matrix(cv::Vec4i const& Quad, std::vector<cv::Point> const& vertexMap, LineSeg input_line, cv::Mat1f& out);
void GeneratingQuad(cv::Vec4i const& Quad, std::vector<cv::Point> const& vertexMap, LineSeg Edges, cv::Mat1f& out);
void DrawMatrix(
	int Rows, int Cols,
	std::vector<cv::Vec4i> const& Quad, 
	std::vector<int> const& Boundary_types, 
	std::vector<cv::Point> const& vertex_map, 
	std::vector<cv::Vec4i> const& seg_lines,
	cv::Mat3b const& img );

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

	cv::Mat3b big_image = image.clone();
	cv::resize(image, image, cv::Size(), 1/DOWNSCALE, 1/DOWNSCALE, cv::INTER_LINEAR);
	cv::Mat2i displace;
	cv::Mat1b border;
	localWarping(image, border, displace);

	std::vector<cv::Point> Mesh;
	std::vector<cv::Vec4i> Quad;
	std::vector<int> Boundary_types;
	UnwarpGrid(displace, Mesh, Quad, Boundary_types, 10);

	std::vector<cv::Vec4i> lines;
	GetLines(image, lines);

	cv::Mat3b tmp(image.clone());
	// for(int i = 0; i < Mesh.size(); i++){
 //        auto p = Mesh[i];
	// 	cv::circle(tmp, p, 2.5, cv::Scalar(255), -1);
 //        if(p==cv::Point(0, 0))
 //            std::cout << "i=" << i << std::endl;
	// }
	for(auto& qd : Quad){
		cv::Point vA(Mesh[qd[0]]);
		cv::Point vB(Mesh[qd[1]]);
		cv::Point vC(Mesh[qd[2]]);
		cv::Point vD(Mesh[qd[3]]);
        cv::line( tmp, vA, vB, cv::Scalar(0,0,255), 1, 4 );
        cv::line( tmp, vA, vC, cv::Scalar(0,0,255), 1, 4 );
        cv::line( tmp, vC, vD, cv::Scalar(0,0,255), 1, 4 );
        cv::line( tmp, vB, vD, cv::Scalar(0,0,255), 1, 4 );
    }
	// for( auto& l : lines )
	// {
	//     if(border(cv::Point(l[0], l[1]))==0 || border(cv::Point(l[2], l[3]))==0)
	//         cv::line( image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 2, 8 );
	// }

#ifdef SHOWIMGS
	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	cv::imshow("Display Image", tmp);
	cv::waitKey(0);
#endif

	DrawMatrix(image.rows, image.cols, Quad, Boundary_types, Mesh, lines, big_image);
	//Converge(image.rows, image.cols, Quad, Boundary_types, Mesh, lines, image);

	return 0;
}
/*
void Converge(
	int Rows, int Cols,
	std::vector<cv::Vec4i> const& Quads, 
	std::vector<int> const& Boundary_types, 
	std::vector<cv::Point> vertex_map, 
	std::vector<cv::Vec4i> const& seg_lines,
	cv::Mat3b& img )
{
	int N = Quads.size();
	int M = seg_lines.size();    
	int K = 0;
	int T = 2*vertex_map.size();
	vector<cv::Mat1f> A_queue;
	vector<float> b_queue;
	for(int i = 0; i < M ; i++)
	{
		//C = R x e_hat x  e_hat_pseudo_inv x R_transpose - I
		//  = M - I
		//to min Ce:
		//Ce = (M-I)e 
		//make sure e[0] is the top left most and e[1] is right most

		
		//std::vector<cv::Vec4i> const& seg_lines
		
		for(int j =0 ;j < N; j++)
		{
			cv::Mat1f output_mat = cv::Mat1f::zeros(2,T);
			LineSeg my_line(cv::Point2f(seg_lines[j][0],seg_lines[j][1]), cv::Point2f(seg_lines[j][2],seg_lines[j][3]));
			Generating_C_mut_vertices_to_e_Matrix(Quads[j], vertex_map, my_line, output_mat);
			//cout << "ok" << endl;
			for(int k = 0;k < vertexMap.size();k++)
			{
				tpx = vertex_map[k].x;
				tpy = vertex_map[k].y;
				vertex_map[k].x += output_mat[0][2*k]*tpx;
				vertex_map[k].y += output_mat[0][2*k+1]*tpy;

				vertex_map[k].x += output_mat[0][2*k]*tpx;
				vertex_map[k].y += output_mat[0][2*k+1]*tpy;
			}
			output_mat = LAMBDA_L * output_mat;	
			if(countNonZero(output_mat.row(0)))
			{
				A_queue.push_back(output_mat.row(0));
				b_queue.push_back(0);
			}
			if(countNonZero(output_mat.row(1)))
			{
				A_queue.push_back(output_mat.row(1));
				b_queue.push_back(0);
			}	
		}  
	}
	
	cv::Mat1f result;
	int rows = A_queue.size();
	cv::Mat1f A = cv::Mat1f::zeros(rows, T); 
	cv::Mat1f b = cv::Mat1f::zeros(rows,1);
	cout << "rows" << rows << endl;
	cout << "cols" << T << endl;
	for(int i = 0 ; i < rows ; i++)
	{
		A_queue[i].copyTo(A(cv::Rect(0,  i, T, 1)));
		b[i][0] = b_queue[i];
	}
	cv::solve(A, b, result, cv::DECOMP_SVD);
	cout << result.size << endl;
	for(int i = 0; i < M ; i++)
	{

	}
}
*/
void general_case(LineSeg input_line, cv::Point& p ,int ct, LineSeg const& edgeAB, std::vector<cv::Point> const&  vertexMap,
 const int high_prior_index, const int low_prior_index, cv::Mat1f& vertices_to_e)
{
	std::cout << "general" << std::endl;
	p = input_line.IntersectionPointWith(edgeAB);
	float inv = std::abs( 1.0f/(vertexMap[high_prior_index].x - vertexMap[low_prior_index].x) );
	if(!std::isnan(inv) && !std::isinf(inv)){
		vertices_to_e[ct][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - p.x) * inv;
		vertices_to_e[ct][2 * low_prior_index] = std::abs(p.x - vertexMap[high_prior_index].x) * inv;
		// cout << vertices_to_e[ct][2 * high_prior_index] << endl;
		// cout << vertices_to_e[ct][2 * low_prior_index] << endl;
		// sleep(1);
	}
		
	//assert(!std::isnan(inv_AB)&&!std::isinf(inv_AB));


	inv = std::abs( 1.0f/(vertexMap[high_prior_index].y - vertexMap[low_prior_index].y) );
	if(!std::isnan(inv) && !std::isinf(inv))
	{
		vertices_to_e[ct][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - p.y) * inv;
		vertices_to_e[ct][2 * low_prior_index + 1] = std::abs(p.y - vertexMap[high_prior_index].y) * inv;
		// cout <<  vertices_to_e[ct][2 * high_prior_index + 1] << endl;
		// cout <<  vertices_to_e[ct][2 * low_prior_index + 1] << endl;
		// sleep(1);
	}
		
	//assert(!std::isnan(inv_AB)&&!std::isinf(inv_AB));

}

void parrallel_case(LineSeg const& input_line, std::vector<cv::Point> const& vertexMap, int high_prior_index, int low_prior_index, 
cv::Mat1f& vertices_to_e, cv::Mat1f& C ,cv::Mat1f& R, cv::Mat1f& e_hat, int mode)
{
	std::cout << "parrallel" << std::endl;
	cv::Mat1f R_transpose = cv::Mat1f::zeros(2, 2);
	transpose(R, R_transpose);
	cv::Point left,right;
	//for mode is vertical, left = up, right = down.
	if(mode == HORIZONTAL)
	{
		if(input_line.origin.x < input_line.end.x)
		{
			left = input_line.origin;
			right = input_line.end;
		}
		else
		{
			right = input_line.origin;
			left = input_line.end;
		}

		if(left.x < vertexMap[high_prior_index].x)
		{
			left.x = vertexMap[high_prior_index].x;
		}
		if(right.x > vertexMap[low_prior_index].x)
		{
			right.x = vertexMap[low_prior_index].x;
		}
	}
	else if(mode == VERTICAL)
	{
		if(input_line.origin.y < input_line.end.y)
		{
			left = input_line.origin;
			right = input_line.end;
		}
		else
		{
			right = input_line.origin;
			left = input_line.end;
		}
		
		if(left.y < vertexMap[high_prior_index].y)
		{
			left.y = vertexMap[high_prior_index].y;
		}
		if(right.y > vertexMap[low_prior_index].y)
		{
			right.y = vertexMap[low_prior_index].y;
		}
	}
		
	e_hat[0][0] = left.x;
	e_hat[1][0] = left.y;
	e_hat[0][1] = right.x;
	e_hat[1][1] = right.y;

	C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
	float inv = std::abs( 1.0f/(vertexMap[low_prior_index].x - vertexMap[high_prior_index].x) );
	if(!std::isnan(inv) && !std::isinf(inv))
	{
		vertices_to_e[0][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - left.x)*inv;
		vertices_to_e[0][2 * low_prior_index] = std::abs(vertexMap[high_prior_index].x - left.x)*inv;   
		vertices_to_e[1][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - right.x)*inv;
		vertices_to_e[1][2 * low_prior_index] = std::abs(vertexMap[high_prior_index].x - right.x)*inv;
	}


	inv = std::abs( 1.0f/(vertexMap[low_prior_index].y - vertexMap[high_prior_index].y) );
	if(!std::isnan(inv) && !std::isinf(inv))
	{
		vertices_to_e[0][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - left.y)*inv;
		vertices_to_e[0][2 * low_prior_index + 1] = std::abs(vertexMap[high_prior_index].y - left.y)*inv;
		vertices_to_e[1][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - right.y)*inv;
		vertices_to_e[1][2 * low_prior_index + 1] = std::abs(vertexMap[high_prior_index].y - right.y)*inv;
	}


}


void Generating_C_mut_vertices_to_e_Matrix(cv::Vec4i const& Quad, std::vector<cv::Point> const& vertexMap, LineSeg input_line, cv::Mat1f& out)
{
	int intersect_ct = 0;
	int vertices_nm  = vertexMap.size();
	//Ray will only intersect two points or one point or even no point
	float theta = atan(input_line.getTan());
	float rotate_angle = find_closest_in_bin(theta, 50);
	cv::Mat1f R = cv::Mat1f::zeros(2, 2);
	cv::Mat1f R_transpose = cv::Mat1f::zeros(2, 2);
	transpose(R, R_transpose);
	R[0][0] = cos(rotate_angle);
	R[0][1] = -sin(rotate_angle);
	R[1][0] = sin(rotate_angle);
	R[1][1] = sin(rotate_angle);


	cv::Mat1f e_hat = cv::Mat1f::zeros(2, 2);
	cv::Mat1f C = cv::Mat1f::zeros(2, 2);
	cv::Mat1f vertices_to_e = cv::Mat1f::zeros(2, 2 * vertices_nm);

	cv::Point intersect_points[4];
	//make sure when doing interpolation the index with higher is v[1], the lower one is v[0] for e and e_hat
	//horizontal
	LineSeg edgeAB(vertexMap[Quad[0]], vertexMap[Quad[1]]);
	LineSeg edgeCD(vertexMap[Quad[2]], vertexMap[Quad[3]]);
	//cout << "before edgeAB" << endl;
	if( input_line.isIntersectionLine(edgeAB))
	{
		int high_prior_index,low_prior_index;
		high_prior_index = Quad[0];
		low_prior_index = Quad[1];
		if( input_line.is_parrallel(edgeAB))
		{    
				 
			parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat, HORIZONTAL);
			out = C*vertices_to_e;
			return;
		}
		else
		{
			
			general_case(input_line, intersect_points[intersect_ct], intersect_ct, edgeAB, vertexMap, high_prior_index, low_prior_index, vertices_to_e);
			intersect_ct++;
		}

	}
	//cout << "edgeAB finish detect" << endl;
	if ( input_line.isIntersectionLine(edgeCD) )
	{
		//cout << "in CD" << endl;
		int high_prior_index,low_prior_index;
		high_prior_index = Quad[2];
		low_prior_index = Quad[3];
		if( input_line.is_parrallel(edgeCD))
		{         
			parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat, HORIZONTAL);
			out = C*vertices_to_e;
			return;
		}
		else
		{
			general_case(input_line, intersect_points[intersect_ct], intersect_ct, edgeCD, vertexMap, high_prior_index, low_prior_index, vertices_to_e);
			intersect_ct++;
		}   
	}
	//vertical
	LineSeg edgeAC(vertexMap[Quad[0]], vertexMap[Quad[2]]);
	LineSeg edgeBD(vertexMap[Quad[1]], vertexMap[Quad[3]]);
	if ( input_line.isIntersectionLine(edgeAC) )
	{
		//cout << "in BD" << endl;
		int high_prior_index,low_prior_index;
		high_prior_index = Quad[0];
		low_prior_index = Quad[2];
		if( input_line.is_parrallel(edgeAC))
		{         
			parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat, VERTICAL);
			out = C*vertices_to_e;
			return;
		}
		else
		{
			general_case(input_line, intersect_points[intersect_ct], intersect_ct,edgeAC, vertexMap, high_prior_index, low_prior_index, vertices_to_e);
			intersect_ct++;
		}     
	}

	if ( input_line.isIntersectionLine(edgeBD) )
	{
		//cout << "in AC" << endl;
		int high_prior_index,low_prior_index;
		high_prior_index = Quad[1];
		low_prior_index = Quad[3];
		if( input_line.is_parrallel(edgeBD))
		{         
			parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat, VERTICAL);
			out = C*vertices_to_e;
			return;
		}
		else
		{
			general_case(input_line, intersect_points[intersect_ct], intersect_ct, edgeBD, vertexMap, high_prior_index, low_prior_index, vertices_to_e);
			intersect_ct++;
		}     
	}
	//cout << "before my assert" << endl;
	
	// if(intersect_ct!=0){
	// 	cout << "intersect_ct " <<  intersect_ct << endl;
	// 	sleep(1);
	// }
	if(intersect_ct>2)
		std::cout << "intersect_ct >= 2" << std::endl;
		
	
	assert(intersect_ct>=0); 
	//assert(intersect_ct<=2);
	
	if(intersect_ct >= 2)
	{
		std::cout << "intersect_ct " <<  intersect_ct << std::endl;
		if(intersect_points[0].x > intersect_points[1].x || 
		((intersect_points[0].x==intersect_points[1].x)&&(intersect_points[0].y < intersect_points[1].y)) )
		{
			cv::Mat1f temp = cv::Mat1f::zeros(1, vertices_nm);
			//cout << "hello" << endl;
			for(int i = 0; i < vertices_nm; i++)
			{
				temp[0][i] = vertices_to_e[0][i];
			}
			//cout << "fell" << endl;
			for(int i = 0; i < vertices_nm; i++)
			{
				vertices_to_e[0][i] = vertices_to_e[1][i];
				vertices_to_e[1][i] = temp[0][i];
			}
			//cout << "bad" << endl;
			e_hat[0][0] = intersect_points[1].x;
			e_hat[1][0] = intersect_points[1].y;
			e_hat[0][1] = intersect_points[0].x;
			e_hat[1][1] = intersect_points[0].y;

		}
		else{
			e_hat[0][0] = intersect_points[0].x;
			e_hat[1][0] = intersect_points[0].y;
			e_hat[0][1] = intersect_points[1].x;
			e_hat[1][1] = intersect_points[1].y;
		}
		//cout << "before C" << endl;
		C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
		out = C*vertices_to_e;
		//cout << "out is filled" << endl;
	}
	else if(intersect_ct == 1)
	{
		return;
		/*
		cout << "intersect_ct == 1" << endl;
		for(int i =0 ;i< vertices_nm; i++)
		{
			if(vertices_to_e[1][i]!=0.0)
				vertices_to_e[0][i] = vertices_to_e[1][i];
			else if(vertices_to_e[0][i]!=0.0)
				vertices_to_e[1][i] = vertices_to_e[0][i];
		}
		cout << "intersect_ct == 1 follow" << endl;
		e_hat[0][0] = intersect_points[0].x;
		e_hat[1][0] = intersect_points[0].y;
		e_hat[0][1] = intersect_points[0].x;
		e_hat[1][1] = intersect_points[0].y;
		//e_hat can't be inverse
		//C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
		C = cv::Mat1f::eye(2, 2);
		out = C*vertices_to_e;
		*/
	}
	//cout << "nothing" << endl;
}

void DrawMatrix(
	int Rows, int Cols,
	std::vector<cv::Vec4i> const& Quads, 
	std::vector<int> const& Boundary_types, 
	std::vector<cv::Point> const& vertex_map, 
	std::vector<cv::Vec4i> const& seg_lines,
	cv::Mat3b const& img )
{
	int N = Quads.size();
	int M = 0;//seg_lines.size();
	int V = vertex_map.size();
	int T = 2*vertex_map.size();	
	int columns = T;

	std::vector<cv::Mat1f> A_queue;
	std::vector<float> b_queue;


	// std::cout <<  << "fucku" << std::endl;

	// Shape Preservation NxT	
	arma::mat Aq = {
		{0,0,1,0},
		{0,0,0,1},
		{0,0,1,0},
		{0,0,0,1},
		{0,0,1,0},
		{0,0,0,1},
		{0,0,1,0},
		{0,0,0,1},
	};
	for(int i = 0; i < N; i++){
		cv::Vec4i qd = Quads[i];

		for(int v = 0; v < 4; v++){
			int index = qd[v];
			int x = vertex_map[index].x;
			int y = vertex_map[index].y;

			Aq(2*v, 0) = x;
			Aq(2*v, 1) = -y;
			// Aq(2*v, 2) = 1;
			// Aq(2*v, 3) = 0;

			Aq(2*v+1, 0) = y;
			Aq(2*v+1, 1) = x;
			// Aq(2*v+1, 2) = 0;
			// Aq(2*v+1, 3) = 1;
		}

		arma::mat AqAq_pI = Aq * pinv(Aq) - arma::eye<arma::mat>(8, 8);
		
		for(int j = 0; j < 8 ;j ++)
		{
			cv::Mat1f forbigA = cv::Mat1f::zeros(1, T);
			int ct = 0;
			for(int v = 0; v < 4; v++)
			{
				int index = qd[v];
				// forbigA[0][2*index] = AqAq_pI[j][ct++];
				// forbigA[0][2*index+1] = AqAq_pI[j][ct++];
				forbigA[0][2*index] = AqAq_pI(j,ct++);
				forbigA[0][2*index+1] = AqAq_pI(j,ct++);
			}
			if(countNonZero(forbigA)>0)
			{
				A_queue.push_back(forbigA);
				b_queue.push_back(0);
			}
		}
		
		
		//for_bigA.copyTo(A(cv::Rect(0, i, T, 1)));
	}
	std::cout << "N:" << N << std::endl;
	std::cout << "8*N:" << 8*N << std::endl;
	std::cout << A_queue.size() << std::endl;
	std::cout << "Shape Preservation ok" << std::endl;
	// Line Preservation (MxNx2)xT
	int ct = 0;
	for(int i = 0; i < N ; i++)
	{
		//C = R x e_hat x  e_hat_pseudo_inv x R_transpose - I
		//  = M - I
		//to min Ce:
		//Ce = (M-I)e 
		//make sure e[0] is the top left most and e[1] is right most

		
		//std::vector<cv::Vec4i> const& seg_lines
		
		for(int j =0 ;j < M; j++)
		{
			cv::Mat1f output_mat = cv::Mat1f::zeros(2,T);
			LineSeg my_line(cv::Point2f(seg_lines[j][0],seg_lines[j][1]), cv::Point2f(seg_lines[j][2],seg_lines[j][3]));
			Generating_C_mut_vertices_to_e_Matrix(Quads[i], vertex_map, my_line, output_mat);
			//cout << "ok" << endl;
			output_mat = LAMBDA_L * output_mat;
			
			if(countNonZero(output_mat.row(0)))
			{
				A_queue.push_back(output_mat.row(0));
				b_queue.push_back(0);
				ct++;
			}
			if(countNonZero(output_mat.row(1)))
			{
				A_queue.push_back(output_mat.row(1));
				b_queue.push_back(0);
				ct++;
			}
			
		}  
	}
	std::cout << "Line Preservation has: " << ct << " rows" << std::endl;
	std::cout << "Line Preservation ok" << std::endl;

	// Boundary Constraint Kx8N, K < N
	for(int i = 0 ; i < N; i++){
		cv::Vec4i qd = Quads[i];

		for(int v = 0; v < 4; v++){
			int v_index = qd[v];


			if(Boundary_types[v_index]<TOP)
				continue;

			if(Boundary_types[v_index]>RIGHT)
				continue;


			int x = vertex_map[v_index].x;
			int y = vertex_map[v_index].y;
			
			int t = Boundary_types[v_index];
			cv::Mat1f temp_mat = cv::Mat1f::zeros(1,T);
		
			if(t==TOP)
			{
				// up
				
				temp_mat[0][ 2 * v_index + 1] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}	
			}
			if(t==BOTTOM)
			{
				// down
				
				temp_mat[0][ 2 * v_index + 1] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back((Rows - 1)*inf);
				}
			}
			if(t==LEFT)
			{
				// left
				
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}

			}
			if(t==RIGHT)
			{
				// right
				
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back((Cols - 1)*inf);
				}
			}

			if(t==CORNER_A)
			{
				// up & left
				temp_mat[0][ 2 * v_index + 1 ] = inf;
				if(countNonZero(temp_mat))
				{// y = 0
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}
				temp_mat = cv::Mat1f::zeros(1,T);
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{// x = 0
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}
			}
			if(t==CORNER_B)
			{
				// up & right				
				temp_mat[0][ 2 * v_index + 1 ] = inf;
				if(countNonZero(temp_mat))
				{// y = 0
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}
				temp_mat = cv::Mat1f::zeros(1,T);
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{// x = c - 1
					A_queue.push_back(temp_mat);
					b_queue.push_back((Cols - 1)*inf);
				}
			}
			if(t==CORNER_C)
			{
				// down & left			
				temp_mat[0][ 2 * v_index + 1 ] = inf;
				if(countNonZero(temp_mat))
				{// y = r - 1
					A_queue.push_back(temp_mat);
					b_queue.push_back((Rows - 1)*inf);
				}
				temp_mat = cv::Mat1f::zeros(1,T);
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{// x = 0
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}

			}
			if(t==CORNER_D)
			{
				// down & right	
				temp_mat[0][ 2 * v_index + 1 ] = inf;
				if(countNonZero(temp_mat))
				{// y = r - 1
					A_queue.push_back(temp_mat);
					b_queue.push_back((Rows - 1)*inf);
				}
				temp_mat = cv::Mat1f::zeros(1,T);
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{// x = c - 1
					A_queue.push_back(temp_mat);
					b_queue.push_back((Cols - 1)*inf);
				}
			}
			
		
			
		}
	}
	
	int rows = A_queue.size();
	cv::Mat1f A = cv::Mat1f::zeros(rows, T); 
	cv::Mat1f b = cv::Mat1f::zeros(rows,1);
	for(int i = 0 ; i < rows ; i++)
	{
		A_queue[i].copyTo(A(cv::Rect(0,  i, T, 1)));
		b[i][0] = b_queue[i];
	}
	std::cerr << "ready for solve!" << std::endl;
	std::cerr << "A rows: " << rows << std::endl;
	std::cerr << "A cols: " << T << std::endl;


	arma::mat armaA(rows, columns);
	arma::vec armab(rows);
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < columns; j++){
			armaA(i,j) = (double)A[i][j];
		}
		armab[i] = (double)b[i][0];
	}
	arma::vec res = arma::solve(armaA, armab) * DOWNSCALE;
	std::cerr << "Solve completed!" << std::endl;


	std::cout << "Making warped vertex map ..." << std::endl;
	std::vector<cv::Point2d> vertex_map_warped(V);
	for(int i = 0; i < V; i++){
		assert(2*i+1 < res.size());

		vertex_map_warped[i].x = res[2*i];
		vertex_map_warped[i].y = res[2*i+1];
		if(Boundary_types[i]==CORNER_A){
			vertex_map_warped[i] = cv::Point2d(0,0);
		}
		else if(Boundary_types[i]==CORNER_B){
			vertex_map_warped[i] = cv::Point2d(img.cols-1,0);
		}else if(Boundary_types[i]==CORNER_C){
			vertex_map_warped[i] = cv::Point2d(0,img.rows-1);
		}else if(Boundary_types[i]==CORNER_D){
			vertex_map_warped[i] = cv::Point2d(img.cols-1,img.rows-1);
		}
	}


	std::cout << "Finding Quad indices ..." << std::endl;
	// Interpolation: Find corresponding Quad index
	cv::Mat1b visit = cv::Mat1d::zeros(img.size());
	cv::Mat1i quadex = cv::Mat1i::zeros(img.size());
	for(int qindex = 0; qindex < N; qindex++ ){
		cv::Vec4i qd = Quads[qindex];
		// cv::Point2f vA = vertex_map_warped[qd[0]];
		// cv::Point2f vB = vertex_map_warped[qd[1]];
		// cv::Point2f vC = vertex_map_warped[qd[2]];
		// cv::Point2f vD = vertex_map_warped[qd[3]];
		cv::Point vA(vertex_map_warped[qd[0]]);
		cv::Point vB(vertex_map_warped[qd[1]]);
		cv::Point vC(vertex_map_warped[qd[2]]);
		cv::Point vD(vertex_map_warped[qd[3]]);
		cv::Point poly[4] = {vA, vB, vD, vC};
		cv::fillConvexPoly(visit, poly, 4, cv::Scalar(1), 8, 0);
		cv::fillConvexPoly(quadex, poly, 4, cv::Scalar(qindex), 8, 0);

		// cv::line( visit, vA, vB, cv::Scalar(1), 1, cv::LINE_8 );
		// cv::line( visit, vA, vC, cv::Scalar(1), 1, cv::LINE_8 );
		// cv::line( visit, vB, vD, cv::Scalar(1), 1, cv::LINE_8 );
		// cv::line( visit, vC, vD, cv::Scalar(1), 1, cv::LINE_8 );

		// cv::line( quadex, vA, vB, cv::Scalar(qindex), 1, cv::LINE_8 );
		// cv::line( quadex, vA, vC, cv::Scalar(qindex), 1, cv::LINE_8 );
		// cv::line( quadex, vB, vD, cv::Scalar(qindex), 1, cv::LINE_8 );
		// cv::line( quadex, vC, vD, cv::Scalar(qindex), 1, cv::LINE_8 );

		// std::queue<cv::Point> test;
		// cv::Point mid((vA + vB + vC)/3.);
		// if(!visit(mid)){
		// 	visit(mid) = 1;
		// 	quadex(mid) = qindex;
		// 	test.push(mid);
		// }
		
		// cv::Point offsets[4] = {{1, 0},{-1, 0},{0, 1},{0, -1}};
		// while(!test.empty()){
		// 	cv::Point pix = test.front();

		// 	for(int nei = 0; nei < 4; nei++){
		// 		cv::Point neighbor = pix + offsets[nei];
				
		// 		if(neighbor.x >= 0 && neighbor.y >= 0 && neighbor.x < img.cols && neighbor.y < img.rows && !visit(neighbor)){
		// 			visit(neighbor) = 1;
		// 			quadex(neighbor) = qindex;
		// 			test.push(cv::Point(neighbor));
		// 		}
		// 	}       
		// 	test.pop();
		// }
	}


	// cv::Mat1i normed;
	// cv::normalize(quadex, normed, 0, 255, cv::NORM_MINMAX, -1);
	// cv::imshow("Display Image", cv::Mat1b(visit*255));
	// cv::waitKey(0);

	
	std::cout << "Final interpolation on pixels ..." << std::endl;
	cv::Mat3b unwarped_img(img.size());
	for(int x = 0; x < img.cols; x++){
		for(int y = 0; y < img.rows; y++){
			cv::Point2d pix(x, y);
			int qindex = quadex(pix);

			cv::Vec4i qd = Quads[qindex];
			cv::Point2d vA = vertex_map_warped[qd[0]];
			cv::Point2d vB = vertex_map_warped[qd[1]];
			cv::Point2d vC = vertex_map_warped[qd[2]];
			cv::Point2d vD = vertex_map_warped[qd[3]];
			cv::Point2d srcA(vertex_map[qd[0]]*DOWNSCALE);
			cv::Point2d srcB(vertex_map[qd[1]]*DOWNSCALE);
			cv::Point2d srcC(vertex_map[qd[2]]*DOWNSCALE);
			cv::Point2d srcD(vertex_map[qd[3]]*DOWNSCALE);


			const double lbd = 0.2;
			arma::mat P = {
				{ vA.x, vB.x, vC.x, vD.x }, 
				{ vA.y, vB.y, vC.y, vD.y },
				{  inf,  inf,  inf,  inf },
				{  lbd,  0.0,  0.0,  0.0 },
				{  0.0,  lbd,  0.0,  0.0 },
				{  0.0,  0.0,  lbd,  0.0 },
				{  0.0,  0.0,  0.0,  lbd },
			};

			arma::vec constraint = {
				(double)x,
				(double)y,
				(double)inf,
				lbd,
				lbd,
				lbd,
				lbd
			};

			arma::mat interp = arma::solve(P, constraint, arma::solve_opts::fast);
			
			arma::mat srcP = {
				{ srcA.x, srcB.x, srcC.x, srcD.x }, 
				{ srcA.y, srcB.y, srcC.y, srcD.y } };

			arma::vec src_v = srcP * interp;

			// arma::mat P = {
			// 	{ 0, 0 },
			// 	{ vB.x - vA.x, vB.y - vA.y },
			// 	{ vC.x - vA.x, vC.y - vA.y },
			// 	{ vD.x - vA.x, vD.y - vA.y }
			// };

			// arma::mat const b_1 = {
			// 	{ 0, 0 },
			// 	{ 1, 0 },
			// 	{ 0, 1 },
			// 	{ 1, 1 } };

			// arma::mat S_trans = arma::solve(P, b_1);

			// arma::vec lm = arma::vec({ x - vA.x, y - vA.y }) * S_trans;

			// cv::Point2d displacement = \
			// (1-lm[0])*(1-lm[1])*(srcA-vA) \
			// + lm[0]*(1-lm[1])*(srcB-vB) \
			// + (1-lm[0])*lm[1]*(srcC-vC) \
			// + lm[0]*lm[1]*(srcD-vD);

			// double src_v[] = {x+displacement.x, y+displacement.y};

			if(src_v[0]>=0 && src_v[0] < img.cols && src_v[1] >= 0 && src_v[1] < img.rows){
				double xbase = std::floor(src_v[0]);
				double ybase = std::floor(src_v[1]);
				double alpha = src_v[0] - xbase;
				double beta = src_v[1] - ybase;

				cv::Point topleft(xbase, ybase);
				cv::Point topright(xbase+1==img.cols ? xbase : (xbase+1), ybase);
				cv::Point botleft(xbase, ybase+1==img.rows ? ybase : (ybase+1));
				cv::Point botright(xbase+1==img.cols ? xbase : (xbase+1), ybase+1==img.rows ? ybase : (ybase+1));

				cv::Vec3f emit = \
				  (1-alpha)*(1-beta)*img(topleft) \
				+ (alpha)*(1-beta)*img(topright) \
				+ (1-alpha)*(beta)*img(topleft) \
				+ (alpha)*(beta)*img(topleft);

				unwarped_img(pix) = cv::Vec3b(emit);
			}
		}
	}
	std::cout << "Unwarp complete." << std::endl;

	for(int i = 0; i < V; i++){
		if(Boundary_types[i]>=CORNER_A){
			cv::circle(unwarped_img, cv::Point(vertex_map_warped[i]), 3, cv::Scalar(0, 0, 255), -1);
		}
	}
	std::cout << "show picture" << std::endl;
	cv::imwrite("results/global.png", unwarped_img);
#ifdef SHOWIMGS
	cv::imshow("Display Image", unwarped_img);
	cv::waitKey(0);
#endif
}
