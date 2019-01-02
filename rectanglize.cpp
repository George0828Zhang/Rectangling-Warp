#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <cassert>
// #include <limits>
#include <queue>
#include <algorithm>
#include <cmath>
#include <string>
#include "LineSeg.h"
//#include <armadillo>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>


constexpr float inf = 100000000;//std::numeric_limits<double>::infinity();
constexpr float LAMBDA_L = 100;
constexpr float DARKTHRES = 2.5;
const cv::Vec3b DARKPIX = cv::Vec3b({0,0,0});

using namespace std;

void Generating_C_mut_vertices_to_e_Matrix(cv::Vec4i const& Quad, std::vector<cv::Point> const& vertexMap, LineSeg input_line, cv::Mat1f& out);
void GeneratingQuad(cv::Vec4i const& Quad, std::vector<cv::Point> const& vertexMap, LineSeg Edges, cv::Mat1f& out);
void cropOuter(cv::Mat3b& img);
void localWarping(cv::Mat3b const& img, cv::Mat1b& border, cv::Mat2i& displacement);
void UnwarpGrid(cv::Mat2i const& displacement, std::vector<cv::Point>& vertex_map, std::vector<cv::Vec4i>& quads, std::vector<int>& bound_types, int rowdiv);
void GetLines(cv::Mat3b const& img, std::vector<cv::Vec4i>& out);
void DrawMatrix(
    int Rows, int Cols,
    std::vector<cv::Vec4i> const& Quad, 
    std::vector<int> const& Boundary_types, 
    std::vector<cv::Point> const& vertex_map, 
    std::vector<cv::Vec4i> const& seg_lines,
    cv::Mat3b& img );

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

	cv::Mat2i displace = cv::Mat2i::zeros(image.size());
    cv::Mat1b border;
	localWarping(image, border, displace);

	std::vector<cv::Point> Mesh;
    std::vector<cv::Vec4i> Quad;
    std::vector<int> Boundary_types;
	UnwarpGrid(displace, Mesh, Quad, Boundary_types, 20);

    std::vector<cv::Vec4i> lines;
    GetLines(image, lines);

    cv::Mat3b tmp(image);
    for(auto& p : Mesh){
        cv::circle(tmp, p, 2.5, cv::Scalar(255), -1);
    }
    // for( auto& l : lines )
    // {
    //     if(border(cv::Point(l[0], l[1]))==0 || border(cv::Point(l[2], l[3]))==0)
    //         cv::line( image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 2, 8 );
    // }


    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cout << Quad.size() << endl;
    // int p;
    //while(cin >> p){
    //     cv::circle(image, Mesh[Quad[p][0]], 2.5, cv::Scalar(255), -1);
    // cv::imshow("Display Image", image);
    // cv::waitKey(0);
    //     cv::circle(image, Mesh[Quad[p][1]], 2.5, cv::Scalar(255), -1);
    // cv::imshow("Display Image", image);
    // cv::waitKey(0);
    //     cv::circle(image, Mesh[Quad[p][2]], 2.5, cv::Scalar(255), -1);
    // cv::imshow("Display Image", image);
    // cv::waitKey(0);
    //     cv::circle(image, Mesh[Quad[p][3]], 2.5, cv::Scalar(255), -1);
    cv::imshow("Display Image", tmp);
    cv::waitKey(0);
    //}

    DrawMatrix(image.rows, image.cols, Quad, Boundary_types, Mesh, lines, image);
	//Converge(image.rows, image.cols, Quad, Boundary_types, Mesh, lines, image);

	return 0;
}

void Converge(
    int Rows, int Cols,
    std::vector<cv::Vec4i> const& Quads, 
    std::vector<int> const& Boundary_types, 
    std::vector<cv::Point> const& vertex_map, 
    std::vector<cv::Vec4i> const& seg_lines,
    cv::Mat3b& img )
{
	int N = Quads.size();
    int M = seg_lines.size();    
    int K = 0;
	int T = 2*vertex_map.size();
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
			output_mat = LAMBDA_L * output_mat;
			
		}  
	}
}

void general_case(LineSeg input_line, cv::Point& p ,int ct, LineSeg const& edgeAB, std::vector<cv::Point> const&  vertexMap,
 const int high_prior_index, const int low_prior_index, cv::Mat1f& vertices_to_e)
{
    cout << "general" << endl;
	p = input_line.IntersectionPointWith(edgeAB);
    float inv = std::abs( 1.0f/(vertexMap[high_prior_index].x - vertexMap[low_prior_index].x) );
	if(isnan(inv) || isinf(inv))
		return;
    //assert(!isnan(inv_AB)&&!isinf(inv_AB));

    vertices_to_e[ct][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - p.x) * inv;
    vertices_to_e[ct][2 * low_prior_index] = std::abs(p.x - vertexMap[high_prior_index].x) * inv;
    inv = std::abs( 1.0f/(vertexMap[high_prior_index].y - vertexMap[low_prior_index].y) );
	if(isnan(inv) || isinf(inv))
		return;
    //assert(!isnan(inv_AB)&&!isinf(inv_AB));
    vertices_to_e[ct][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - p.y) * inv;
    vertices_to_e[ct][2 * low_prior_index + 1] = std::abs(p.y - vertexMap[high_prior_index].y) * inv; 
}

void parrallel_case(LineSeg const& input_line, std::vector<cv::Point> const&  vertexMap, int high_prior_index, int low_prior_index, 
cv::Mat1f& vertices_to_e, cv::Mat1f& C ,cv::Mat1f& R, cv::Mat1f& e_hat)
{
	cout << "parrallel" << endl;
	cv::Mat1f R_transpose = cv::Mat1f::zeros(2, 2);
	transpose(R, R_transpose);
	cv::Point left,right;
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
    e_hat[0][0] = left.x;
    e_hat[1][0] = left.y;
    e_hat[0][1] = right.x;
    e_hat[1][1] = right.y;

    C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
    float inv = std::abs( 1.0f/(vertexMap[low_prior_index].x - vertexMap[high_prior_index].x) );
	if(isnan(inv) || isinf(inv))
		return;
    //assert(!isnan(inv) && !isinf(inv));
    vertices_to_e[0][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - left.x)*inv;
    vertices_to_e[0][2 * low_prior_index] = std::abs(vertexMap[high_prior_index].x - left.x)*inv;   
    vertices_to_e[1][2 * high_prior_index] = std::abs(vertexMap[low_prior_index].x - right.x)*inv;
    vertices_to_e[1][2 * low_prior_index] = std::abs(vertexMap[high_prior_index].x - right.x)*inv;

    inv = std::abs( 1.0f/(vertexMap[low_prior_index].y - vertexMap[high_prior_index].y) );
	if(isnan(inv) || isinf(inv))
		return;
    //assert(!isnan(inv) && !isinf(inv));
    vertices_to_e[0][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - left.y)*inv;
    vertices_to_e[0][2 * low_prior_index + 1] = std::abs(vertexMap[high_prior_index].y - left.y)*inv;
    vertices_to_e[0][2 * high_prior_index + 1] = std::abs(vertexMap[low_prior_index].y - left.y)*inv;
    vertices_to_e[0][2 * low_prior_index + 1] = std::abs(vertexMap[high_prior_index].y - left.y)*inv;
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

    cv::Point intersect_points[2];
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
			     
            parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat);
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
        if( input_line.is_parrallel(edgeAB))
        {         
            parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat);
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
    if ( input_line.isIntersectionLine(edgeBD) )
    {
        //cout << "in BD" << endl;
		int high_prior_index,low_prior_index;
        high_prior_index = Quad[0];
        low_prior_index = Quad[2];
        if( input_line.is_parrallel(edgeAB))
        {         
            parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat);
            out = C*vertices_to_e;
            return;
        }
        else
        {
            general_case(input_line, intersect_points[intersect_ct], intersect_ct,edgeAC, vertexMap, high_prior_index, low_prior_index, vertices_to_e);
            intersect_ct++;
        }     
    }

    if ( input_line.isIntersectionLine(edgeAC) )
    {
		//cout << "in AC" << endl;
        int high_prior_index,low_prior_index;
        high_prior_index = Quad[1];
        low_prior_index = Quad[3];
        if( input_line.is_parrallel(edgeAB))
        {         
            parrallel_case(input_line, vertexMap, high_prior_index, low_prior_index, vertices_to_e, C, R, e_hat);
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
	//cout << "intersect_ct " <<  intersect_ct << endl;
    assert(intersect_ct>=0); 
    assert(intersect_ct<=2);
	
    if(intersect_ct == 2)
    {
        cout << "intersect_ct == 2" << endl;
		if(intersect_points[0].x > intersect_points[1].x || 
		((intersect_points[0].x==intersect_points[1].x)&&(intersect_points[0].y < intersect_points[1].y)) )
        {
            cv::Mat1f temp = cv::Mat1f::zeros(1, vertices_nm);
			cout << "hello" << endl;
            for(int i = 0; i < vertices_nm; i++)
            {
                temp[0][i] = vertices_to_e[0][i];
            }
			cout << "fell" << endl;
            for(int i = 0; i < vertices_nm; i++)
            {
                vertices_to_e[0][i] = vertices_to_e[1][i];
                vertices_to_e[1][i] = temp[0][i];
            }
			cout << "bad" << endl;
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
		cout << "before C" << endl;
		C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
        out = C*vertices_to_e;
		cout << "out is filled" << endl;
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
		C = R * e_hat * e_hat.inv(cv::DECOMP_SVD) * R_transpose - cv::Mat1f::eye(2, 2);
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
    cv::Mat3b& img )
{
    int N = Quads.size();
    int M = seg_lines.size();    
    int K = 0;
	int T = 2*vertex_map.size();
    for(auto& type : Boundary_types){
        if(type >= 0 && type <= 3) K++;
    }// count number of vertices that is on boundary
    //int rows = 8*N + 2*M + K + 1;
	//int rows = 8*N + 2*N*M + K ;
    int columns = T;

	vector<cv::Mat1f> A_queue;
	vector<float> b_queue;
    //cv::Mat1f A = cv::Mat1f::zeros(rows, columns); 
    // Shape Preservation NxT
	
    for(int i = 0; i < N; i++){
        cv::Vec4i qd = Quads[i];
        cv::Mat1f Aq = cv::Mat1f::zeros(8, 4);

        for(int v = 0; v < 4; v++){
            int index = qd[v];
            int x = vertex_map[index].x;
            int y = vertex_map[index].y;

            Aq[2*v][0] = x;
            Aq[2*v][1] = -y;
            Aq[2*v][2] = 1;
            Aq[2*v][3] = 0;

            Aq[2*v+1][0] = y;
            Aq[2*v+1][1] = x;
            Aq[2*v+1][2] = 0;
            Aq[2*v+1][3] = 1;
        }


        cv::Mat1f AqAq_pI = Aq*Aq.inv(cv::DECOMP_SVD) - cv::Mat1f::eye(8, 8);
		
		for(int j = 0; j < 8 ;j ++)
		{
			cv::Mat1f forbigA = cv::Mat1f::zeros(1, T);
			int ct = 0;
			for(int v = 0; v < 4; v++)
			{
				int index = qd[v];
				forbigA[0][2*index] = AqAq_pI[j][ct++];
				forbigA[0][2*index+1] = AqAq_pI[j][ct++];
			}
			if(countNonZero(forbigA)>0)
			{
				A_queue.push_back(forbigA);
				b_queue.push_back(0);
			}
		}
		
		
		//for_bigA.copyTo(A(cv::Rect(0, i, T, 1)));
    }
	cout << "N:" << N << endl;
	cout << "8*N:" << 8*N << endl;
	cout << A_queue.size() << endl;
	cout << "Shape Preservation ok" << endl;
    // Line Preservation (MxNx2)xT
    
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
	cout << "Line Preservation ok" << endl;

    // Boundary Constraint Kx8N, K < N

	
    for(int i = 0 ; i < N; i++){
        cv::Vec4i qd = Quads[i];

        for(int v = 0; v < 4; v++){
            int v_index = qd[v];


            // cout << i << " " << v << " " << v_index << " " << Boundary_types[v_index] << endl;

            if(Boundary_types[v_index]<0 || Boundary_types[v_index]>3)
                continue;


            int x = vertex_map[v_index].x;
            int y = vertex_map[v_index].y;
            
            int t = Boundary_types[v_index];
			cv::Mat1f temp_mat = cv::Mat1f::zeros(1,T);
		
			if(t==0)
			{
				// up
				
				temp_mat[0][ 2 * v_index + 1] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}	
			}
			if(t==1)
			{
				// down
				
				temp_mat[0][ 2 * v_index + 1] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back((Rows - 1)*inf);
				}
			}
			if(t==2)
			{
				// left
				
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{
					A_queue.push_back(temp_mat);
					b_queue.push_back(0);
				}

			}
			if(t==3)
			{
				// right
				
				temp_mat[0][ 2 * v_index ] = inf;
				if(countNonZero(temp_mat))
				{
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
    cerr << "ready for solve!" << endl;
	cerr << "A rows: " << rows << endl;
	cerr << "A cols: " << T << endl;
    cv::Mat1f result;
	cv::solve(A, b, result, cv::DECOMP_SVD);
	for(int i =0 ;i< T; i++)
	{
		//cout << result[0][i] << endl;
	}
	/*
	gsl_matrix *gslA = gsl_matrix_alloc(rows,columns);
	gsl_vector *gslb = gsl_vector_alloc(rows);
	gsl_matrix_set_zero(gslA);
	gsl_vector_set_zero(gslb);
	for(int i=0;i<rows;i++){
		for(int j=0;j<columns;j++){
			gsl_matrix_set(gslA,i,j,A[i][j]);
			
		}
		gsl_vector_set(gslb,i,b[i][0]);
	}

	gsl_matrix *V = gsl_matrix_alloc(columns,columns);
	gsl_vector *S = gsl_vector_alloc(columns);
	gsl_vector *work = gsl_vector_alloc(columns);
	gsl_vector *x = gsl_vector_alloc(columns);
	gsl_linalg_SV_decomp(gslA,V,S,work);
	gsl_linalg_SV_solve(gslA,V,S,gslb,x);
	for(int i = 0; i < columns ; i++)
	{
		result[i][0] = gsl_vector_get(x,i);
	}
	*/
    
    cerr << "Solve completed!" << endl;
	/*
	arma::mat armaA(A.rows,A.cols);//cvImg is a cv::Mat
	Cv_mat_to_arma_mat(A,armaA);
	arma::mat armab(b.rows,b.cols);//cvImg is a cv::Mat
	Cv_mat_to_arma_mat(b,armab);
	arma::mat x1 = spsolve(A, b,"superlu");
	Arma_mat_to_cv_mat<float>(arma_img,result);
	*/

    for(int i = 0; i < result.rows; i+=2){
        float x = result[0][i];
        float y = result[0][i+1];
        cv::circle(img, cv::Point((int)x, (int)y), 2.5, cv::Scalar(255), -1);
    }

    cout << "show picture" << endl;
    cv::imshow("Display Image", img);
	cv::imwrite("results/global.png", img);
    cv::waitKey(0);
}















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






















////////////////////// Local Warping

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
	int p = 0;
	for(int x = 0; x < c; inc(x, g, c-1)){
		for(int y = 0; y < r; inc(y, g, r-1)){
			vertex_map[p] = cv::Point(-displacement[y][x]) + cv::Point(x, y);            
            
            if(y==0){
                bound_types[p] = 0;
            }else if(x == c - 1){
                bound_types[p] = 3;
            }else if(y == r - 1){
                bound_types[p] = 1;
            }else if(x == 0){
                bound_types[p] = 2;
            }else
                bound_types[p] = -1;

            if(x > 0 && y > 0){
                quads.push_back({p - virt_elem - 1, p - 1, p - virt_elem, p});
            }

            p++;
		}
	}
}


void SeamHorizontal(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed,
	cv::Mat4i& displacements,
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
			if(border[i][j]){
				delta[i][j] = inf;
			}else if(seamed[i][j]){
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
		
		if(up && best_i > 0){
			displacements[best_i][j][0] += 1;
		}
		else if(best_i < r-1){          
			displacements[best_i][j][1] += 1;

		}

		seamed[best_i][j] = 255;
		best_i = phi[best_i][j];
	}
}







// ///////////////////////////////////////
// ///////////////////////////////////////







void SeamVertical(
	cv::Mat1b& img, 
	cv::Mat1b& border, 
	cv::Mat1b& seamed, 
	cv::Mat4i& displacements, 
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
			if(border[i][j]){
				delta[i][j] = inf;
			}else if(seamed[i][j]){
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
		
		if(left && best_j > 0){
			displacements[i][best_j][2] += 1;
		}
		else if(best_j < c-1){
			displacements[i][best_j][3] += 1;
		}

		seamed[i][best_j] = 255;
		best_j = phi[i][best_j];
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
    cv::Mat1b minpool;
    auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );
    cv::erode( img, minpool, element );

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

		if(minpool(pix) < DARKTHRES){
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
	element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(dilation_size, dilation_size) );
	cv::Mat tmp;
	cv::dilate( out, tmp, element );
	cv::GaussianBlur(tmp, out, cv::Size(11, 11), 0, 0, cv::BORDER_DEFAULT);
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
	cv::Mat4i disps = cv::Mat4i::zeros(r, c);

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
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	while((tp = subImage(border_tmp, sub)) != -1){
		subseq.push_back(sub);
		tpseq.push_back(tp);

		work.push_back(sub.width*(!(tp/2))+sub.height*(tp/2));
		if(work.size()>1)
			work.back() += work[work.size()-2];

		cv::Rect window;
		cv::Point move;
		switch(tp){
			case 0:
			window = cv::Rect(sub.x, 0, sub.width, 1);
			move = cv::Point(0, 1);			
			// cerr << "Up" << endl;
			break;
			case 1:
			window = cv::Rect(sub.x, r-1, sub.width, 1);
			move = cv::Point(0, -1);
			// cerr << "Down" << endl;
			break;
			case 2:
			window = cv::Rect(0, sub.y, 1, sub.height);
			move = cv::Point(1, 0);
			// cerr << "Left" << endl;
			break;
			case 3:
			window = cv::Rect(c-1, sub.y, 1, sub.height);
			move = cv::Point(-1, 0);
			// cerr << "Right" << endl;
			break;
		}
        // cerr << border_tmp.rows << " " << border_tmp.cols << endl; 
		while(cv::countNonZero(border_tmp(window))>0){
            // cerr << window.x << " " << window.y << " " << window.width << " " << window.height << endl;
			cv::Rect src(window.x+move.x, window.y+move.y, window.width, window.height);
            if(src.x < 0 || src.y < 0 || src.x >= c || src.y >= r)
                break;
			border_tmp(src).copyTo(border_tmp(window));
			window = src;
		}        
        // cv::imshow("Display Image", border_tmp);
        // cv::waitKey(0);
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
		auto seamed_sub = seamed(sub);
		auto energy_sub = energy(sub);
		auto disp_sub = disps(sub);

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
		std::string bar(30, '.');
		std::fill(bar.begin(), bar.begin() + (int)std::floor(30 * work[i] / work.back()),'=');
		cerr << "(" << i+1 << "/" << subseq.size() << ") seams done. Progress [" << bar << "]\r";
	}
	cerr << endl << "Local Warping Completed." << endl;

	// compute displacement
	// cerr << "all zero? " << (cv::countNonZero(disps)==0) << endl;
	for(int i = 1; i < r; i++){
		for(int j = 0; j < c; j++){
			disps[r-1-i][j][0] += disps[r-i][j][0];
			disps[i][j][1] += disps[i-1][j][1];
		}
	}
	for(int j = 1; j < c; j++){
		for(int i = 0; i < r; i++){
			disps[i][c-1-j][2] += disps[i][c-j][2];
			disps[i][j][3] += disps[i][j-1][3];
		}
	}
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			int y = disps[i][j][1] - disps[i][j][0]; 
			int x = disps[i][j][3] - disps[i][j][2]; 
			displacement[i][j] = cv::Vec2i(x, y);
		}
	}


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
