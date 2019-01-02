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
#include<assert.h>

#include <string>
inline double PI() { return std::atan(1)*4; }

inline float find_closest_in_bin(float theta, int bin)
{
	float min = 1000;
	float angle = 0;
	for(int i = 1; i< bin ; i++)
	{
		if(std::abs(theta - PI()/i) < min)
		{
			min = std::abs(theta - PI()/i);
			angle = PI()/i - theta;
		}
	}
	return angle;
}
class LineSeg
{
	public:
		cv::Point2f origin;
		cv::Point2f end;

		LineSeg()
		{

		}

		LineSeg(cv::Point2f in_origin, cv::Point2f in_end)
		{
			origin = in_origin;
			end = in_end;
		}

		float getTan()
		{
			return (end.y-origin.y)/(end.x-origin.x);
		}

		cv::Point2f getdir()
		{
			float x = end.x - origin.x;
			float y = end.y - origin.y;
			cv::Point2f dir(x,y);
			dir = dir/sqrtf(x*x + y*y);
			return dir;
		}

		bool is_parrallel(LineSeg line)
		{
			if(line.getdir().x == this->getdir().x && line.getdir().y == this->getdir().y )
			{
				return true;
			}
			return false;
		}
		cv::Point2f IntersectionPointWith(LineSeg line)
		{
			cv::Point2f dir = end - origin;
			cv::Point2f this_normal(-dir.y,dir.x);
			//float normalize_normal = std::sqrtf(this_normal.x * this_normal.x + this_normal.y * this_normal.y);
			float this_constant = this_normal.dot(origin);
			float a_dis_ratio = std::abs(this_normal.dot(line.origin) - this_constant);
			float b_dis_ratio = std::abs(this_normal.dot(line.end) - this_constant);
			return (line.origin * b_dis_ratio + line.end * a_dis_ratio)/(a_dis_ratio+b_dis_ratio);
		}

		//from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
		bool onSegment(cv::Point2f p, cv::Point2f q, cv::Point2f r) 
		{ 
			if (q.x <= fmax(p.x, r.x) && q.x >= fmin(p.x, r.x) && 
				q.y <= fmax(p.y, r.y) && q.y >= fmin(p.y, r.y)) 
			return true; 
		
			return false; 
		} 
		
		int orientation(cv::Point2f p, cv::Point2f q, cv::Point2f r) 
		{ 
			int val = (q.y - p.y) * (r.x - q.x) - 
					(q.x - p.x) * (r.y - q.y); 
		
			if (val == 0) return 0; 
		
			return (val > 0)? 1: 2; 
		} 

		bool isIntersectionLine(LineSeg line)
		{
			cv::Point2f p1 = origin;
			cv::Point2f q1 = end;

			cv::Point2f p2 = line.origin;
			cv::Point2f q2 = line.end;

			int o1 = orientation(p1, q1, p2); 
			int o2 = orientation(p1, q1, q2); 
			int o3 = orientation(p2, q2, p1); 
			int o4 = orientation(p2, q2, q1); 
			if (o1 != o2 && o3 != o4) 
				return true; 
		
			if (o1 == 0 && onSegment(p1, p2, q1)) return true; 

			if (o2 == 0 && onSegment(p1, q2, q1)) return true; 
 
			if (o3 == 0 && onSegment(p2, p1, q2)) return true; 

			if (o4 == 0 && onSegment(p2, q1, q2)) return true; 
		
			return false;
		}
};
