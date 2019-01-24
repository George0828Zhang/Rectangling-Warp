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
#include <limits>
#include <unistd.h>
#include <string>
inline double PI() { return std::atan(1)*4; }

inline float find_closest_in_bin(float theta, int bin)
{
	float min = 1000;
	float angle = 0;
	for(int i = 1; i < bin ; i++)
	{
		
		//std::cout << theta - PI() * (float)i/bin << std::endl;
		//sleep(1);
		if(std::abs(theta - PI() * (float)i/bin + PI()/2) < min)
		{
			min = std::abs(theta - PI() * (float)i/bin + PI()/2);
			//std::cout << "find" << min << std::endl;
			//angle = PI() * (float)i/bin - PI()/2;
			angle = PI() * (float)i/bin - PI()/2 - theta;
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

			if(in_origin.x > in_end.x || ((in_origin.x == in_end.x)&&(in_origin.y > in_end.y)) )
			{
				origin = in_end;
				end = in_origin;
			}
			else{
				origin = in_origin;
				end = in_end;
			}
			
		}

		float getTan()
		{

			return std::abs(end.y-origin.y)/(end.x-origin.x);
		}

		cv::Point2f getdir()
		{
			float x = end.x - origin.x;
			float y = end.y - origin.y;
			cv::Point2f dir(x,y);
			dir = dir/sqrtf(x*x + y*y);
			return dir;
		}

		bool is_parrallel(LineSeg& line)
		{
			//std::cout << "line" << line.getdir().x << "this" << this->getdir().x << std::endl;
			//std::cout << "line" << line.getdir().y << "this" << this->getdir().y << std::endl;
			if(line.getdir().x == this->getdir().x && line.getdir().y == this->getdir().y )
			{
				return true;
			}
			return false;
		}
		cv::Point2f IntersectionPointWith(LineSeg const& line)
		{
			cv::Point2f dir = end - origin;
			cv::Point2f this_normal(-dir.y,dir.x);
			//float normalize_normal = std::sqrtf(this_normal.x * this_normal.x + this_normal.y * this_normal.y);
			float this_constant = this_normal.dot(origin);
			float a_dis_ratio = std::abs(this_normal.dot(line.origin) - this_constant);
			float b_dis_ratio = std::abs(this_normal.dot(line.end) - this_constant);
			//std::cout << "a_dis" << a_dis_ratio << std::endl;
			//std::cout << "b_dis" << b_dis_ratio << std::endl;
			return (line.origin * b_dis_ratio + line.end * a_dis_ratio)/(a_dis_ratio+b_dis_ratio);
		}

		//from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
		//from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
		bool onSegment(cv::Point2f& p, cv::Point2f& q, cv::Point2f& r) 
		{ 
			if (q.x <= fmax(p.x, r.x) && q.x >= fmin(p.x, r.x) && 
				q.y <= fmax(p.y, r.y) && q.y >= fmin(p.y, r.y)) 
			return true; 
		
			return false; 
		} 
		
		int orientation(cv::Point2f& p, cv::Point2f& q, cv::Point2f& r) 
		{ 
			int val = (q.y - p.y) * (r.x - q.x) - 
					(q.x - p.x) * (r.y - q.y); 
		
			if (val == 0) return 0; 
		
			return (val > 0)? 1: 2; 
		} 

		bool isIntersectionLine(LineSeg const& line)
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

		bool doIntersect(cv::Point2f& p1, cv::Point2f& q1, cv::Point2f& p2, cv::Point2f& q2) 
		{ 
		    // Find the four orientations needed for general and 
		    // special cases 
		    int o1 = orientation(p1, q1, p2); 
		    int o2 = orientation(p1, q1, q2); 
		    int o3 = orientation(p2, q2, p1); 
		    int o4 = orientation(p2, q2, q1); 
		  
		    // General case 
		    if (o1 != o2 && o3 != o4) 
		        return true; 
		  
		    // Special Cases 
		    // p1, q1 and p2 are colinear and p2 lies on segment p1q1 
		    if (o1 == 0 && onSegment(p1, p2, q1)) return true; 
		  
		    // p1, q1 and q2 are colinear and q2 lies on segment p1q1 
		    if (o2 == 0 && onSegment(p1, q2, q1)) return true; 
		  
		    // p2, q2 and p1 are colinear and p1 lies on segment p2q2 
		    if (o3 == 0 && onSegment(p2, p1, q2)) return true; 
		  
		     // p2, q2 and q1 are colinear and q1 lies on segment p2q2 
		    if (o4 == 0 && onSegment(p2, q1, q2)) return true; 
		  
		    return false; // Doesn't fall in any of the above cases 
		} 

		// Returns true if the point p lies inside the polygon[] with n vertices 
		bool isInside(cv::Point2f polygon[], int n, cv::Point2f& p) 
		{ 
		    // There must be at least 3 vertices in polygon[] 
		    if (n < 3)  return false; 
		  
		    // Create a point for line segment from p to infinite 
		    cv::Point2f extreme = {std::numeric_limits< int >::max(), p.y}; 
		  
		    // Count intersections of the above line with sides of polygon 
		    int count = 0, i = 0; 
		    do
		    { 
		        int next = (i+1)%n; 
		  
		        // Check if the line segment from 'p' to 'extreme' intersects 
		        // with the line segment from 'polygon[i]' to 'polygon[next]' 
		        if (doIntersect(polygon[i], polygon[next], p, extreme)) 
		        { 
		            // If the point 'p' is colinear with line segment 'i-next', 
		            // then check if it lies on segment. If it lies, return true, 
		            // otherwise false 
		            if (orientation(polygon[i], p, polygon[next]) == 0) 
		               return onSegment(polygon[i], p, polygon[next]); 
		  
		            count++; 
		        } 
		        i = next; 
		    } while (i != 0); 
		  
		    // Return true if count is odd, false otherwise 
		    return count&1;  // Same as (count%2 == 1) 
		}

		bool origin_is_Inside(cv::Point2f polygon[], int n)
		{
			return isInside(polygon, n, this->origin) ;
		} 

		bool end_is_Inside(cv::Point2f polygon[], int n)
		{
			return isInside(polygon, n, this->end); 
		}
};
