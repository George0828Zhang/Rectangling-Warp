#include "Ray.h"
void GeneratingQuad(Vec4i& Quad, std::vector<cv::Point>& vertexMap)
{
    int intersect_ct = 0;
    //Ray will only intersect two points or one point
    cv::Mat1f Z = cv::Mat1f.zeros(4, 8);
    //horizontal
    cv::Vec2i edgeAB = vertexMap[Quad[0]] - vertexMap[Quad[1]];
    cv::Vec2i edgeCD = vertexMap[Quad[2]] - vertexMap[Quad[3]];

    if( Ray.isIntersectionLine(edgeAB))
    {
        cv::Point p = Ray.IntersectionPoint(edgeAB);
        inv_AB = 1/edgeAB;
        assert(inv_AB!=0);
        Z[0 + intersect_ct*2][0] = (vertexMap[Quad[1]].x - p.x) * inv_AB;
        Z[0 + intersect_ct*2][1] = (p.x - vertexMap[Quad[0]].x) * inv_AB;

        Z[1 + intersect_ct*2][0] = (vertexMap[Quad[1]].y - p.y) * inv_AB;
        Z[1 + intersect_ct*2][1] = (p.y - vertexMap[Quad[0]].y) * inv_AB; 
        intersect_ct++;    
    }

    if ( Ray.isIntersectionLine(edgeCD) )
    {
        cv::Point p = Ray.IntersectionPoint(edgeCD);
        inv_CD = 1/edgeCD;
        assert(inv_CD!=0);
        Z[0 + intersect_ct*2][2] = (vertexMap[Quad[3]].x - p.x) * inv_CD;
        Z[0 + intersect_ct*2][3] = (p.x - vertexMap[Quad[2]].x) * inv_CD;

        Z[1 + intersect_ct*2][2] = (vertexMap[Quad[3]].y - p.y) * inv_CD;
        Z[1 + intersect_ct*2][3] = (p.y - vertexMap[Quad[2]].y) * inv_CD;
        intersect_ct++;    
    }
    //vertical
    cv::Vec2i edgeAC = vertexMap[Quad[0]] - vertexMap[Quad[2]];
    cv::Vec2i edgeBD = vertexMap[Quad[1]] - vertexMap[Quad[3]];
    if ( Ray.isIntersectionLine(edgeBD) )
    {
        cv::Point p = Ray.IntersectionPoint(edgeBD);
        inv_BD = 1/edgeBD;
        assert(inv_BD!=0);
        Z[0 + intersect_ct*2][3] = (vertexMap[Quad[1]].x - p.x) * inv_BD;
        Z[0 + intersect_ct*2][1] = (p.x - vertexMap[Quad[3]].x) * inv_BD;

        Z[1 + intersect_ct*2][3] = (vertexMap[Quad[1]].y - p.y) * inv_BD;
        Z[1 + intersect_ct*2][1] = (p.y - vertexMap[Quad[3]].y) * inv_BD;
        intersect_ct++;    
    }

    if ( Ray.isIntersectionLine(edgeAC) )
    {
        cv::Point p = Ray.IntersectionPoint(edgeAC);
        inv_AC = 1/edgeAC;
        assert(inv_AC!=0);
        Z[0 + intersect_ct*2][2] = (vertexMap[Quad[0]].x - p.x) * inv_AC;
        Z[0 + intersect_ct*2][0] = (p.x - vertexMap[Quad[2]].x) * inv_AC;

        Z[1 + intersect_ct*2][2] = (vertexMap[Quad[0]].y - p.y) * inv_AC;
        Z[1 + intersect_ct*2][0] = (p.y - vertexMap[Quad[2]].y) * inv_AC;
        intersect_ct++;    
    }
    
    assert(intersect_ct>=1); 
    assert(intersect_ct<=2);

}
