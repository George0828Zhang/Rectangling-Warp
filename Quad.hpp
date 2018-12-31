using Vec8i = cv::Vec<int, 8>;

void DrawMatrix(std::vector<Vec8i> Quad, std::vector<cv::Vec4i> seg_lines){
    int rows = 8*Quad.size() + seg_lines.size() + 1;
    int columns = Quad.size() * 8;

    cv::Mat1f A = cv::Mat1f.zeros(rows, columns); 
}