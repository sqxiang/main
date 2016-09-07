#ifndef TARGET_DETECT_H
#define TARGET_DETECT_H

#include <opencv2/opencv.hpp>

using namespace cv;

class Detect {
   public:
    ///@brief find the contours using sobel
    Mat SobelMat(Mat src,int kernel,int scale=1,int delta=0);
    /**
     * @brief find the grow point
     *  pointTh: the minimal pixel of the grow point
     *  stdTh: the max std of the area around grow point
     *  stdSize: the area of std
     *  choosedPoint: the grow point has been choosed, the next should eliminate it    
     */
    Point GrowPoint(Mat src,int pointTh,int stdTh, int stdSize,vector<Point> choosedPoint);
    /**
     * @brief grow region from grow point
     * pt: grow point
     * regionIh: the max diff between point that can be grow
     * outline: the contour of the src
     */
    Mat RegionGrow(Mat src,Point2i pt, int regionTh,Mat outline);
    ///@brief reserve the pixel inside the contours
    Mat ContourInside(Mat src,vector<vector<Point> > contours);
    ///@brief count the std around the pt with size of 'size' 
    float CountStd(Mat src,Point pt,int size);
    /**@brief find the contours of thresh(img)
     * area: the minimal area of the contours that should keep
     */  
    vector<vector<Point> > FindContours(Mat thresh,double area);
    ///draw the rect in src using the contours
    vector<Rect> DrawRect(Mat src,vector<vector<Point> > contours);

 
};
#endif
