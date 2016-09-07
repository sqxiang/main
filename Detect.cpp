#include "Detect.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat Detect::SobelMat(Mat src,int kernel,int scale,int delta){
   //sobel算子找轮廓 
  Mat src_gray,grad;
  int ddepth = CV_16S;
 GaussianBlur( src, src_gray, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// 创建 grad_x 和 grad_y 矩阵
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// 求 X方向梯度
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, kernel, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  /// 求Y方向梯度
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, kernel, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
  /// 合并梯度(近似)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  return grad;

}


Point Detect::GrowPoint(Mat src,int pointTh,int stdTh,int stdSize,vector<Point> choosedPoint){
   //找出生长点
   int rows = src.rows,cols = src.cols;
   int minGrey=255;
   vector<Point>::iterator it;
   Point dst;
   for(int i=0;i<rows;i++){
     for(int j=0;j<cols;j++){
     if(src.at<uchar>(i,j)<pointTh) {continue;}
       if(src.at<uchar>(i,j)<minGrey && CountStd(src,Point(j,i),stdSize)<stdTh){
            it = find(choosedPoint.begin(),choosedPoint.end(),Point(j,i));
            if(it!=choosedPoint.end()) { continue; }
            minGrey = src.at<uchar>(i,j); 
            dst = Point(j,i);  
           }
         }
        }
 if(minGrey==255) dst = Point(-1,-1);
 cout<<"the grow pixel: "<<minGrey<<endl;
    return dst;
}


Mat Detect::RegionGrow(Mat src, Point2i pt, int regionTh, Mat outline)  
  //生长算法
{  
    Point2i ptGrowing;                      //待生长点位置  
    int nGrowLable = 0;                             //标记是否生长过  
    int nSrcValue = 0;                              //生长起点灰度值  
    int nCurValue = 0;                              //当前生长点灰度值  
    Mat matDst = Mat::zeros(src.size(), CV_8UC1);   //创建一个空白区域，填充为黑色  
    //生长方向顺序数据  
    int DIR[8][2] = {{-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}};    
    Vector<Point2i> vcGrowPt;                     //生长点栈  
    vcGrowPt.push_back(pt);                         //将生长点压入栈中  
    matDst.at<uchar>(pt.y, pt.x) = 255;               //标记生长点  
    nSrcValue = src.at<uchar>(pt.y, pt.x);            //记录生长点的灰度值  
      
    while (!vcGrowPt.empty())                       //生长栈不为空则生长  
    {  
        pt = vcGrowPt.back();                       //取出一个生长点  
       nSrcValue = src.at<uchar>(pt.y, pt.x);       //记录当前生长点的灰度值  
        vcGrowPt.pop_back();
 
   
        //分别对八个方向上的点进行生长  
        for (int i = 0; i<9; ++i)  
        {  
            ptGrowing.x = pt.x + DIR[i][0];       
            ptGrowing.y = pt.y + DIR[i][1];   
            
            //检查是否是轮廓边缘
            if(outline.at<uchar>(ptGrowing.y,ptGrowing.x)==255){
               continue;
             } 
//        float std_pxl = CountStd(src,ptGrowing,3);      //计算该点附近方差
  //        if(std_pxl>10) continue;     
 
            //检查是否是边缘点  
            if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x > (src.cols-1) || (ptGrowing.y > src.rows -1))  
                continue;  
  
            nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);      //当前待生长点的灰度值  
  
            if (nGrowLable == 0)                    //如果标记点还没有被生长  
            {  
                nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);            
                if (abs(nSrcValue - nCurValue) < regionTh)                 //在阈值范围内则生长  
                {  
                    matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = nCurValue;     //标记为当前像素 
                    vcGrowPt.push_back(ptGrowing);                  //将下一个生长点压入栈中  
                    
                }  
            }  
        }  
    }  
    return matDst.clone();  
}

Mat Detect::ContourInside(Mat src,vector<vector<Point> > contours){
   //找出轮廓内的像素
  Mat dst = Mat::zeros(src.size(),src.type());
  int flag;
  for(int i=0;i<dst.rows;i++)
    for(int j=0;j<dst.cols;j++){
       for(int k=0;k<contours.size();k++){
          flag = pointPolygonTest(contours[k],Point(j,i),false);
          if(flag>=0) {break;}
       }
    if(flag>=0){
        dst.at<uchar>(i,j) = src.at<uchar>(i,j); }
     }
   return dst;
}

float Detect::CountStd(Mat src,Point pt,int size){
       //求以某点为中心的周围方差
        int a = (size-1)/2;
        int tx = (pt.x-a<0)?0:(pt.x-a);
        int ty = (pt.y-a<0)?0:(pt.y-a);
        int sx = (pt.x+a>src.cols-1)?(src.cols-1-tx+1):(pt.x+a-tx+1);       
        int sy = (pt.y+a>src.rows-1)?(src.rows-1-ty+1):(pt.y+a-ty+1);       
//        cout<<"("<<tx<<","<<ty<<")"<<" ["<<sx<<","<<sy<<"]"<<endl;
        Mat roi = src(Rect(tx,ty,sx,sy));
        Scalar mean,std;
        meanStdDev(roi,mean,std);                    
        float mean_pxl,std_pxl;
        std_pxl = std.val[0];
//        cout<<"std "<<std_pxl<<endl;
        return std_pxl;
 }


vector<vector<Point> > Detect::FindContours(Mat thresh,double area){
  //寻找轮廓并删除过小的轮廓
  /*寻找轮廓*/
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours( thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );  
  
   
    /*删除较小的轮廓面积*/
  vector<vector<Point> >::iterator itc = contours.begin();
  while(itc!=contours.end()){
    if(contourArea(*itc)<area)
       itc = contours.erase(itc);
    else
      ++itc;
    }
   
     return contours;
}


vector<Rect> Detect::DrawRect(Mat src,vector<vector<Point> > contours){
  //画出轮廓并用矩形包围
   Mat pContourImg = Mat::zeros( src.size(), CV_8UC1 );
  for(int i=0;i<contours.size();i++){
     drawContours(pContourImg, contours,i,Scalar(255,0,0),CV_FILLED,8,0,0);
    }
  //   imshow("morph",pContourImg);
 
  /// 获取矩形边界框
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );

  for( int i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       boundRect[i].x = boundRect[i].x-30>=0?boundRect[i].x-30:0;
       boundRect[i].y = boundRect[i].y-30>=0?boundRect[i].y-30:0;
       boundRect[i].width = boundRect[i].x+boundRect[i].width+60<src.cols?boundRect[i].width+60:src.cols-boundRect[i].x;
       
       boundRect[i].height = boundRect[i].y+boundRect[i].height+60<src.rows?boundRect[i].height+60:src.rows-boundRect[i].y;
     }


  /// 画多边形轮廓 + 包围的矩形框 + 圆形框
  for( int i = 0; i< contours.size(); i++ )
     {
       rectangle( src, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
     }

  return boundRect;

 }


