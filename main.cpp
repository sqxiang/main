#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <cmath>
#include "Detect.h"
#include "Thresh.h"

#define PI 3.1415926
using namespace cv;
using namespace std;

const int GAP = 5; //第二步剔除峰值点，峰值必须比左右两边gap内的点都大
const int THRESH = 10; //第三步排除峰值过小的点，在原图中所占比例较小
const int PEAK_GAP = 12; //第四步排除相隔较近，峰值差较低的点
const float AREARATE = 0.5; //第5步排除两个峰值所属重合过大的点，保留剩下的最大的两个峰值
const int histSize = 255;
const double contoursAreaSize = 500.0; //可以作为最终划分出的物体轮廓的最小值
const int sobelKernelSize = 3;         //sobel算子核心大小
const int openningSize = 7;           //开运算算子大小
const int threshSize = 20;             //轮廓二值化门限，大于此值作为轮廓点
const int minPixelTh = 20;             //生长点的最小值门限，小于此值不考虑当做生长点
const int stdSize = 9;                //所需计算的方差范围
const int stdTh = 20;                //可以生长的方差最大值门限，大于此值不作为生长点
const int regionTh = 10;               //可以生长的差值最大值门限，大于此值不再生长


int main(int argc,char **argv){
  string dir = argv[1];
  struct stat fileStat;
   bool isDir=false;
   bool isFile=false;
  if ((stat(dir.c_str(), &fileStat) == 0) && S_ISDIR(fileStat.st_mode))
  {
    isDir = true;
  }
   if ((stat(dir.c_str(), &fileStat) == 0) && S_ISREG(fileStat.st_mode))
  {
    isFile = true;
  }
  
  cout<<isFile<<isDir<<endl;
  Detect det;
  Thresh qtt;  
  Mat src;
  DIR *dp;
  struct dirent *dirp;
  if((dp=opendir(dir.c_str()))==NULL)
    cout<<"Can't open "<<argv[1]<<endl;
  while(true){
    if(isDir){
      dirp=readdir(dp);
      if(dirp==NULL) break;
    if(strcmp(".",dirp->d_name)==0 || strcmp("..",dirp->d_name)==0)
         continue;
    string imgName = dir+"/"+ dirp->d_name;
    src = imread(imgName,-1);
    }
    if(isFile){
     src = imread(dir,-1);
   }
   if(!isDir && !isFile) break;

    if(src.channels()==3){
      vector<Mat> rgb_planes;
      split(src,rgb_planes);
     }
   
    if(src.channels()==1){
      float range[] = {0,255};
      const float *histRange = {range};
      Mat hist;
      bool uniform=true,accumulate=false;
      calcHist(&src,1,0,Mat(),hist,1,&histSize,&histRange,uniform,accumulate);
      // 创建直方图画布
      Mat histImage = qtt.drawHist(hist);

    //１,从大到小，找出可能的峰值，比左右两点都大的像素值
     vector<int> peak = qtt.StepOne(hist); 

    //2,从第一步的峰值中找出比左右两边５点都大的像素值（忽略小于５的像素点）
    vector<int> peak_s = qtt.StepTwo(hist,peak,GAP);

     //3,从第二步中剩余的像素点中找出个数大于阈值的像素点
     vector<int> peak_t = qtt.StepThree(hist,peak_s,THRESH);
 
     //4,从第三步剩余的像素点中排除相隔１２像素以内，个数之差小于１０的像素点中较小的那个
     vector<int> peak_f = qtt.StepFour(hist,peak_t,PEAK_GAP);
 
     int imin = 255;
     int imin2 = 255;
     int valley = 255;
     //当剩余点只有1个时，取该点
     if(peak_f.size()==1 && hist.at<float>(peak_f[0])>180){
       imin = peak_f[0]+10;
       valley = peak_f[0]+10;
       imin2 = peak_f[0]+10;
     }
       //5,当剩余点大于2时,考虑相邻两点间的面积
     if(peak_f.size()>=2){
       int res = qtt.StepFive(hist,peak_f,AREARATE);
       if(res==0) {imin = peak_f[0]+10;valley = peak_f[0]+10;imin2 = peak_f[0]+10;}
       if(res!=0){
        //利用递减性寻找波谷
        valley = qtt.FindValley(hist,peak_f[res],peak_f[res-1]);
        cout<<"valley: "<<valley<<endl;
        
        //6,高斯寻找两点之间的波谷
          imin = qtt.FindValley2(hist,peak_f,res);
          cout<<"imin: "<<imin<<endl;

          //利用最小值寻找波谷  
          imin2 = qtt.FindValley3(hist,peak_f,res);
          cout<<"imin2 "<<imin2<<endl;
         }
       }
              
      Mat histImage1(histSize,histSize,CV_8U,Scalar(0,0,0));
      for(int i=0;i<peak_f.size();i++){
         float binVal = hist.at<float>(peak_f[i]);
         int intensity = static_cast<int>(binVal);
         line(histImage1,Point(peak_f[i],histSize),Point(peak_f[i],histSize-intensity),Scalar::all(255));
      }
      Mat newSrc;
      cout<<"imin2: "<<imin2<<endl;
      threshold(src,newSrc,imin2,255,CV_THRESH_TOZERO_INV);
      imshow("dst",newSrc);
      imshow("histimage",histImage);
     // GaussianBlur(histImage,histImage,Size(13,13),0,0); 
      //imshow("hist",histImage);
      imshow("histimage1",histImage1);
      waitKey(0);
      destroyWindow("dst");   
      destroyWindow("histimage");   
      destroyWindow("histimage1");   
     //开运算
     Mat openning;
     Mat element = getStructuringElement( MORPH_ELLIPSE,Size(openningSize,openningSize),Point((openningSize-1)/2,(openningSize-1)/2));
     morphologyEx(newSrc,openning,2,element);
   
    //sobel算子查找轮廓
    Mat outline = det.SobelMat(openning,sobelKernelSize); 
//    imshow( "outline", outline );
//    waitKey(0);

    //轮廓二值化
    threshold(outline,outline,threshSize,255,CV_THRESH_BINARY);
    imshow("outline1",outline);
    waitKey(0);
    destroyWindow("outline1");
   
    //找出轮廓,删除过小的轮廓面积
    Mat thresh_tmp = openning.clone();
    imshow("thresh_tmp",thresh_tmp);
    waitKey(0);
    vector<vector<Point> > contours = det.FindContours(thresh_tmp,contoursAreaSize);
    //找出轮廓内的像素
   Mat inside = det.ContourInside(openning,contours);
//   namedWindow("inside");
    imshow("inside",inside);
    waitKey(0);    
    destroyWindow("inside");   
    
    //生长
    Mat region_1;
    Point growPoint;
    Mat dst_n=inside.clone();
    vector<Rect> result;
    int n = contours.size();
    cout<<"n: "<<n<<endl;
    vector<Point> po;

    //循环找出物体
    while(true){
    //找出像素的最小值
    Point growPoint = det.GrowPoint(dst_n,minPixelTh,stdTh,stdSize,po);
    cout<<growPoint<<endl;
    po.push_back(growPoint);
    //当前图像已经无法找到满足要求的生长点
    if(growPoint==Point(-1,-1)) break;
    //生长
    region_1 = det.RegionGrow(dst_n,growPoint,regionTh,outline);
    imshow("region",region_1);
    //提取轮廓并画出物体
    vector<vector<Point> > contours_region;
    contours_region = det.FindContours(region_1,contoursAreaSize);
    vector<Rect> rec = det.DrawRect(src,contours_region);
   //将rect都存入result
    result.insert(result.end(),rec.begin(),rec.end());
    //创建蒙版
    Mat mask = Mat::ones(dst_n.size(),dst_n.type());
    for(int i=0;i<rec.size();i++){
     mask(rec[i]).setTo(0);
    }
   // imshow("mask",mask);
    Mat img;
    dst_n.copyTo(img,mask);
   dst_n.release();
    dst_n = img.clone();
    img.release();
    for(int i=0;i<region_1.rows;i++)
     for(int j=0;j<region_1.cols;j++){
          if(region_1.at<uchar>(i,j)!=0)
              dst_n.at<uchar>(i,j)=0;
     }
   region_1.release();
   imshow("dst_n",dst_n);
    waitKey(0);
    destroyWindow("region");
    destroyWindow("dst_n"); 
   }
 
        //如果两个框有重叠，合并
    vector<Rect> extend;
    for(int i=0;i<result.size()-1;i++)
     for(int j=i+1;j<result.size();j++){
       if((result[i]&result[j]).width>0 || (result[i]-Point(10,10)+Size(20,20) & result[j]).width>0){
             extend.push_back(result[i]|result[j]);
      }
     }

    result.insert(result.end(),extend.begin(),extend.end());

     for(int i=0;i<result.size();i++){
     rectangle(src,result[i].tl(),result[i].br(),Scalar(255,0,0),2,8,0);
     }
   
    imshow("src",src);
    waitKey(0);
      }
     if(isFile) break;
    }
      return 0;
}
