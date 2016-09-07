#include "Thresh.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#define PI 3.1415926
using namespace std;
using namespace cv;

int histSize = 255;
//step1
vector<int> Thresh::StepOne(Mat hist){
     int sub[histSize-1];
     for(int i=histSize-1;i>0;i--){
       sub[i-1] = hist.at<float>(i-1)-hist.at<float>(i)>0?1:-1;
     }
     vector<int> peak;
     for(int i=histSize-2;i>0;i--){
       if(sub[i]==1 && sub[i-1]==-1){
           peak.push_back(i);
       }
     }
    return peak;
}

//step2
vector<int> Thresh::StepTwo(Mat hist,vector<int> peak,int gap){
    vector<int> peak_s;
    for(int j=0;j<peak.size();j++){
       for(int i=1;i<=gap;i++){
           int imax=peak[j]+i<histSize?peak[j]+i:histSize-1;
           int imin = peak[j]-i;
           if(imin<0 || hist.at<float>(peak[j])<hist.at<float>(imax) || hist.at<float>(peak[j])<hist.at<float>(imin)){
                 break;
               } 
           if(i==gap) {peak_s.push_back(peak[j]);   
                }
        }
      }
    return peak_s;
}

//step3
vector<int> Thresh::StepThree(Mat hist,vector<int> peak_s,int thresh){
    vector<int> peak_t;
    for(int i=0;i<peak_s.size();i++){
        if(hist.at<float>(peak_s[i])>thresh)
        peak_t.push_back(peak_s[i]);
     } 
   return peak_t;
}

//step4
vector<int> Thresh::StepFour(Mat hist,vector<int> peak_t,int peak_gap){
  if(peak_t.size()>0){
       vector<int>::iterator itc = peak_t.begin();
       itc++;
       while(itc!=peak_t.end()){
         if(abs(*itc-*(itc-1))<peak_gap && abs(hist.at<float>(*itc)-hist.at<float>(*(itc-1))<10)){
             vector<int>::iterator it = hist.at<float>(*itc)<hist.at<float>(*(itc-1))?itc:itc-1;
             itc = peak_t.erase(it);
          }
         else
          ++itc;
       }
     }
  return peak_t;
}

//step5
int Thresh::StepFive(Mat hist,vector<int> peak_f,float areaRate){
        int res=0;
        float area = 0.0;
        float pointArea = 0.0;
        float arearate = 0.0;
        for(int i=1;i<peak_f.size();i++){
           pointArea = 0.0;
           area = (hist.at<float>(peak_f[i])+hist.at<float>(peak_f[i-1]))*abs(peak_f[i-1]-peak_f[i]+1)/2;
         for(int j=peak_f[i];j<=peak_f[i-1];j++) pointArea+=hist.at<float>(j);
           arearate = pointArea/area;
           if (arearate<areaRate){
             res = i;
             break;
           }
        }
   return res;
}

bool Thresh::StepDown(Mat hist,int l,int r){
    for(int i=l+1;i<=r;i++){
      if(hist.at<float>(i)>hist.at<float>(i-1)){
         if(hist.at<float>(i)-hist.at<float>(i-1)>10 || hist.at<float>(i+1)>=hist.at<float>(i))
           return false;
      }
   }
  return true;
}

bool Thresh::StepUp(Mat hist,int l,int r){
    for(int i=l+1;i<=r;i++){
      if(hist.at<float>(i)<hist.at<float>(i-1)){
        return false;
      }
   }
   if(hist.at<float>(r)-hist.at<float>(l)<r-l){return false;}
   return true;
}

int Thresh::FindValley(Mat hist, int left, int right){
   int l = left,r=right;
   int res;
   int i=1;
   while(i<r-l){
        if(!StepDown(hist,l,l+i)){
            if(!StepDown(hist,l+i+2,l+i+8)){
              res = l+i+2;break;
            }
            else
             {l = l+i+2; i=1;}
      }
     i++;
    }
   return res;
}


int Thresh::FindValley2(Mat hist, vector<int> peak_f, int res){
    int skip=0;
    int imin = 255;
    double mu = peak_f[res-1];
    double multi = hist.at<float>(mu)*sqrt(2)*PI*0.4937;
          for(int theta=1;theta<20;theta++){
              double pointsum = 0.0;
              double sum = multi*theta/3;
              for(int j=mu;j<=mu+theta;j++){
                 pointsum+=hist.at<float>(j);   
              }
             double rate = pointsum/sum;
             if(rate>0.9 && rate<1.1){
                skip = theta;
             }
          }
    imin = (int)(mu-skip);
    return imin;
}

int Thresh::FindValley3(Mat hist, vector<int> peak_f, int res){
   float lmin = 255;
    int imin2 = 255;
          for(int i=peak_f[res-1];i>peak_f[res];i--){
             if(hist.at<float>(i)<lmin){
               if(StepDown(hist,i-7,i)){
                  lmin = hist.at<float>(i);
                  imin2 = i;
                 }
               
             }
          }
        return imin2;
 }

Mat Thresh::drawHist(Mat hist){
      int histSize = hist.rows;
      // 创建直方图画布
      Mat histImage( histSize, histSize, CV_8U, Scalar(0,0,0) );
      /// 将直方图归一化到范围 [ 0, histImage.rows ]
      normalize(hist,hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
   	  
      /// 在直方图画布上画出直方图
      for(int i=0;i<histSize;i++){
         float binVal = hist.at<float>(i);
         int intensity = static_cast<int>(binVal);
      line(histImage,Point(i,histSize),Point(i,histSize-intensity),Scalar::all(255));
       }
    return histImage;
}
