#ifndef THRESH_H
#define THRESH_H

#include<opencv2/opencv.hpp>

using namespace cv;

class Thresh {
  public:
  ///@brief determine if it is descending from left to right 
  bool StepDown(Mat hist,int l,int r);
  ///@brief determine if it is increasing from left to right
  bool StepUp(Mat hist,int l, int r);
  ///@brief find the valley from left peak to right peak
  int FindValley(Mat hist, int left, int right);
  int FindValley2(Mat hist, vector<int> peak_f, int res);
  int FindValley3(Mat hist, vector<int> peak_f, int res);
  ///@brief step 1,find the peak that higher than left and right
  vector<int> StepOne(Mat hist);
  //@brief step 2,find the peak that higher than from left-gap and left+gap
  vector<int> StepTwo(Mat hist, vector<int> peak,int gap);
  ///@biref step 3,find the peak that higher than the thresh
  vector<int> StepThree(Mat hist, vector<int> peak_s,int thresh);
  ///@brief step 4,if two peak are closed less than peak_gap,keep one
  vector<int> StepFour(Mat hist, vector<int> peak_t,int peak_gap);
  ///@ biref step 5, using the areaRate to find the split point
  int StepFive(Mat hist, vector<int> peak_f,float areaRate);
  ///@ brief ,draw the hist image
  Mat drawHist(Mat hist);
};
#endif
