#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp	>

#include <iostream>

#include "cvmat_serilization.h"

using namespace std;
using namespace cv;


bool getFeatureFromImg(Mat img, vector<float> &feature)
{
	HOGDescriptor hog;
	hog.winSize = Size(img.cols, img.rows);
	Size winStride = Size(16, 16);


	hog.compute(img, feature, winStride);
	return true;
}

bool getImgProj(Mat img,Mat mean,Mat eigenVec,vector<float> &feature_proj)
{

	resize(img,img,Size(64,64));
	vector<float> feature;
	getFeatureFromImg(img,feature);

	PCAProject(feature,mean,eigenVec,feature_proj);
	return true;
}
float predictModel(vector<float> sample,CvSVM &model)
{
	Mat sample_mat(1,sample.size(),CV_32FC1);

	for (int i=0;i<sample.size();i++)
		sample_mat.at<float>(i)=sample[i];
	float score=model.predict(sample_mat,true);
	//if (score!=-1)
	//	cout<<score<<endl;
	return -score;
}
bool genLoc(int w, int h, int sw,int sh,int step, vector<Rect> &Locs)
{
	Locs.clear();
	for (int i=0;i<w-sw+1;i+=step)
		for (int j=0;j<h-sh+1;j+=step)
		{
			Rect t(i,j,sw,sh);
			Locs.push_back(t);
		}
	return true;
}

bool slideWindowDet(Mat img,vector<Rect> locs, Mat mean, Mat eigenVec,CvSVM &model,vector<float> &score)
{
	int64 t;
	score.clear();
	for (int i=0;i<locs.size();i++)
	{
		vector<float> feature;
		Mat roi=img(locs[i]);
		t=getTickCount();
		getImgProj(roi,mean,eigenVec,feature);
		//cout<<"feature time: "<<(getTickCount()-t)*1.0/getTickFrequency()<<endl;
		t=getTickCount();
		score.push_back(predictModel(feature,model));
		//cout<<"predict time: "<<(getTickCount()-t)*1.0/getTickFrequency()<<endl;
	}
	
	return true;
}

bool slideWindowDet(Mat img,vector<Rect> locs,CvSVM &model,vector<float> &score)
{
	int64 t;
	score.clear();
	for (int i=0;i<locs.size();i++)
	{
		vector<float> feature;
		Mat roi=img(locs[i]);
		t=getTickCount();
		getFeatureFromImg(roi,feature);
		//cout<<"feature time: "<<(getTickCount()-t)*1.0/getTickFrequency()<<endl;
		t=getTickCount();
		score.push_back(predictModel(feature,model));
		//cout<<"predict time: "<<(getTickCount()-t)*1.0/getTickFrequency()<<endl;
	}
	
	return true;
}



bool drawWindow(Mat img,vector<Rect> locs,vector<float> score,Mat &crop_window)
{
	crop_window=img.clone();
	double minVal,maxVal;
	Point minLoc,maxLoc;
	minMaxLoc(score,&minVal,&maxVal,&minLoc,&maxLoc);
	for(int i=0;i<locs.size();i++)
		if (abs(score[i]-maxVal)<0.2*abs(maxVal))
			rectangle(crop_window,locs[i],Scalar(0,0,255));
	return true;

}
bool drawPotential(Mat img,vector<Rect> locs,vector<float> score,Mat &pot)
{
	Mat tmpimg;
	Mat scaled_sc=Mat::zeros(img.size(),CV_8UC1);
	cvtColor(img,tmpimg,CV_BGR2GRAY);
	
	pot=Mat::zeros(tmpimg.size(),CV_8UC3);

	//min-max normalization
	double min,max;
	cv::minMaxIdx(score,&min,&max);

	vector<uchar> tmp_sc;
	for (int i=0;i<locs.size();i++)
	{
		tmp_sc.push_back(uchar((score[i]-min)/(max-min)*255));
	}

	//equalization
	equalizeHist(tmp_sc,tmp_sc);
	
	for (int i=0;i<locs.size();i++)
	{
		scaled_sc.at<uchar>(locs[i].y+locs[i].height/2-1,locs[i].x+locs[i].width/2-1)=tmp_sc[i];
	}
	

	Mat src[]={tmpimg,tmpimg,scaled_sc};
	int from_to[]={0,0,1,1,2,2};
	mixChannels(src,3,&pot,1,from_to,3);
	return true;

}

int main(int argc,char * argv[])
{
	bool pca=false;


	CvSVM model;
	model.load("../../data/model/svm8.dat");
	std::cout << model.get_support_vector_count() << " " << model.get_var_count() << std::endl;
	model.optimize_linear_svm();
	std::cout << model.get_support_vector_count() << " " << model.get_var_count() <<  std::endl;
	//input a test image
	Mat input;
	input=imread("../../data/test/8.jpg");
	std::cout<<"image size : "<<input.cols<<","<<input.rows<<endl;

	//generate detection locations
	vector<Rect> locs;
	genLoc(input.cols,input.rows,64,64,4,locs);
	std::cout<<"possible locations: "<<locs.size()<<endl;
	//slide window detection
	vector<float> score;
	if (!pca)
		slideWindowDet(input,locs,model,score);
	
	//draw potential visualization
	Mat potential;
	drawPotential(input,locs,score,potential);
	imshow("potential",potential);
	
	//draw detection rects
	Mat crop_window;
	drawWindow(input,locs,score,crop_window);

	
	imshow("crop_window",crop_window);
	
	cvWaitKey();

}