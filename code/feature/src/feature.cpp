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

bool inputImg(string path,bool crop,int width,int height,Mat &img)
{
	img=imread(path);
	if (!img.data)
	{
		return false;	
	}
	if (crop)
	{
		boost::filesystem::path img_path(path);
		string crop_path=img_path.parent_path().string()+"/"+img_path.stem().string()+".txt";
		fstream file;
		file.open(crop_path.c_str(),std::ios::in);
		if (!file.is_open())
			return false;
		int lx,ly,rx,ry;
		file>>lx>>ly>>rx>>ry;
		file.close();
		img=img(Rect(lx,ly,rx-lx+1,ry-ly+1));
	}

	resize(img,img,Size(width,height));
//	imshow("img",img);
	//cvWaitKey(2);
	return true;
}

bool getImgDirs(string parentDir,std::vector<string>& imgDirs)
{
	boost::filesystem::path parentPath(parentDir);
	if (!boost::filesystem::exists(parentPath))
		return false;
	boost::filesystem::directory_iterator it(parentPath);
	boost::filesystem::directory_iterator end;
	for (;it!=end;it++)
	{
		if (boost::filesystem::is_directory(it->path()))
			imgDirs.push_back(it->path().string());
	}
	return true;
}

bool getImgPathFromImgDir(string imgDir,std::vector<string> &imgPaths)
{
	if (!boost::filesystem::exists(imgDir))
		return false;
	boost::filesystem::directory_iterator it(imgDir);
	boost::filesystem::directory_iterator end;
	for (;it!=end;it++)
	{
		if (it->path().extension()==".png"||it->path().extension()==".bmp"||it->path().extension()==".jpg"||it->path().extension()==".pgm")

			imgPaths.push_back(it->path().string());
	}
	return true;
}
bool getImgPathFromImgDirs(std::vector<string> imgDirs,std::vector<string> &imgPaths)
{
	for (int i=0;i<imgDirs.size();i++)
	{
		if (!getImgPathFromImgDir(imgDirs[i],imgPaths))
			return false;
	}
	return true;
}

bool getFeatureFromImg(Mat img,vector<float> &feature)
{
	HOGDescriptor hog;
	hog.winSize=Size(img.cols,img.rows);
	Size winStride=Size(16,16);
	

	hog.compute(img,feature,winStride);
	return true;
}

bool getFeaturesFromTrainImages(vector<string> imgPaths,bool crop,Mat &features)
{
	
	vector<vector<float>> features_vec;
	for (int i=0;i<imgPaths.size();i++)
	{
		Mat img;
		vector<float> feature;
		inputImg(imgPaths[i],crop,64,64,img);
		getFeatureFromImg(img,feature);
		features_vec.push_back(feature);
		cout<<"compute features "<<i<<"/"<<imgPaths.size()<<endl;
	}
	
	features=Mat(features_vec.size(),features_vec[0].size(),CV_32FC1);
	for (int i=0;i<features_vec.size();i++)
	{
		for (int j=0;j<features_vec[i].size();j++)
		{
			
			features.at<float>(i,j)=features_vec[i][j];
		}
		cout<<"transform to CV::MAT "<<i<<"/"<<features_vec.size()<<endl;
	}
	return true;
}

bool getPCAfromFeatures(Mat features,Mat &features_PCA,Mat &mean, Mat &eigenVecs)
{
	PCA pca(features,Mat(),CV_PCA_DATA_AS_ROW,0.8f);
	features_PCA=pca.project(features);
	mean=pca.mean;
	eigenVecs=pca.eigenvectors;
	return true;
}

bool getFeaturesFromTrainPath(string path,bool crop,Mat &features)
{
	vector<string> img_dirs,img_paths;
	
	getImgDirs(path,img_dirs);
	getImgPathFromImgDirs(img_dirs,img_paths);
	getFeaturesFromTrainImages(img_paths,crop,features);

	
	return true;
}

bool outputFeaturesPCA(Mat features_pca,int posN,int negN,string path)
{
	fstream file;
	file.open(path.c_str(),std::ios::out);

	if (!file.is_open())
		return false;

	for (int i=0;i<features_pca.rows;i++)
	{
		if (i<posN)
			file<<"+1 ";
		else
			file<<"-1 ";
		for (int j=0;j<features_pca.cols;j++)
			file<<j+1<<":"<<features_pca.at<float>(i,j)<<" ";
		file<<endl;
		file.flush();
		
		//cout<<"writing pca %"<<i*1.0/features_pca.cols<<endl;
	}
	return true;
}

bool trainModel(Mat Data,bool pca,int posN,int negN,CvSVM &model)
{
	string path;
	if (pca)
		path="data/svm_pca.dat";
	else
		path="data/svm.dat";
	if (boost::filesystem::exists(path))
	{
		cout<<"use pre_computed svm"<<endl;
		model.load(path.c_str());
		return true;
	}
	cout<<"training svm"<<endl;
	Mat response(posN+negN,1,CV_32FC1);
	for (int i=0;i<posN+negN;i++)
		response.at<float>(i)=i<posN?+1:-1;
	CvSVMParams params;
	params.kernel_type=CvSVM::RBF;
	model.train_auto(Data,response,Mat(),Mat(),params,10);
	model.save(path.c_str());
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
		if (abs(score[i]-maxVal)<0.1*abs(maxVal))
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
	Mat all_features,all_features_pca,mean,eigenVecs;
	int posN,negN;
	bool pca=true;
	if (boost::filesystem::exists("data/all_features.dat"))
	{
		cout<<"use pre_computed features!"<<endl;
		ifstream ifs("data/all_features.dat",std::ios::in|std::ios::binary);
		{
			boost::archive::binary_iarchive ia(ifs);
			ia>>all_features>>posN>>negN;
		}
		ifs.close();
		cout<<"pos# "<<posN<<" neg# "<<negN<<endl;
	}
	else
	{
		Mat pos_features,neg_features;
		cout<<"-------feature for pos images-------"<<endl;
		getFeaturesFromTrainPath(argv[1],true,pos_features);
		cout<<"-------feature for neg images-------"<<endl;
		getFeaturesFromTrainPath(argv[2],false,neg_features);

	

		
		posN=pos_features.rows;
		negN=neg_features.rows;
	

		vconcat(pos_features,neg_features,all_features);
		
		ofstream ofs("data/all_features.dat",std::ios::out|std::ios::binary);
		{
			boost::archive::binary_oarchive oa(ofs);
			oa<<all_features<<posN<<negN;
		}
		ofs.close();

	}
	cout<<"-------concatenate features: "<<all_features.rows<<","<<all_features.cols<<endl;

	if (pca)
	{
		if (boost::filesystem::exists("data/all_features_pca.dat"))
		{
			cout<<"use pre_computed features_pca!"<<endl;
			ifstream ifs("data/all_features_pca.dat",std::ios::in|std::ios::binary);
			{
				boost::archive::binary_iarchive ia(ifs);
				ia>>all_features_pca>>mean>>eigenVecs;
			}
			ifs.close();
		
		}
		else
		{
			cout<<"-------compute PCA for all images--------"<<endl;
			getPCAfromFeatures(all_features,all_features_pca,mean,eigenVecs);
			ofstream ofs("data/all_features_pca.dat",std::ios::out|std::ios::binary);
			{
				boost::archive::binary_oarchive oa(ofs);
				oa<<all_features_pca<<mean<<eigenVecs;
			}
			ofs.close();

		}
	}
	CvSVM model;
	if (pca)
		trainModel(all_features_pca,pca,posN,negN,model);
	else
		trainModel(all_features,pca,posN,negN,model);
	if (pca)
		cout<<"-------pca features: "<<all_features_pca.rows<<","<<all_features_pca.cols<<endl;


	//input a test image
	Mat input;
	input=imread("car15.jpg");
	cout<<"image size : "<<input.cols<<","<<input.rows<<endl;

	//generate detection locations
	vector<Rect> locs;
	genLoc(input.cols,input.rows,64,64,4,locs);
	cout<<"possible locations: "<<locs.size()<<endl;
	//slide window detection
	vector<float> score;
	if (!pca)
		slideWindowDet(input,locs,model,score);
	else
		slideWindowDet(input,locs,mean,eigenVecs,model,score);

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