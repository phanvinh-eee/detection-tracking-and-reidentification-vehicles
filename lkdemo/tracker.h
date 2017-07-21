#pragma once

class Vtracker {
public:	
	vector<bool> addRemovePt;
	int hbins, sbins;
	int channels[2];
	int histSize[2];
	float hranges[2];
	float sranges[2];
	const float* ranges[2];
	int numlable;
	float b1, b2, ac;
	vector<Point2f> points[2];//khai bao bien
	vector<Rect> bound;

	vector<int> new_lable;
	vector<MatND> histId, histLbpId;
	Rect box;
	vector<Mat> imgA, imgB, maskA, maskB, imgTem;
	vector<int> lablecar;
	
	MatND HistA, HistB, HistL, HistP;
	Vtracker();
	MatND histlbp(Mat img);
	MatND histImg(Mat img, Mat mask);
	float matching_level(MatND his1, MatND his2, MatND lbp1, MatND lbp2);
};