#include "tracker.h"
#include "opencv2/opencv.hpp"
Vtracker::Vtracker(){
	hbins = 30;
	sbins = 32;
	channels[] = { 0,  1 };
	histSize[] = { hbins, sbins };
	hranges[] = { 0, 180 };
	sranges[] = { 0, 255 };
	ranges[] = { hranges, sranges };
	numlable = 0;
};
MatND Vtracker::histImg(Mat img, Mat mask) {
	MatND HistA;
	Mat patch_HSV;
	cvtColor(img, patch_HSV, CV_BGR2HSV);
	calcHist(&patch_HSV, 1, channels, mask, // do not use mask  
		HistA, 2, histSize, ranges,
		true, // the histogram is uniform  
		false);
	normalize(HistA, HistA, 0, 255, CV_MINMAX);
	return HistA;
}

MatND Vtracker::histlbp(Mat img) {
	MatND HistL;
	Mat lbpa, dst;
	cvtColor(img, dst, CV_BGR2GRAY);
	lbpa = libfacerec::varlbp(dst, 1, 8);
	HistL = libfacerec::spatial_histogram(lbpa, 255);
	normalize(HistL, HistL, 0, 255, CV_MINMAX);
	return HistL;
}

float Vtracker::matching_level(MatND his1, MatND his2, MatND lbp1, MatND lbp2) {
	float b1, b2, ac;
	b1 = compareHist(his1, his2, CV_COMP_BHATTACHARYYA);
	b2 = compareHist(lbp1, lbp2, CV_COMP_BHATTACHARYYA);
	ac = b1*0.5 + b2*0.5;
	return ac;
}