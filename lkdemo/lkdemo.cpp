#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "blobs.h"
#include "lbp.hpp"
#include <iostream>
#include <ctype.h>
#include "tracker.h"

using namespace cv;
using namespace std;
Cblobs blobs;
Vtracker tracker; 
static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
MatND HistA, HistB, HistL, HistP;

int main( int argc, char** argv )
{
    help();
	Mat frame,mask;
	Scalar mauXanh = Scalar(0, 255, 0);
	BackgroundSubtractorMOG2 bgSubtractor(105,16,false);
	int frameNum=0;
	IplImage* out;
	IplImage* binary;
	IplImage* img;
	Mat resize_blur_Img; 
	int heightImg,widthImg;
    VideoCapture cap("G:\\xu ly anh\\nhan dang bien so xe\\video\\20170228-090037.mp4");
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;
    

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }


    Mat gray, prevGray, image;

 
	Rect box;
	int fontFace = CV_FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.8;
	int thickness = 2;
	int baseline=0;
	unsigned long numframe = 0;
	stringstream ss;
	tracker = Vtracker();
    for(;;)
    {
		int A = getTickCount();
        Mat frame;
		int numadd;
        cap >> frame;
        if(frame.empty())
            break;
		numframe++;
		resize(frame, resize_blur_Img, Size(frame.size().width, frame.size().height)); 
		heightImg=resize_blur_Img.size().height;
		widthImg=resize_blur_Img.size().width;
		bgSubtractor(resize_blur_Img,mask,0.001);
        resize_blur_Img.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
		binary = new IplImage(mask);
		img = new IplImage(resize_blur_Img);
		out= new IplImage(mask);
		imshow("mask",mask);
		if (numframe > 200) {
			blobs.BlobAnalysis(binary, 0, 0, binary->width, binary->height, 0, 200);
			blobs.BlobExclude(BLOBCIRCULARITY, 1.1, 5);
			blobs.BlobExclude(BLOBCIRCULARITY, .1, .9);
			printf("Znaleziono %d blobow", blobs.BlobCount);
			blobs.PrintRegionDataArray(1);
			line(image,cvPoint(0,10),cvPoint(image.cols,10),mauXanh);
			line(image,cvPoint(0,image.rows-10),cvPoint(image.cols,image.rows-10),mauXanh);	

			numadd = 0;
			for (int i = 1; i <= blobs.BlobCount; i++) {
				box = Rect(cvPoint(blobs.RegionData[i][BLOBMINX], blobs.RegionData[i][BLOBMINY]), cvPoint(blobs.RegionData[i][BLOBMAXX], blobs.RegionData[i][BLOBMAXY]));
				point = Point2f((float)box.x + box.width / 2, (float)box.y + box.height / 2);
				if (box.y > 10 && box.y + box.height < resize_blur_Img.rows - 10){
				rectangle(image, box.tl(), box.br(), Scalar(0, 0, 255), 1, 8, 0);
				numadd++;
				tracker.addRemovePt.push_back(true);
				Mat temA = resize_blur_Img(Rect(box.tl(), box.br()));
				Mat temMask = mask(Rect(box.tl(), box.br()));
				int update_num = 0;
				float tem;
				vector<float> hist_min;
				HistA = tracker.histImg(temA, temMask);//tinh histogram cua 1 doi tuong hien tai sau do tim 1 xe co gtri his phu hop roi
				//HistL = histlbp(temA);
				if (tracker.points[0].empty() != 1) {//cap nhap lai doi tuong, neu khong co se gan thanh doi tuong moi

					for (int j = 0; j < tracker.points[0].size(); j++) {//tim doi tuong hien tai la cua xe nao					
						//HistB = histImg(imgA[j], maskA[j]);//tinh histogram cua tung xe
						//HistP = histlbp(imgA[j]);
						tracker.b1 = compareHist(HistA, tracker.histId[tracker.lablecar[j]], CV_COMP_BHATTACHARYYA);
						//b2 = compareHist(HistP, HistL, CV_COMP_BHATTACHARYYA);

						update_num = j;//ghi gia tri xe thu bao nhieu co gia tri histogram nho nhat

						//cap nhap lai dac diem cua xe
						if (box.contains(tracker.points[0][j]) && tracker.b1 < 0.4) {//hoac neu doi tuong chua 1 diem cua xe thi cap nhap						
							tracker.bound[update_num] = box;
							tracker.points[0][update_num] = point;
							tracker.addRemovePt[numadd-1] = false;
							tracker.imgA[update_num] = temA;
							tracker.maskA[update_num] = temMask;
							tracker.histId[tracker.lablecar[j]] = HistA;

							vector<float> minhis;
							float tem;
							int updatenumhist = 0;

						}
					}
				}
			}

			}

			if (!tracker.points[0].empty())
			{
				vector<uchar> status;
				vector<float> err;

				calcOpticalFlowPyrLK(prevGray, gray, tracker.points[0], tracker.points[1], status, err, winSize,
					3, termcrit, 0, 0.001);
				size_t i, k, t;
				int dispoint, num = 0;

				for (i = t = 0; i < tracker.points[0].size(); i++)
				{

					if (!status[i])
						continue;
					int numlabel = 0, num = 0;
					float histmin, tem, prev;

					if (tracker.bound[i].y + tracker.bound[i].height > resize_blur_Img.rows - 10 || tracker.bound[i].y <10||
						tracker.bound[i].x + tracker.bound[i].width > resize_blur_Img.cols - 10 || tracker.bound[i].x < 10) {
						tracker.new_lable[tracker.lablecar[i]] = 0;
					}
					tracker.points[0][t] = tracker.points[0][i];
					tracker.points[1][t] = tracker.points[1][i];
					tracker.bound[t] = tracker.bound[i];
					tracker.imgA[t] = tracker.imgA[i];
					tracker.lablecar[t] = tracker.lablecar[i];
					tracker.maskA[t++] = tracker.maskA[i];
					tracker.bound[i].x = tracker.points[0][i].x - tracker.bound[i].width / 2;
					tracker.bound[i].y = tracker.points[0][i].y - tracker.bound[i].height / 2;

					// center the text			

					string text = to_string((_ULonglong)tracker.lablecar[i]);
					Size textSize = getTextSize(text, fontFace,
						fontScale, thickness, &baseline);
					Point textOrg(tracker.bound[i].x, tracker.bound[i].y + textSize.height);
					Point recpoint(tracker.bound[i].x + textSize.width, tracker.bound[i].y);
					baseline += thickness;
					//if(points[1][i].y<heightImg-80){
					//circle( image, points[1][i], 3, Scalar(255,0,0), -1, 8);

					rectangle(image, tracker.bound[i].tl(), tracker.bound[i].br(), mauXanh, 1, 8, 0);
					if (tracker.new_lable[tracker.lablecar[i]] == 1) {
						rectangle(image, textOrg, recpoint, mauXanh, -1, 8, 0);
						putText(image, text, textOrg, fontFace, fontScale,
							Scalar::all(255), thickness, 8);
					}
					line(image, tracker.points[0][i], tracker.points[1][i], mauXanh, 2, 8, 0);
				}
				tracker.imgB.clear();
				tracker.maskB.clear();
				tracker.points[1].resize(t);
				tracker.points[0].resize(t);
				tracker.bound.resize(t);
				tracker.imgA.resize(t);
				tracker.maskA.resize(t);
				tracker.imgB = tracker.imgA;
				tracker.maskB = tracker.maskA;
				tracker.lablecar.resize(t);
			}
			numadd = 0;
			for (int i = 1; i <= blobs.BlobCount; i++) {
				box = Rect(cvPoint(blobs.RegionData[i][BLOBMINX], blobs.RegionData[i][BLOBMINY]), cvPoint(blobs.RegionData[i][BLOBMAXX], blobs.RegionData[i][BLOBMAXY]));
				point = Point2f((float)box.x + box.width / 2, (float)box.y + box.height / 2);
				if (box.y > 10 && box.y + box.height < resize_blur_Img.rows - 10) {
					if (tracker.addRemovePt[numadd] && tracker.points[1].size() < (size_t)MAX_COUNT) {//them doi tuong
						vector<Point2f> tmp;
						Mat temA = resize_blur_Img(Rect(box.tl(), box.br()));
						Mat temMask = mask(Rect(box.tl(), box.br()));
						tracker.imgTem.push_back(temA);
						tracker.imgA.push_back(temA);
						tracker.maskA.push_back(temMask);
						tracker.bound.push_back(box);
						tmp.push_back(point);
						cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
						tracker.points[1].push_back(tmp[0]);
						tracker.histId.push_back(tracker.histImg(temA, temMask));
						//histLbpId.push_back(histlbp(temA));
						tracker.new_lable.push_back(1);
						tracker.lablecar.push_back(tracker.numlable);
						tracker.numlable++;
						tracker.addRemovePt[numadd] = false;
					}
					numadd++;
				}
			}
			tracker.addRemovePt.clear();
		}
			int B = getTickCount();			
			double time_period =  getTickFrequency()/(B-A);
			string framerate = to_string((_ULonglong)time_period)+" frames/s";
			Size textSize = getTextSize(framerate, fontFace,
				fontScale, thickness, &baseline);
			Point textOrg1(resize_blur_Img.cols - textSize.width - 9, textSize.height+15);
			baseline += thickness;
			rectangle(image, Point(resize_blur_Img.cols-textSize.width-10, textSize.height*2 + 40), Point(resize_blur_Img.cols, 10), mauXanh, -1, 8, 0);			

			string numFrame = "Frames: "+to_string((_ULonglong)numframe) ;
			textSize = getTextSize(numFrame, fontFace,
				fontScale, thickness, &baseline);
			Point textOrg2(resize_blur_Img.cols - textSize.width - 7, textSize.height*2 + 35);
			baseline += thickness;
			rectangle(image, Point(resize_blur_Img.cols - textSize.width - 10, textSize.height * 2 + 40), Point(resize_blur_Img.cols, 10), mauXanh, -1, 8, 0);
			putText(image, framerate, textOrg1, fontFace, fontScale,
				Scalar::all(255), thickness, 8);
			putText(image, numFrame, textOrg2, fontFace, fontScale,
				Scalar::all(255), thickness, 8);
			imshow("LK Demo", image);
        char c = (char)waitKey(10);

        std::swap(tracker.points[1], tracker.points[0]);
        cv::swap(prevGray, gray);
    }

    return 0;
}
