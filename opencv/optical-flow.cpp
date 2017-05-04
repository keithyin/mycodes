#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <Windows.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{

 // add your file name
 VideoCapture cap("yourFile.mp4");

 Mat flow, frame;
 // some faster than mat image container
 UMat  flowUmat, prevgray;

 for (;;){
    bool Is = cap.grab();
    if (Is == false) {
    // if video capture failed
    cout << "Video Capture Fail" << endl;
    break;
  }
 else {
    Mat img;
    Mat original;

    // capture frame from video file
    // 下面这行等价于cap>>img;
    cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
    //关于size构造函数Size(width, height)!!!!而Point是Point(x,y),不能向python那样思考了.
    resize(img, img, Size(640, 480));

    // save original for later
    img.copyTo(original);

    // just make current frame gray
    cvtColor(img, img, COLOR_BGR2GRAY);

    // For all optical flow you need a sequence of images.. Or at least 2 of them. Previous and current frame
    //if there is no current frame
    // go to this part and fill previous frame
    //else {
    // img.copyTo(prevgray);
    //   }
    // if previous frame is not empty.. There is a picture of previous frame. Do some optical flow alg.

    if (prevgray.empty() == false ) {

    /*计算光流特征,返回值给flowUmat,flowUmat是一个CV_32FC2,为什么两个channel呢?两个channel
    分别保存了 x轴 和 y轴 的值.(x,y)表示一个向量.

    0.4- image pyramid or simple image scale
    1 is number of pyramid layers. 1 mean that flow is calculated only from previous image.
    12 is win size.. Flow is computed over the window larger value is more robust to the noise.
    2 mean number of iteration of algorithm
    8 is polynomial degree expansion recommended value are 5 - 7
    1.2 standard deviation used to smooth used derivatives recommended values from 1.1 - 1,5
    */

    calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
    // copy Umat container to standard Mat
    flowUmat.copyTo(flow);

    // By y += 5, x += 5 you can specify the grid
    for (int y = 0; y < original.rows; y += 5) {
     for (int x = 0; x < original.cols; x += 5)
     {
        // get the flow from y, x position * 10 for better visibility
        //at中第一个函数表示第几行,第二个表示第几列......蒙逼
        const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
        // draw line at flow direction
        line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
        // draw initial point
        circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);

      }
     }
    // draw the results
    namedWindow("prew", WINDOW_AUTOSIZE);//这句可要可不要
    imshow("prew", original);

    // fill previous image again
    img.copyTo(prevgray);

    }
    else {
    // fill previous image in case prevgray.empty() == true
    img.copyTo(prevgray);

    }
    //如果想实时显示视频的话,必须要加上这句,要不然,显示不出来.
    int key1 = waitKey(20);
  }
 }
}
