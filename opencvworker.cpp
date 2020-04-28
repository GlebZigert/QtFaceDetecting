#include "opencvworker.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>

OpenCvWorker::OpenCvWorker(QObject *parent) :
    QObject(parent),
    status(false),
    toogleStream(false),
    binaryThresholdEnable(false),
    binaryThreshold(127)

{

    cap = new cv::VideoCapture();
    int retval	=	faceCascade.load("/opt/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");

}
OpenCvWorker::~OpenCvWorker()
{
    if(cap->isOpened())cap->release();
    delete cap;
}

void OpenCvWorker::receiveGrabFrame()
{
if (!toogleStream) return;
(*cap)>>_frameOriginal;
if (_frameOriginal.empty()) return;

process();

if (binaryThresholdEnable)
{
QImage output(
            (const unsigned char *)_frameProcessed.data,
            _frameProcessed.cols,
            _frameProcessed.rows,
            QImage::Format_Indexed8
            );
emit send_frame(output);
}
else
{
QImage output(
            (const unsigned char *)_frameProcessed.data,
            _frameProcessed.cols,
            _frameProcessed.rows,
            QImage::Format_RGB888
            );
emit send_frame(output);
}

}

void OpenCvWorker::face_detect()
{
    cv::Mat grey_image;
    cv::cvtColor(_frameOriginal, grey_image, CV_BGR2GRAY);
    cv::equalizeHist(grey_image, grey_image);

    std::vector<cv::Rect> faces;

    faceCascade.detectMultiScale(grey_image, faces, 1.1, 2,  0|CV_HAAR_SCALE_IMAGE,
                                 cv::Size(_frameOriginal.cols/4, _frameOriginal.rows/4)); // Minimum size of obj
    for( size_t i = 0; i < faces.size(); i++)
    {
        cv::rectangle(_frameOriginal, faces[i], cv::Scalar( 255, 0, 255 ));

    }

}

void OpenCvWorker::process()
{


    if (binaryThresholdEnable)
    {
      cv::cvtColor(_frameOriginal,_frameProcessed,cv::COLOR_BGR2GRAY);
      cv::threshold(_frameProcessed,_frameProcessed,binaryThreshold,255,cv::THRESH_BINARY);

    }
    else
    {
        face_detect();
       cv::cvtColor(_frameOriginal,_frameProcessed,cv::COLOR_BGR2RGB);

    }

}

void OpenCvWorker::checkIfDeviceAlreadyOpened(const int device)
{
    if(cap->isOpened())cap->release();
    cap->open(device);
}

void OpenCvWorker::receiveSetup(const int device)
{
checkIfDeviceAlreadyOpened(device);
if(!cap->isOpened())
    {
    status=false;
    return;

    }
status=true;
}

void OpenCvWorker::receiveToggleStream()
{
toogleStream=!toogleStream;
}

void OpenCvWorker::receiveEnableBinaryThreshold()
{
binaryThresholdEnable=!binaryThresholdEnable;
}

void OpenCvWorker::receiveBinaryThreshold(int threshold)
{
binaryThreshold=threshold;
qDebug()<<binaryThreshold;
}









