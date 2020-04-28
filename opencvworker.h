#ifndef OPENCVWORKER_H
#define OPENCVWORKER_H

#include <QObject>
#include <QImage>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class OpenCvWorker : public QObject
{
    Q_OBJECT
public:
    explicit OpenCvWorker(QObject *parent = nullptr);
    ~OpenCvWorker();

private:
    cv::Mat _frameOriginal;
    cv::Mat _frameProcessed;
    cv::VideoCapture *cap;

    bool status;
    bool toogleStream;

    bool binaryThresholdEnable;
    int binaryThreshold;

    void process();

    cv::CascadeClassifier faceCascade;

    void face_detect();

    void checkIfDeviceAlreadyOpened(const int device);

signals:
    void send_frame(QImage FrameProcessed);

public slots:
    void receiveGrabFrame();
    void receiveSetup(const int device);
    void receiveToggleStream();
    void receiveEnableBinaryThreshold();
    void receiveBinaryThreshold(int threshold);


};

#endif // OPENCVWORKER_H
