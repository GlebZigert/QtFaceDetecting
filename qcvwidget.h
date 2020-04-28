#ifndef QCVWIDGET_H
#define QCVWIDGET_H

#include <QWidget>
#include <QThread>
#include <opencvworker.h>
#include <QDebug>

namespace Ui {
class QCvWidget;
}

class QCvWidget : public QWidget
{
    Q_OBJECT

public:
    explicit QCvWidget(QWidget *parent = 0);
    ~QCvWidget();

private:
    Ui::QCvWidget *ui;
    QThread *thread;




    void setup();

signals:
    void sendSetup(int device);
    void sendToggleStream();
private slots:
    void receiveFrame(QImage Frame);
     void receiveToggleStream();
};

#endif // QCVWIDGET_H
