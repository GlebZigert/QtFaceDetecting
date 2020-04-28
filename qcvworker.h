#ifndef QCVWORKER_H
#define QCVWORKER_H

#include <QObject>

class QCvWorker : public QObject
{
    Q_OBJECT
public:
    explicit QCvWorker(QObject *parent = nullptr);

signals:

public slots:
};

#endif // QCVWORKER_H